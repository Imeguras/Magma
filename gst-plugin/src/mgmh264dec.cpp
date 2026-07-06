#include "mgmh264dec.hpp"
#include "magma-meta.h"

// Local copy of RocVideoDecoder utility (modified for Magma)
#include "roc_video_dec.h"

#include <gst/allocators/gstdmabuf.h>
#include <hip/hiprtc.h>
#include <string>
#include <queue>
#include <cstring>
#include <vector>
#include <cstdint>

GST_DEBUG_CATEGORY_STATIC(magma_h264_dec_debug);
#define GST_CAT_DEFAULT magma_h264_dec_debug

G_DEFINE_TYPE(GstMagmaH264Dec, gst_magma_h264_dec, GST_TYPE_VIDEO_DECODER)

// ─── Release context for MagmaHipMeta ─────────────────────────────────
// Just frees the hipMalloc'd copy. The INTERNAL surface is released
// immediately after the copy, not when downstream frees the buffer.
struct HipReleaseCtx {
    hipDeviceptr_t d_ptr;
};

static void hip_release_func(void* data) {
    auto* ctx = static_cast<HipReleaseCtx*>(data);
    if (ctx->d_ptr)
        hipFree(ctx->d_ptr);
    delete ctx;
}

// ─── Pending frame queue helpers (linear array, no holes) ───────────
static void pending_push(GstMagmaH264Dec* self, GstVideoCodecFrame* frame, int64_t pts_roc) {
    if (self->pending_count >= MAX_PENDING_FRAMES) {
        GST_WARNING_OBJECT(self, "Pending frame queue overflow!");
        return;
    }
    PendingFrame* pf = &self->pending[self->pending_count];
    pf->frame = frame;
    pf->pts_roc = pts_roc;
    self->pending_count++;
}

static PendingFrame* pending_front(GstMagmaH264Dec* self) {
    if (self->pending_count == 0) return nullptr;
    return &self->pending[0];
}

static void pending_pop_front(GstMagmaH264Dec* self) {
    if (self->pending_count == 0) return;
    // Compact: move all entries forward
    self->pending_count--;
    for (gint i = 0; i < self->pending_count; i++)
        self->pending[i] = self->pending[i + 1];
}

static GstVideoCodecFrame* pending_match(GstMagmaH264Dec* self, int64_t pts_roc) {
    for (gint i = 0; i < self->pending_count; i++) {
        if (self->pending[i].pts_roc == pts_roc) {
            GstVideoCodecFrame* f = self->pending[i].frame;
            // Remove by compaction
            self->pending_count--;
            for (gint j = i; j < self->pending_count; j++)
                self->pending[j] = self->pending[j + 1];
            return f;
        }
    }
    return nullptr;
}

// ─── Finish a pending frame with the given output buffer ─────────────
static void finish_pending_frame(GstVideoDecoder* decoder,
                                  GstVideoCodecFrame* frame,
                                  GstBuffer* out_buf,
                                  int64_t pts_ns) {
    frame->pts = pts_ns;
    frame->output_buffer = out_buf;
    gst_video_decoder_finish_frame(decoder, frame);
}

// ─── Pad templates ─────────────────────────────────────────────────────
// We accept AVC/AVCC (length-prefixed) or byte-stream and convert to Annex B
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-h264, stream-format=(string)avc; "
                    "video/x-h264, stream-format=(string)byte-stream; "
                    "video/x-h264")
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw, format=NV12; "
                    "video/x-raw, format=I420; "
                    "video/x-raw(memory:DMABuf), format=NV12; "
                    "video/x-raw(memory:DMABuf), format=I420")
);

// ─── AVCC → Annex B conversion ───────────────────────────────────────
// rocDecode parser expects Annex B (start codes 0x00000001), but
// h264parse typically outputs AVC (4-byte length prefix) format.
// We convert here.

static void append_annex_b_nal(std::vector<uint8_t>& out,
                                const uint8_t* data, uint32_t size) {
    out.push_back(0x00); out.push_back(0x00);
    out.push_back(0x00); out.push_back(0x01);
    out.insert(out.end(), data, data + size);
}

static std::vector<uint8_t> convert_avcc_to_annex_b(
    const uint8_t* avcc_data, size_t avcc_size) {
    std::vector<uint8_t> out;
    size_t i = 0;
    while (i + 4 <= avcc_size) {
        uint32_t nal_size = ((uint32_t)avcc_data[i] << 24) |
                            ((uint32_t)avcc_data[i+1] << 16) |
                            ((uint32_t)avcc_data[i+2] << 8)  |
                             (uint32_t)avcc_data[i+3];
        i += 4;
        if (nal_size > 0 && i + nal_size <= avcc_size) {
            append_annex_b_nal(out, avcc_data + i, nal_size);
            i += nal_size;
        } else break;
    }
    return out;
}

// Decode AVCC codec_data (extradata) into Annex B bytes.
// codec_data format (ISO 14496-15):
//   byte 0: version (0x01)
//   byte 1: profile
//   byte 2: compatibility
//   byte 3: level
//   byte 4: (0xFC | (nal_length_size_minus_1 & 0x03))   — top 6 bits reserved, bottom 2 = lengthSizeMinusOne
//   byte 5: (0xE0 | (num_sps & 0x1F))                   — top 3 bits reserved, bottom 5 = num SPS
//   byte 6+: SPS NALs (2-byte length prefix each), then num_pps byte, then PPS NALs
static std::vector<uint8_t> codec_data_to_annex_b(const uint8_t* cd, size_t cd_size) {
    std::vector<uint8_t> out;
    if (!cd || cd_size < 7) return out;

    int num_sps = cd[5] & 0x1F;
    size_t pos = 6;
    for (int i = 0; i < num_sps && pos + 2 <= cd_size; i++) {
        uint16_t sps_size = ((uint16_t)cd[pos] << 8) | cd[pos+1];
        pos += 2;
        if (pos + sps_size > cd_size) break;
        append_annex_b_nal(out, cd + pos, sps_size);
        pos += sps_size;
    }
    if (pos >= cd_size) return out;
    int num_pps = cd[pos++] & 0x1F;
    for (int i = 0; i < num_pps && pos + 2 <= cd_size; i++) {
        uint16_t pps_size = ((uint16_t)cd[pos] << 8) | cd[pos+1];
        pos += 2;
        if (pos + pps_size > cd_size) break;
        append_annex_b_nal(out, cd + pos, pps_size);
        pos += pps_size;
    }
    return out;
}

// ─── start / stop ─────────────────────────────────────────────────────
static gboolean gst_magma_h264_dec_start(GstVideoDecoder* decoder) {
    auto* self = GST_MAGMA_H264_DEC(decoder);

    // Input from h264parse is already packetized (one frame per buffer)
    gst_video_decoder_set_packetized(decoder, TRUE);

    try {
        // INTERNAL mode: get raw decoder surface pointers. We do our own
        // pitch-normalized copy to a contiguous NV12 buffer.
        self->roc_decoder = new RocVideoDecoder(
            0,                              // device_id = 0 (first GPU)
            OUT_SURFACE_MEM_DEV_INTERNAL,   // raw decoder surfaces
            rocDecVideoCodec_AVC,           // H.264
            false,                          // allow B-frame reordering
            nullptr,                        // crop rect
            false,                          // extract SEI
            0                               // display delay
        );

        self->width = 0;
        self->height = 0;
        self->configured = FALSE;

        GST_INFO_OBJECT(self, "rocDecode decoder created");
        return TRUE;
    } catch (const std::exception& e) {
        GST_ERROR_OBJECT(self, "Failed to create RocVideoDecoder: %s", e.what());
        return FALSE;
    }
}

static gboolean gst_magma_h264_dec_stop(GstVideoDecoder* decoder) {
    auto* self = GST_MAGMA_H264_DEC(decoder);

    self->pending_count = 0;

    if (self->roc_decoder) {
        delete static_cast<RocVideoDecoder*>(self->roc_decoder);
        self->roc_decoder = nullptr;
    }

    g_free(self->codec_data);
    self->codec_data = nullptr;
    self->codec_data_size = 0;

    return TRUE;
}

// ─── set_format — called when caps are negotiated ───────────────────
static gboolean gst_magma_h264_dec_set_format(GstVideoDecoder* decoder,
                                               GstVideoCodecState* state) {
    auto* self = GST_MAGMA_H264_DEC(decoder);

    GstCaps* caps = state->caps;
    GstStructure* s = gst_caps_get_structure(caps, 0);

    // Extract codec_data (SPS/PPS) from caps and convert to Annex B
    g_free(self->codec_data);
    self->codec_data = nullptr;
    self->codec_data_size = 0;

    const GValue* cd_val = gst_structure_get_value(s, "codec_data");
    if (cd_val && G_VALUE_HOLDS(cd_val, GST_TYPE_BUFFER)) {
        GstBuffer* cd_buf = gst_value_get_buffer(cd_val);
        GstMapInfo cd_map;
        if (gst_buffer_map(cd_buf, &cd_map, GST_MAP_READ)) {
            auto anb = codec_data_to_annex_b(cd_map.data, cd_map.size);
            if (!anb.empty()) {
                self->codec_data = (uint8_t*)g_malloc(anb.size());
                memcpy(self->codec_data, anb.data(), anb.size());
                self->codec_data_size = anb.size();
                GST_INFO_OBJECT(self, "codec_data: %zu bytes → %zu bytes Annex B",
                                cd_map.size, anb.size());
            }
            gst_buffer_unmap(cd_buf, &cd_map);
        }
    }

    GstVideoCodecState* out = gst_video_decoder_set_output_state(
        decoder, GST_VIDEO_FORMAT_I420, 0, 0, state);
    gst_video_codec_state_unref(out);
    self->output_format = GST_VIDEO_FORMAT_I420;

    self->configured = FALSE;
    return TRUE;
}

// ─── HIP kernel: deinterleave NV12 UV plane → I420 separate U/V ────
// Compiled via hiprtc at runtime (avoids <<<>>> syntax needing hipcc).
static const char nv12_i420_kernel_src[] =
    "extern \"C\" __global__ void nv12_to_i420("
    "    unsigned char* dst_u, unsigned char* dst_v,"
    "    const unsigned char* src_uv, int src_pitch,"
    "    int width, int height) {"
    "  int x = blockIdx.x * blockDim.x + threadIdx.x;"
    "  int y = blockIdx.y * blockDim.y + threadIdx.y;"
    "  if (x >= width / 2 || y >= height / 2) return;"
    "  int off = y * src_pitch + x * 2;"
    "  dst_u[y * (width / 2) + x] = src_uv[off];"
    "  dst_v[y * (width / 2) + x] = src_uv[off + 1];"
    "}";

// Cache the compiled kernel module+function
static hipModule_t nv12_i420_module = nullptr;
static hipFunction_t nv12_i420_func = nullptr;

static hipError_t ensure_nv12_i420_kernel(GstMagmaH264Dec* self) {
    if (nv12_i420_func) return hipSuccess;

    hiprtcProgram prog = nullptr;
    hiprtcResult rt = hiprtcCreateProgram(&prog, nv12_i420_kernel_src,
                                           "nv12_i420_kernel", 0, nullptr, nullptr);
    if (rt != HIPRTC_SUCCESS) {
        GST_ERROR_OBJECT(self, "hiprtcCreateProgram failed: %d", (int)rt);
        return hipErrorUnknown;
    }

    const char* opts[] = {"--gpu-architecture=gfx1101"};
    rt = hiprtcCompileProgram(prog, 1, opts);
    if (rt != HIPRTC_SUCCESS) {
        size_t log_sz = 0;
        hiprtcGetProgramLogSize(prog, &log_sz);
        std::string log(log_sz, '\0');
        hiprtcGetProgramLog(prog, &log[0]);
        GST_ERROR_OBJECT(self, "hiprtc compile failed: %s", log.c_str());
        hiprtcDestroyProgram(&prog);
        return hipErrorUnknown;
    }

    size_t code_sz = 0;
    hiprtcGetCodeSize(prog, &code_sz);
    std::vector<char> code(code_sz);
    hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);

    hipError_t e = hipModuleLoadData(&nv12_i420_module, code.data());
    if (e != hipSuccess) return e;
    e = hipModuleGetFunction(&nv12_i420_func, nv12_i420_module, "nv12_to_i420");
    return e;
}

// ─── Create output GstBuffer from a decoded HIP frame ───────────────
// INTERNAL surfaces have separate Y/UV pointers with hardware pitch.
// We copy into a contiguous I420 or NV12 buffer in HIP memory, then
// immediately release the INTERNAL surface.
static GstBuffer* create_output_buffer(GstMagmaH264Dec* self,
                                        RocVideoDecoder* roc_dec,
                                        uint8_t* y_ptr, uint8_t* uv_ptr,
                                        uint32_t pitch_y, uint32_t pitch_uv,
                                        int64_t pts,
                                        int width, int height,
                                        GstVideoFormat out_fmt) {
    gboolean is_i420 = (out_fmt == GST_VIDEO_FORMAT_I420);
    size_t y_size = (size_t)width * height;
    size_t uv_size = (size_t)(width / 2) * (height / 2);
    size_t frame_bytes = y_size + (is_i420 ? 2 * uv_size : uv_size);
    hipDeviceptr_t d_frame = 0;
    hipError_t herr = hipMalloc(&d_frame, frame_bytes);
    if (herr != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipMalloc(%zu) failed: %s", frame_bytes, hipGetErrorString(herr));
        return nullptr;
    }

    uint8_t* d_y  = (uint8_t*)d_frame;
    uint8_t* d_uv = d_y + y_size;

    // Copy Y plane: pitch_y → width
    herr = hipMemcpy2D(d_y, width,
                       y_ptr, pitch_y,
                       width, height,
                       hipMemcpyDeviceToDevice);
    if (herr != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipMemcpy2D(Y) failed: %s", hipGetErrorString(herr));
        (void)hipFree(d_frame);
        return nullptr;
    }

    if (is_i420) {
        // I420: deinterleave UV → separate U and V planes
        herr = ensure_nv12_i420_kernel(self);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "nv12_to_i420 kernel init failed: %s", hipGetErrorString(herr));
            (void)hipFree(d_frame);
            return nullptr;
        }
        uint8_t* d_u = d_uv;
        uint8_t* d_v = d_u + uv_size;
        void* args[] = { &d_u, &d_v, &uv_ptr, &pitch_uv, &width, &height };
        int bx = 32, by = 16;
        dim3 grid((width / 2 + bx - 1) / bx, (height / 2 + by - 1) / by);
        herr = hipModuleLaunchKernel(nv12_i420_func,
                                      grid.x, grid.y, 1,
                                      bx, by, 1,
                                      0, nullptr, args, nullptr);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "nv12_to_i420 kernel launch failed: %s", hipGetErrorString(herr));
            (void)hipFree(d_frame);
            return nullptr;
        }
    } else {
        // NV12: copy UV plane directly (already interleaved from decoder)
        herr = hipMemcpy2D(d_uv, width,
                           uv_ptr, pitch_uv,
                           width, height / 2,
                           hipMemcpyDeviceToDevice);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMemcpy2D(UV) failed: %s", hipGetErrorString(herr));
            (void)hipFree(d_frame);
            return nullptr;
        }
    }

    // Wait for copies to complete
    herr = hipDeviceSynchronize();
    if (herr != hipSuccess) {
        GST_WARNING_OBJECT(self, "hipDeviceSynchronize: %s", hipGetErrorString(herr));
    }

    // Release the INTERNAL surface back to the decoder pool immediately.
    roc_dec->ReleaseFrame(pts);

    // Small host-side GstBuffer with our hipMalloc'd copy in MagmaHipMeta
    GstBuffer* buf = gst_buffer_new_and_alloc(16);
    gst_buffer_memset(buf, 0, 0, 16);

    auto* ctx = new HipReleaseCtx{ d_frame };
    magma_buffer_add_hip_meta(buf, d_frame, hip_release_func, ctx);

    return buf;
}

// ─── handle_frame — feeds bitstream, outputs decoded frames ────────
static GstFlowReturn gst_magma_h264_dec_handle_frame(GstVideoDecoder* decoder,
                                                       GstVideoCodecFrame* frame) {
    auto* self = GST_MAGMA_H264_DEC(decoder);

    GstBuffer* in_buf = frame->input_buffer;
    if (!in_buf) {
        GST_ERROR_OBJECT(self, "No input buffer");
        return GST_FLOW_ERROR;
    }

    GstMapInfo map;
    if (!gst_buffer_map(in_buf, &map, GST_MAP_READ)) {
        GST_ERROR_OBJECT(self, "Failed to map input buffer");
        return GST_FLOW_ERROR;
    }

    auto* roc_dec = static_cast<RocVideoDecoder*>(self->roc_decoder);

    GST_LOG_OBJECT(self, "Input buffer: %zu bytes, first bytes: %02x %02x %02x %02x %02x",
                   map.size,
                   map.size > 0 ? (unsigned)map.data[0] : 0,
                   map.size > 1 ? (unsigned)map.data[1] : 0,
                   map.size > 2 ? (unsigned)map.data[2] : 0,
                   map.size > 3 ? (unsigned)map.data[3] : 0,
                   map.size > 4 ? (unsigned)map.data[4] : 0);

    // Determine if data is in AVC (AVCC, length-prefixed) or Annex B (start codes)
    gboolean is_avcc = FALSE;
    if (map.size >= 4) {
        // Annex B: starts with 0x00 0x00 0x00 0x01 or 0x00 0x00 0x01
        bool is_annex_b =
            (map.data[0] == 0x00 && map.data[1] == 0x00 && map.data[2] == 0x01) ||
            (map.data[0] == 0x00 && map.data[1] == 0x00 && map.data[2] == 0x00 && map.data[3] == 0x01);
        is_avcc = !is_annex_b;
    }

    // Build the full bitstream for rocDecode: prepend codec_data (first frame only),
    // then convert AVCC → Annex B if needed
    std::vector<uint8_t> bitstream;
    if (is_avcc) {
        // Prepend codec_data on first frame or reconfig
        if (self->codec_data && self->codec_data_size > 0 && !self->configured) {
            bitstream.insert(bitstream.end(),
                             self->codec_data,
                             self->codec_data + self->codec_data_size);
        }
        auto converted = convert_avcc_to_annex_b(map.data, map.size);
        bitstream.insert(bitstream.end(), converted.begin(), converted.end());
    } else {
        bitstream.assign(map.data, map.data + map.size);
    }

    const uint8_t* data_ptr = bitstream.data();
    size_t data_size = bitstream.size();
    if (data_size == 0) {
        gst_buffer_unmap(in_buf, &map);
        return GST_FLOW_OK;
    }

    GST_LOG_OBJECT(self, "Feeding %zu bytes to decoder", data_size);

    // Feed to rocDecode
    int pkt_flags = ROCDEC_PKT_TIMESTAMP;
    if (GST_BUFFER_FLAG_IS_SET(in_buf, GST_BUFFER_FLAG_MARKER))
        pkt_flags |= ROCDEC_PKT_ENDOFPICTURE;

    int64_t pts_gst = frame->pts;  // nanoseconds
    int64_t pts_roc = pts_gst / 100; // rocDecode uses 10MHz (100ns units)

    int num_decoded = 0;
    try {
        roc_dec->DecodeFrame((uint8_t*)data_ptr, data_size, pkt_flags, pts_roc, &num_decoded);
        GST_LOG_OBJECT(self, "DecodeFrame returned, num_decoded=%d", num_decoded);
    } catch (const RocVideoDecodeException& e) {
        gst_buffer_unmap(in_buf, &map);
        GST_ERROR_OBJECT(self, "DecodeFrame failed: %s (err=%d)", e.what(), e.Geterror_code());
        return GST_FLOW_OK;
    } catch (const std::exception& e) {
        gst_buffer_unmap(in_buf, &map);
        GST_ERROR_OBJECT(self, "DecodeFrame failed: %s", e.what());
        return GST_FLOW_ERROR;
    }
    gst_buffer_unmap(in_buf, &map);

    // Update output resolution from decoder if it changed
    int w = (int)roc_dec->GetWidth();
    int h = (int)roc_dec->GetHeight();
    if (w > 0 && h > 0 && (!self->configured || w != self->width || h != self->height)) {
        self->width = w;
        self->height = h;
        // Query downstream caps to determine preferred output format
        GstCaps* nv12_query = gst_caps_new_simple("video/x-raw",
            "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, w, "height", G_TYPE_INT, h, NULL);
        GstCaps* downstream = gst_pad_peer_query_caps(decoder->srcpad, nv12_query);
        gboolean use_nv12 = downstream && !gst_caps_is_empty(downstream);
        if (downstream) gst_caps_unref(downstream);
        gst_caps_unref(nv12_query);

        GstVideoFormat fmt = use_nv12 ? GST_VIDEO_FORMAT_NV12 : GST_VIDEO_FORMAT_I420;
        GstVideoCodecState* out = gst_video_decoder_set_output_state(
            decoder, fmt, w, h, nullptr);
        gst_video_codec_state_unref(out);
        gst_video_decoder_negotiate(decoder);

        // Store the actual negotiated format
        GstVideoCodecState* out_state = gst_video_decoder_get_output_state(decoder);
        self->output_format = GST_VIDEO_INFO_FORMAT(&out_state->info);
        gst_video_codec_state_unref(out_state);
        GST_INFO_OBJECT(self, "Resolution set: %dx%d, format: %s",
            w, h, gst_video_format_to_string(self->output_format));

        self->configured = TRUE;
    }

    // Queue this input frame as pending — it will be matched to a decoder
    // output (possibly a future one, due to H.264 reordering).
    pending_push(self, frame, pts_roc);

    // Pull all available decoded frames, matching them to pending inputs
    while (true) {
        int64_t out_pts_roc = 0;
        uint8_t* uv_ptr = nullptr;
        uint32_t pitch_y = 0, pitch_uv = 0;
        uint8_t* y_ptr = roc_dec->GetFrame(&out_pts_roc, &uv_ptr, &pitch_y, &pitch_uv);
        if (!y_ptr)
            break;

        int64_t out_pts = out_pts_roc * 100;
        GstBuffer* out_buf = create_output_buffer(self, roc_dec,
                                                    y_ptr, uv_ptr,
                                                    pitch_y, pitch_uv,
                                                    out_pts_roc,
                                                    self->width, self->height,
                                                    self->output_format);

        // Find matching pending frame by PTS
        GstVideoCodecFrame* target = pending_match(self, out_pts_roc);
        if (!target) {
            // No PTS match — output belongs to the oldest pending frame
            PendingFrame* pf = pending_front(self);
            if (pf) {
                target = pf->frame;
                pending_pop_front(self);
            }
        }
        if (target) {
            finish_pending_frame(decoder, target, out_buf, out_pts);
        } else {
            GST_WARNING_OBJECT(self, "No pending frame for decoder output PTS=%ld", out_pts_roc);
            gst_buffer_unref(out_buf);
        }
    }

    // Always return OK — the current frame is queued (it'll be finished
    // when the decoder outputs its display-order frame later).
    return GST_FLOW_OK;
}

// ─── drain — flush all remaining decoded frames ─────────────────────
static GstFlowReturn gst_magma_h264_dec_drain(GstVideoDecoder* decoder) {
    auto* self = GST_MAGMA_H264_DEC(decoder);
    auto* roc_dec = static_cast<RocVideoDecoder*>(self->roc_decoder);

    // Flush the decoder: send null/EOF to output remaining frames
    roc_dec->DecodeFrame(nullptr, 0, 0, 0, 0);

    // Pull all flushed frames
    while (true) {
        int64_t out_pts_roc = 0;
        uint8_t* uv_ptr = nullptr;
        uint32_t pitch_y = 0, pitch_uv = 0;
        uint8_t* y_ptr = roc_dec->GetFrame(&out_pts_roc, &uv_ptr, &pitch_y, &pitch_uv);
        if (!y_ptr)
            break;

        int64_t out_pts = out_pts_roc * 100;
        GstBuffer* out_buf = create_output_buffer(self, roc_dec,
                                                    y_ptr, uv_ptr,
                                                    pitch_y, pitch_uv,
                                                    out_pts_roc,
                                                    self->width, self->height,
                                                    self->output_format);

        GstVideoCodecFrame* target = pending_match(self, out_pts_roc);
        if (!target) {
            PendingFrame* pf = pending_front(self);
            if (pf) {
                target = pf->frame;
                pending_pop_front(self);
            }
        }
        if (target) {
            finish_pending_frame(decoder, target, out_buf, out_pts);
        } else {
            GST_WARNING_OBJECT(self, "Drain: no pending frame for PTS=%ld", out_pts_roc);
            gst_buffer_unref(out_buf);
        }
    }

    // Finish any remaining pending frames with stub buffers
    while (self->pending_count > 0) {
        PendingFrame* pf = pending_front(self);
        GST_WARNING_OBJECT(self, "Drain: finishing unmatched frame PTS=%ld", pf->pts_roc);
        GstBuffer* stub = gst_buffer_new_and_alloc(16);
        gst_buffer_memset(stub, 0, 0, 16);
        finish_pending_frame(decoder, pf->frame, stub, pf->pts_roc * 100);
        pending_pop_front(self);
    }

    return GST_FLOW_OK;
}

// ─── flush ───────────────────────────────────────────────────────────
static gboolean gst_magma_h264_dec_flush(GstVideoDecoder* decoder) {
    auto* self = GST_MAGMA_H264_DEC(decoder);
    self->pending_count = 0;
    return TRUE;
}

// ─── class init ───────────────────────────────────────────────────────
static void gst_magma_h264_dec_class_init(GstMagmaH264DecClass* klass) {
    auto* decoder_class = GST_VIDEO_DECODER_CLASS(klass);
    auto* element_class = GST_ELEMENT_CLASS(klass);

    gst_element_class_set_metadata(element_class,
        "Magma H.264 Decoder",
        "Codec/Decoder/Video",
        "AMD ROCm-native H.264 decoder using rocDecode",
        "Magma Team");

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    decoder_class->start       = gst_magma_h264_dec_start;
    decoder_class->stop        = gst_magma_h264_dec_stop;
    decoder_class->set_format  = gst_magma_h264_dec_set_format;
    decoder_class->handle_frame = gst_magma_h264_dec_handle_frame;
    decoder_class->drain       = gst_magma_h264_dec_drain;
    decoder_class->flush       = gst_magma_h264_dec_flush;
}

// ─── instance init ────────────────────────────────────────────────────
static void gst_magma_h264_dec_init(GstMagmaH264Dec* self) {
    self->roc_decoder = nullptr;
    self->width = 0;
    self->height = 0;
    self->configured = FALSE;
    self->codec_data = nullptr;
    self->codec_data_size = 0;
    self->pending_count = 0;
}

static gboolean plugin_init(GstPlugin* plugin) {
    GST_DEBUG_CATEGORY_INIT(magma_h264_dec_debug, "mgmh264dec", 0,
                            "Magma H.264 Decoder");
    return gst_element_register(plugin, "mgmh264dec", GST_RANK_PRIMARY + 1,
                                GST_TYPE_MAGMA_H264_DEC);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mgmh264dec,
    "Magma H.264 Decoder Plugin", plugin_init, "0.1.0", "LGPL", "magma",
    "https://imeguras.eu.org")
