#include "mgmdisplay.hpp"
#include "kernel_utils.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <drm/drm_fourcc.h>

GST_DEBUG_CATEGORY_STATIC(magma_display_debug);
#define GST_CAT_DEFAULT magma_display_debug

enum {
    PROP_0,
    PROP_SYNC,
};



G_DEFINE_TYPE(GstMagmaDisplay, gst_magma_display, GST_TYPE_BASE_SINK)

// ─── Page flip handler ──────────────────────────────────────────────
struct FlipData {
    GstMagmaDisplay* self;
    uint32_t old_fb_id;
};

static void page_flip_handler(int fd, unsigned int frame, unsigned int sec, unsigned int usec, void* data) {
    auto* flip = static_cast<FlipData*>(data);
    auto* self = flip->self;
    if (flip->old_fb_id)
        drmModeRmFB(self->drm_fd, flip->old_fb_id);
    self->flip_pending = FALSE;
    delete flip;
}

static void wait_flip(GstMagmaDisplay* self) {
    if (!self->flip_pending)
        return;
    drmEventContext evctx{};
    evctx.version = 3;
    evctx.page_flip_handler = page_flip_handler;
    struct pollfd pfd = {.fd = self->drm_fd, .events = POLLIN};
    while (self->flip_pending) {
        int r = poll(&pfd, 1, 3000);
        if (r <= 0) {
            GST_WARNING_OBJECT(self, "poll on DRM fd timed out (r=%d)", r);
            self->flip_pending = FALSE;
            return;
        }
        drmHandleEvent(self->drm_fd, &evctx);
    }
}

/**
 * @brief Create a DRM framebuffer from a DMABuf FD.
 *
 * Imports the buffer via drmPrimeFDToHandle, optionally tries
 * drmModeAddFB2WithModifiers on NV12 failure, and aligns pitch
 * to 256 bytes for hardware compatibility.
 *
 * @param self      Display element
 * @param dmabuf_fd DMABuf file descriptor (caller retains ownership)
 * @param fb_w      Framebuffer width
 * @param fb_h      Framebuffer height
 * @param pitch     Row stride in bytes (0 = auto)
 * @return DRM FB ID on success, -1 on failure
 */
static int fb_from_dmabuf(GstMagmaDisplay* self, int dmabuf_fd, int fb_w, int fb_h, uint32_t pitch) {
    uint32_t gem_handle = 0;
    int ret = drmPrimeFDToHandle(self->drm_fd, dmabuf_fd, &gem_handle);
    if (ret || !gem_handle) {
        GST_ERROR_OBJECT(self, "drmPrimeFDToHandle failed: %d", ret);
        return -1;
    }

    uint32_t fourcc = self->gbm_fourcc ? self->gbm_fourcc : DRM_FORMAT_NV12;
    uint32_t stride = pitch ? pitch : (uint32_t)self->gbm_stride;
    if (!stride) stride = (uint32_t)self->in_width;
    stride = (stride + 255) & ~255;

    uint32_t handles[4] = {gem_handle, gem_handle, 0, 0};
    uint32_t pitches[4] = {stride, stride, 0, 0};
    uint32_t offsets[4] = {0, 0, 0, 0};
    uint32_t fb_id = 0;

    if (fourcc == DRM_FORMAT_NV12) {
        offsets[1] = stride * self->in_height;
    }

    ret = drmModeAddFB2(self->drm_fd, fb_w, fb_h,
                        fourcc, handles, pitches, offsets, &fb_id, 0);
    if (ret) {
        const uint64_t mods[4] = {DRM_FORMAT_MOD_LINEAR, DRM_FORMAT_MOD_LINEAR, 0, 0};
        ret = drmModeAddFB2WithModifiers(self->drm_fd, fb_w, fb_h,
                                         fourcc, handles, pitches, offsets,
                                         mods, &fb_id, DRM_MODE_FB_MODIFIERS);
    }
    if (ret) {
        GST_ERROR_OBJECT(self, "drmModeAddFB2 failed (err=%d) — need DRM master + aligned pitch", ret);
        return -1;
    }
    return (int)fb_id;
}

/**
 * @brief Allocate a mode-sized XRGB8888 GBM BO and import it to HIP.
 *
 * Creates a scanout-capable BO at display resolution, exports an
 * FD, and imports it as HIP external memory for GPU copy access.
 *
 * @param self Display element
 * @return TRUE on success
 */
static gboolean create_gpu_dmabuf(GstMagmaDisplay* self) {
    // Use XRGB8888 at mode resolution (some drivers reject SetCrtc with undersized FB)
    int bw = self->mode.hdisplay;
    int bh = self->mode.vdisplay;
    self->gbm_bo = gbm_bo_create(self->gbm_dev, bw, bh,
                                  GBM_FORMAT_XRGB8888, GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
    if (!self->gbm_bo) {
        GST_ERROR_OBJECT(self, "gbm_bo_create(XRGB8888, %dx%d) failed", bw, bh);
        return FALSE;
    }
    self->gbm_fourcc = DRM_FORMAT_XRGB8888;

    self->gbm_stride = gbm_bo_get_stride(self->gbm_bo);
    self->gpu_size = self->gbm_stride * bh;
    GST_INFO_OBJECT(self, "GBM BO: %dx%d (video %dx%d) fourcc=0x%x stride=%d size=%zu",
                   bw, bh, self->in_width, self->in_height, self->gbm_fourcc, self->gbm_stride, self->gpu_size);

    int bo_fd = gbm_bo_get_fd(self->gbm_bo);
    if (bo_fd < 0) {
        GST_ERROR_OBJECT(self, "gbm_bo_get_fd failed");
        gbm_bo_destroy(self->gbm_bo);
        self->gbm_bo = nullptr;
        return FALSE;
    }

    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = bo_fd;
    desc.size = self->gpu_size;

    hipError_t herr = hipImportExternalMemory(&self->ext_mem, &desc);
    close(bo_fd);
    if (herr != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipImportExternalMemory failed: %s", hipGetErrorString(herr));
        gbm_bo_destroy(self->gbm_bo);
        self->gbm_bo = nullptr;
        return FALSE;
    }

    hipExternalMemoryBufferDesc bdesc{};
    bdesc.offset = 0;
    bdesc.size = self->gpu_size;

    herr = hipExternalMemoryGetMappedBuffer(&self->d_image, self->ext_mem, &bdesc);
    if (herr != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipExternalMemoryGetMappedBuffer failed: %s", hipGetErrorString(herr));
        hipDestroyExternalMemory(self->ext_mem);
        self->ext_mem = nullptr;
        gbm_bo_destroy(self->gbm_bo);
        self->gbm_bo = nullptr;
        return FALSE;
    }

    if (!self->hip_stream) {
        hipError_t e = hipStreamCreate(&self->hip_stream);
        if (e != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipStreamCreate failed: %s", hipGetErrorString(e));
            hipDestroyExternalMemory(self->ext_mem);
            self->ext_mem = nullptr;
            gbm_bo_destroy(self->gbm_bo);
            self->gbm_bo = nullptr;
            return FALSE;
        }
    }

    self->gpu_ready = TRUE;
    GST_INFO_OBJECT(self, "GPU DMABuf ready: fd=%d mapped=%p", bo_fd, (void*)self->d_image);
    return TRUE;
}

/**
 * @brief Open a DRM card, acquire master, and find a connected connector.
 *
 * Iterates /dev/dri/card0-3, calls drmSetMaster, and selects the
 * first connected connector with available modes. Stores connector
 * ID, CRTC ID, and preferred mode in self.
 *
 * @param self Display element
 * @return TRUE on success
 */
static gboolean open_drm(GstMagmaDisplay* self) {
    for (int i = 0; i < 4; i++) {
        char path[64];
        snprintf(path, sizeof(path), "/dev/dri/card%d", i);
        int fd = open(path, O_RDWR);
        if (fd < 0)
            continue;

        drmModeResPtr res = drmModeGetResources(fd);
        if (!res) {
            close(fd);
            continue;
        }

        if (drmSetMaster(fd) != 0) {
            // Can't become DRM master (compositor holds it) — skip
            drmModeFreeResources(res);
            close(fd);
            continue;
        }

        gboolean found = FALSE;
        for (int c = 0; c < res->count_connectors && !found; c++) {
            drmModeConnectorPtr conn = drmModeGetConnector(fd, res->connectors[c]);
            if (!conn || conn->connection != DRM_MODE_CONNECTED || conn->count_modes == 0) {
                if (conn) drmModeFreeConnector(conn);
                continue;
            }

            uint32_t crtc_id = 0;
            if (conn->encoder_id) {
                drmModeEncoderPtr enc = drmModeGetEncoder(fd, conn->encoder_id);
                if (enc) {
                    for (int cr = 0; cr < res->count_crtcs; cr++) {
                        if (enc->possible_crtcs & (1 << cr)) {
                            crtc_id = res->crtcs[cr];
                            break;
                        }
                    }
                    drmModeFreeEncoder(enc);
                }
            }
            if (!crtc_id && res->count_crtcs > 0)
                crtc_id = res->crtcs[0];
            if (!crtc_id) {
                drmModeFreeConnector(conn);
                continue;
            }

            self->connector_id = conn->connector_id;
            self->crtc_id = crtc_id;
            self->mode = conn->modes[0];
            self->saved_crtc = drmModeGetCrtc(fd, crtc_id);
            self->drm_fd = fd;

            drmModeFreeConnector(conn);
            found = TRUE;
            break;
        }

        drmModeFreeResources(res);
        if (found) {
            // Also open GBM on a render node
            for (int r = 0; r < 64; r++) {
                char rpath[64];
                snprintf(rpath, sizeof(rpath), "/dev/dri/renderD%d", 128 + r);
                int rfd = open(rpath, O_RDWR);
                if (rfd < 0) continue;
                drmVersionPtr ver = drmGetVersion(rfd);
                if (ver) {
                    self->gbm_dev = gbm_create_device(rfd);
                    drmFreeVersion(ver);
                    if (self->gbm_dev) {
                        GST_INFO_OBJECT(self, "GBM device from %s", rpath);
                        break;
                    }
                }
                close(rfd);
            }

            if (!self->gbm_dev) {
                GST_ERROR_OBJECT(self, "Failed to create GBM device");
                close(fd);
                return FALSE;
            }

            GST_INFO_OBJECT(self, "DRM: card%d connector=%d crtc=%d  %dx%d@%.2fHz",
                           i, self->connector_id, self->crtc_id,
                           self->mode.hdisplay, self->mode.vdisplay,
                           self->mode.vrefresh / 100.0);
            self->drm_initialized = TRUE;
            return TRUE;
        }
        close(fd);
    }

    GST_ERROR_OBJECT(self, "No usable DRM card found (need DRM master — run on separate VT)");
    return FALSE;
}

// ─── Properties ────────────────────────────────────────────────────
static void gst_magma_display_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
    GstMagmaDisplay* self = GST_MAGMA_DISPLAY(object);
    switch (prop_id) {
    case PROP_SYNC:
        self->sync = g_value_get_boolean(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_magma_display_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
    GstMagmaDisplay* self = GST_MAGMA_DISPLAY(object);
    switch (prop_id) {
    case PROP_SYNC:
        g_value_set_boolean(value, self->sync);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

// ─── start / stop ──────────────────────────────────────────────────
static gboolean gst_magma_display_start(GstBaseSink* sink) {
    GstMagmaDisplay* self = GST_MAGMA_DISPLAY(sink);
    return open_drm(self);
}

static gboolean gst_magma_display_stop(GstBaseSink* sink) {
    GstMagmaDisplay* self = GST_MAGMA_DISPLAY(sink);
    if (!self->drm_initialized)
        return TRUE;

    wait_flip(self);

    if (self->current_fb_id) {
        drmModeRmFB(self->drm_fd, self->current_fb_id);
        self->current_fb_id = 0;
    }

    if (self->saved_crtc) {
        drmModeSetCrtc(self->drm_fd, self->saved_crtc->crtc_id,
                       self->saved_crtc->buffer_id,
                       self->saved_crtc->x, self->saved_crtc->y,
                       &self->connector_id, 1,
                       &self->saved_crtc->mode);
        drmModeFreeCrtc(self->saved_crtc);
        self->saved_crtc = nullptr;
    }

    if (self->gbm_bo) {
        gbm_bo_destroy(self->gbm_bo);
        self->gbm_bo = nullptr;
    }
    if (self->ext_mem) {
        hipDestroyExternalMemory(self->ext_mem);
        self->ext_mem = nullptr;
    }
    if (self->gbm_dev) {
        gbm_device_destroy(self->gbm_dev);
        self->gbm_dev = nullptr;
    }
    if (self->hip_stream) {
        hipStreamDestroy(self->hip_stream);
        self->hip_stream = nullptr;
    }
    if (self->drm_fd >= 0) {
        drmDropMaster(self->drm_fd);
        close(self->drm_fd);
        self->drm_fd = -1;
    }

    self->gpu_ready = FALSE;
    self->drm_initialized = FALSE;
    GST_INFO_OBJECT(self, "DRM released, %lu frames displayed", (unsigned long)self->frames_rendered);
    return TRUE;
}

// ─── set_caps ──────────────────────────────────────────────────────
static gboolean gst_magma_display_set_caps(GstBaseSink* sink, GstCaps* caps) {
    GstMagmaDisplay* self = GST_MAGMA_DISPLAY(sink);
    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, caps)) {
        GST_ERROR_OBJECT(self, "failed to parse caps");
        return FALSE;
    }
    self->in_width = GST_VIDEO_INFO_WIDTH(&info);
    self->in_height = GST_VIDEO_INFO_HEIGHT(&info);
    GST_INFO_OBJECT(self, "configured %dx%d %s", self->in_width, self->in_height, GST_VIDEO_INFO_NAME(&info));
    return TRUE;
}

// ─── Display a frame ───────────────────────────────────────────────
/**
 * @brief Display a DRM framebuffer via SetCrtc or PageFlip.
 *
 * First frame: calls drmModeSetCrtc. Subsequent frames: calls
 * drmModePageFlip if sync is enabled, otherwise drmModeSetCrtc.
 * Old framebuffers are destroyed after flip.
 *
 * @param self  Display element
 * @param fb_id DRM framebuffer ID
 * @return GST_FLOW_OK on success
 */
static GstFlowReturn display_fb(GstMagmaDisplay* self, int fb_id) {
    if (fb_id < 0)
        return GST_FLOW_ERROR;

    if (!self->current_fb_id) {
        // First frame: set CRTC
        int ret = drmModeSetCrtc(self->drm_fd, self->crtc_id, fb_id,
                                 0, 0, &self->connector_id, 1, &self->mode);
        if (ret) {
            GST_ERROR_OBJECT(self, "drmModeSetCrtc failed: %d", ret);
            drmModeRmFB(self->drm_fd, fb_id);
            return GST_FLOW_ERROR;
        }
        self->current_fb_id = fb_id;
        return GST_FLOW_OK;
    }

    if (self->sync) {
        wait_flip(self);
        auto* fd = new FlipData{self, self->current_fb_id};
        int ret = drmModePageFlip(self->drm_fd, self->crtc_id, fb_id,
                                  DRM_MODE_PAGE_FLIP_EVENT, fd);
        if (ret) {
            GST_ERROR_OBJECT(self, "drmModePageFlip failed: %d", ret);
            delete fd;
            drmModeRmFB(self->drm_fd, fb_id);
            return GST_FLOW_ERROR;
        }
        self->current_fb_id = fb_id;
        self->flip_pending = TRUE;
    } else {
        if (self->flip_pending) {
            // Async mode: drop frame if previous flip still pending
            drmModeRmFB(self->drm_fd, fb_id);
            return GST_FLOW_OK;
        }
        uint32_t old = self->current_fb_id;
        int ret = drmModeSetCrtc(self->drm_fd, self->crtc_id, fb_id,
                                 0, 0, &self->connector_id, 1, &self->mode);
        if (ret) {
            GST_ERROR_OBJECT(self, "drmModeSetCrtc (async) failed: %d", ret);
            drmModeRmFB(self->drm_fd, fb_id);
            return GST_FLOW_ERROR;
        }
        self->current_fb_id = fb_id;
        if (old) drmModeRmFB(self->drm_fd, old);
    }

    self->frames_rendered++;
    return GST_FLOW_OK;
}

// ─── Copy GPU data in GBM BO to display ──────────────────────────────
/**
 * @brief Export the GBM BO as a DMABuf FD and display it.
 *
 * Called at the end of copy_nv12_to_gbm after data has been
 * copied/converted into the GBM BO.
 *
 * @param self Display element
 * @return GST_FLOW_OK on success
 */
static GstFlowReturn gbm_bo_to_display(GstMagmaDisplay* self) {
    int bo_fd = gbm_bo_get_fd(self->gbm_bo);
    if (bo_fd < 0) {
        GST_ERROR_OBJECT(self, "gbm_bo_get_fd failed");
        return GST_FLOW_ERROR;
    }
    int fb_id = fb_from_dmabuf(self, bo_fd, self->mode.hdisplay, self->mode.vdisplay, 0);
    close(bo_fd);
    return display_fb(self, fb_id);
}

// ─── Copy NV12 data to GBM BO (used by Path 2 & 3) ───────────────────
/**
 * @brief Copy and optionally convert NV12 data to the GBM BO.
 *
 * If the GBM BO is NV12, performs a direct 2D memcpy of Y and UV
 * planes. If XRGB8888, runs the JIT-compiled nv12_to_xrgb8888 GPU
 * kernel for format conversion.
 *
 * @param self       Display element
 * @param src        Source data pointer (HIP device or host)
 * @param src_stride Source row stride in bytes
 * @param kind       hipMemcpyKind (DeviceToDevice or HostToDevice)
 * @return GST_FLOW_OK on success
 */
static GstFlowReturn copy_nv12_to_gbm(GstMagmaDisplay* self, const void* src, guint src_stride, hipMemcpyKind kind) {
    guint8* d_dst = (guint8*)self->d_image;
    guint dst_stride = self->gbm_stride;
    guint h = self->in_height;

    if (self->gbm_fourcc == DRM_FORMAT_NV12) {
        hipError_t herr;
        if (kind == hipMemcpyHostToDevice) {
            herr = hipMemcpy2D(d_dst, dst_stride, src, src_stride,
                               src_stride, h, hipMemcpyHostToDevice);
        } else {
            herr = hipMemcpy2DAsync(d_dst, dst_stride, src, src_stride,
                                    src_stride, h, kind, self->hip_stream);
        }
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMemcpy2D(Y) failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }

        const void* uv_src = (const guint8*)src + src_stride * h;
        void* uv_dst = d_dst + dst_stride * h;
        if (kind == hipMemcpyHostToDevice) {
            herr = hipMemcpy2D(uv_dst, dst_stride, uv_src, src_stride,
                               src_stride, h / 2, hipMemcpyHostToDevice);
        } else {
            herr = hipMemcpy2DAsync(uv_dst, dst_stride, uv_src, src_stride,
                                    src_stride, h / 2, kind, self->hip_stream);
        }
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMemcpy2D(UV) failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }

        if (kind != hipMemcpyHostToDevice) {
            herr = hipStreamSynchronize(self->hip_stream);
            if (herr != hipSuccess)
                GST_WARNING_OBJECT(self, "hipStreamSynchronize: %s", hipGetErrorString(herr));
        }
        return gbm_bo_to_display(self);
    }

    // XRGB8888 path: NV12→RGB GPU conversion
    if (!self->nv12_to_rgb_func) {
        std::string kpath = std::string(MAGMA_KERNEL_SRC_DIR) + "/nv12_to_xrgb8888.hip";
        HipKernel k = compile_kernel(kpath.c_str(), "nv12_to_xrgb8888", nullptr);
        if (!k.func) {
            GST_ERROR_OBJECT(self, "failed to compile nv12_to_xrgb8888 kernel");
            return GST_FLOW_ERROR;
        }
        self->rgb_module = k.module;
        self->nv12_to_rgb_func = k.func;
    }

    // Copy Y and UV planes to temp GPU buffers, run kernel, write to GBM BO
    gsize y_size = (gsize)src_stride * h;
    gsize uv_size = (gsize)src_stride * h / 2;
    hipDeviceptr_t d_y = 0, d_uv = 0;
    hipError_t herr;

    herr = hipMalloc(&d_y, y_size);
    if (herr != hipSuccess) { GST_ERROR_OBJECT(self, "hipMalloc(Y) failed"); return GST_FLOW_ERROR; }
    herr = hipMalloc(&d_uv, uv_size);
    if (herr != hipSuccess) { hipFree(d_y); GST_ERROR_OBJECT(self, "hipMalloc(UV) failed"); return GST_FLOW_ERROR; }

    if (kind == hipMemcpyHostToDevice) {
        herr = hipMemcpy2D(d_y, src_stride, src, src_stride, src_stride, h, hipMemcpyHostToDevice);
        if (herr == hipSuccess)
            herr = hipMemcpy2D(d_uv, src_stride, (const guint8*)src + src_stride * h, src_stride,
                               src_stride, h / 2, hipMemcpyHostToDevice);
    } else {
        herr = hipMemcpy2DAsync(d_y, src_stride, src, src_stride, src_stride, h, kind, self->hip_stream);
        if (herr == hipSuccess)
            herr = hipMemcpy2DAsync(d_uv, src_stride, (const guint8*)src + src_stride * h, src_stride,
                                    src_stride, h / 2, kind, self->hip_stream);
    }
    if (herr != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipMemcpy2D(temp) failed: %s", hipGetErrorString(herr));
        hipFree(d_y); hipFree(d_uv);
        return GST_FLOW_ERROR;
    }
    if (kind != hipMemcpyHostToDevice)
        hipStreamSynchronize(self->hip_stream);

    // Launch kernel: nv12_to_xrgb8888(y, y_stride, uv, uv_stride, dst, dst_stride, w, h)
    int block_size = 16;
    int ksrc_stride = src_stride;
    int dst_stride_pixels = dst_stride / 4;
    void* args[] = {
        &d_y, &ksrc_stride, &d_uv, &ksrc_stride,
        &d_dst, &dst_stride_pixels,
        &self->in_width, &h
    };
    dim3 grid((self->in_width + block_size - 1) / block_size,
              (h + block_size - 1) / block_size);
    herr = hipModuleLaunchKernel(self->nv12_to_rgb_func,
                                  grid.x, grid.y, 1,
                                  block_size, block_size, 1,
                                  0, self->hip_stream, args, nullptr);
    if (herr != hipSuccess) {
        GST_ERROR_OBJECT(self, "nv12_to_xrgb8888 launch failed: %s", hipGetErrorString(herr));
        hipFree(d_y); hipFree(d_uv);
        return GST_FLOW_ERROR;
    }
    herr = hipStreamSynchronize(self->hip_stream);
    hipFree(d_y); hipFree(d_uv);
    if (herr != hipSuccess)
        GST_WARNING_OBJECT(self, "hipStreamSynchronize: %s", hipGetErrorString(herr));

    return gbm_bo_to_display(self);
}

/**
 * @brief Main render entry point.
 *
 * Dispatches to one of three paths depending on buffer memory type:
 *   1. DMABuf — import FD to HIP, convert NV12→XRGB8888, display via GBM BO
 *   2. MagmaHipMeta — copy HIP pointer to GBM BO with format conversion
 *   3. System memory — CPU→GPU copy then format conversion
 *
 * @param sink The base sink element
 * @param buf  Incoming video buffer
 * @return GST_FLOW_OK on success
 */
static GstFlowReturn gst_magma_display_render(GstBaseSink* sink, GstBuffer* buf) {
    GstMagmaDisplay* self = GST_MAGMA_DISPLAY(sink);

    GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
    guint stride = vmeta && vmeta->stride[0] > 0 ? (guint)vmeta->stride[0] : (guint)self->in_width;

    // Path 1: DMABuf memory — import to HIP, convert NV12→XRGB8888, display via GBM BO
    GstMemory* mem = gst_buffer_peek_memory(buf, 0);
    if (mem && gst_is_dmabuf_memory(mem)) {
        gint fd = gst_dmabuf_memory_get_fd(mem);
        if (fd < 0) {
            GST_ERROR_OBJECT(self, "invalid DMABuf fd");
            return GST_FLOW_ERROR;
        }
        int dup_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
        if (dup_fd < 0) {
            GST_ERROR_OBJECT(self, "fcntl(DUPFD) failed");
            return GST_FLOW_ERROR;
        }

        if (!self->gpu_ready && !create_gpu_dmabuf(self)) {
            close(dup_fd);
            GST_ERROR_OBJECT(self, "failed to create GPU DMABuf for DMABuf→HIP copy");
            return GST_FLOW_ERROR;
        }

        gsize buf_size = (gsize)stride * self->in_height * 3 / 2;
        hipExternalMemoryHandleDesc desc{};
        desc.type = hipExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = dup_fd;
        desc.size = buf_size;
        hipExternalMemory_t ext_mem;
        hipError_t herr = hipImportExternalMemory(&ext_mem, &desc);
        close(dup_fd);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipImportExternalMemory(DMABuf) failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }
        hipExternalMemoryBufferDesc bdesc{};
        bdesc.offset = 0;
        bdesc.size = buf_size;
        hipDeviceptr_t d_src = 0;
        herr = hipExternalMemoryGetMappedBuffer(&d_src, ext_mem, &bdesc);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipExternalMemoryGetMappedBuffer(DMABuf) failed: %s", hipGetErrorString(herr));
            hipDestroyExternalMemory(ext_mem);
            return GST_FLOW_ERROR;
        }
        GstFlowReturn gret = copy_nv12_to_gbm(self, (const void*)d_src, stride, hipMemcpyDeviceToDevice);
        hipDestroyExternalMemory(ext_mem);
        return gret;
    }

    // Path 2: MagmaHipMeta — GPU copy to GBM BO
    MagmaHipMeta* hmeta = magma_buffer_get_hip_meta(buf);
    if (hmeta) {
        if (!self->gpu_ready && !create_gpu_dmabuf(self)) {
            GST_ERROR_OBJECT(self, "failed to create GPU DMABuf for HIP copy");
            return GST_FLOW_ERROR;
        }
        return copy_nv12_to_gbm(self, (const void*)hmeta->d_ptr, stride, hipMemcpyDeviceToDevice);
    }

    // Path 3: System memory — CPU→GPU copy to GBM BO
    if (!self->gpu_ready && !create_gpu_dmabuf(self)) {
        GST_ERROR_OBJECT(self, "failed to create GPU DMABuf for sysmem copy");
        return GST_FLOW_ERROR;
    }

    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
        GST_ERROR_OBJECT(self, "gst_buffer_map failed for sysmem path");
        return GST_FLOW_ERROR;
    }
    GstFlowReturn gret = copy_nv12_to_gbm(self, map.data, stride, hipMemcpyHostToDevice);
    gst_buffer_unmap(buf, &map);
    return gret;
}

// ─── init / class_init / finalize ──────────────────────────────────
static void gst_magma_display_init(GstMagmaDisplay* self) {
    self->in_width = 0;
    self->in_height = 0;
    self->sync = TRUE;
    self->drm_fd = -1;
    self->crtc_id = 0;
    self->connector_id = 0;
    self->current_fb_id = 0;
    self->saved_crtc = nullptr;
    self->drm_initialized = FALSE;
    self->flip_pending = FALSE;
    self->frames_rendered = 0;

    self->gbm_dev = nullptr;
    self->gbm_bo = nullptr;
    self->gbm_stride = 0;
    self->gbm_fourcc = 0;
    self->ext_mem = nullptr;
    self->d_image = 0;
    self->gpu_size = 0;
    self->gpu_ready = FALSE;
    self->hip_stream = nullptr;
    self->rgb_module = nullptr;
    self->nv12_to_rgb_func = nullptr;
}

static void gst_magma_display_finalize(GObject* object) {
    GstMagmaDisplay* self = GST_MAGMA_DISPLAY(object);
    if (self->rgb_module) {
        (void)hipModuleUnload(self->rgb_module);
        self->rgb_module = nullptr;
        self->nv12_to_rgb_func = nullptr;
    }
    if (self->drm_initialized)
        gst_magma_display_stop(GST_BASE_SINK(self));
    G_OBJECT_CLASS(gst_magma_display_parent_class)->finalize(object);
}

static void gst_magma_display_class_init(GstMagmaDisplayClass* klass) {
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstBaseSinkClass* sink_class = GST_BASE_SINK_CLASS(klass);

    gobject_class->set_property = gst_magma_display_set_property;
    gobject_class->get_property = gst_magma_display_get_property;
    gobject_class->finalize = gst_magma_display_finalize;

    g_object_class_install_property(
        gobject_class, PROP_SYNC,
        g_param_spec_boolean("sync", "Sync to vblank",
                            "Wait for vertical blank before flipping (vsync)",
                            TRUE, G_PARAM_READWRITE));

    static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
        "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
        GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12;"
                        "video/x-raw,format=(string)NV12"));

    gst_element_class_add_static_pad_template(element_class, &sink_template);

    gst_element_class_set_static_metadata(element_class,
        "Magma Display", "Sink/Video",
        "DRM/KMS fullscreen video sink with MagmaHipMeta and DMABuf support",
        "Magma");

    sink_class->start = GST_DEBUG_FUNCPTR(gst_magma_display_start);
    sink_class->stop = GST_DEBUG_FUNCPTR(gst_magma_display_stop);
    sink_class->set_caps = GST_DEBUG_FUNCPTR(gst_magma_display_set_caps);
    sink_class->render = GST_DEBUG_FUNCPTR(gst_magma_display_render);

    GST_DEBUG_CATEGORY_INIT(magma_display_debug, "magma_display", 0, "Magma DRM/KMS Display Sink");
}

static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "mgmdisplay", GST_RANK_NONE, GST_TYPE_MAGMA_DISPLAY);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mgmdisplay,
    "Magma DRM/KMS Display Plugin", plugin_init,
    "0.1.0", "LGPL", "magma", "https://imeguras.eu.org")
