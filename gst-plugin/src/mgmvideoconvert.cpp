#include "mgmvideoconvert.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <xf86drm.h>
#include "kernel_utils.hpp"
#include "magma-meta.h"

/** --- GOBJECT / GSTREAMER STUFF --- */
GST_DEBUG_CATEGORY_STATIC(magma_videoconvert_debug);
#define GST_CAT_DEFAULT magma_videoconvert_debug

enum {
    PROP_0,
};

G_DEFINE_TYPE(GstMagmaVideoConvert, gst_magma_videoconvert, GST_TYPE_BASE_TRANSFORM)

/** --- GBM BACKED DMABUF ALLOCATOR --- */

static gboolean create_gpu_dmabuf(GstMagmaVideoConvert* self) {
    // 1) Open DRM render node
    const char* drm_path = nullptr;
    for (int i = 0; i < 64; i++) {
        char path[64];
        snprintf(path, sizeof(path), "/dev/dri/renderD%d", 128 + i);
        int fd = open(path, O_RDWR);
        if (fd < 0)
            continue;
        drmVersionPtr ver = drmGetVersion(fd);
        if (ver) {
            GST_INFO_OBJECT(self, "DRM node %s: %s", path, ver->name);
            drmFreeVersion(ver);
            self->drm_fd = fd;
            drm_path = path;
            break;
        }
        close(fd);
    }
    if (self->drm_fd < 0) {
        GST_ERROR_OBJECT(self, "No usable DRM render node found");
        return FALSE;
    }

    // 2) Create GBM device
    self->gbm_dev = gbm_create_device(self->drm_fd);
    if (!self->gbm_dev) {
        GST_ERROR_OBJECT(self, "gbm_create_device failed");
        close(self->drm_fd);
        self->drm_fd = -1;
        return FALSE;
    }
    GST_INFO_OBJECT(self, "GBM device created from %s", drm_path);

    // 3) Allocate GBM BO (R8 format, enough height for NV12 with padding)
    // gbm_bo_get_stride returns the aligned stride (e.g. 768 for width 640)
    // So total size = stride * height + stride * (height/2) = stride * height * 3/2
    guint bo_height = self->in_height * 3 / 2;
    self->gbm_bo = gbm_bo_create(self->gbm_dev, self->in_width, bo_height, GBM_FORMAT_R8, GBM_BO_USE_RENDERING);
    if (!self->gbm_bo) {
        GST_ERROR_OBJECT(self, "gbm_bo_create(%dx%d, R8) failed", self->in_width, bo_height);
        gbm_device_destroy(self->gbm_dev);
        self->gbm_dev = nullptr;
        close(self->drm_fd);
        self->drm_fd = -1;
        return FALSE;
    }

    self->gbm_stride = gbm_bo_get_stride(self->gbm_bo);
    self->gpu_size = self->gbm_stride * self->in_height * 3 / 2;
    GST_INFO_OBJECT(self, "GBM BO created: %dx%d stride=%d size=%zu", self->in_width, self->in_height, self->gbm_stride, self->gpu_size);

    // 4) Export BO as DMABuf fd
    int bo_fd = gbm_bo_get_fd(self->gbm_bo);
    if (bo_fd < 0) {
        GST_ERROR_OBJECT(self, "gbm_bo_get_fd failed");
        gbm_bo_destroy(self->gbm_bo);
        self->gbm_bo = nullptr;
        gbm_device_destroy(self->gbm_dev);
        self->gbm_dev = nullptr;
        close(self->drm_fd);
        self->drm_fd = -1;
        return FALSE;
    }

    // 5) Import DMABuf into HIP
    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = bo_fd;
    desc.size = self->gpu_size;

    hipError_t err = hipImportExternalMemory(&self->ext_mem, &desc);
    close(bo_fd);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipImportExternalMemory failed: %s", hipGetErrorString(err));
        gbm_bo_destroy(self->gbm_bo);
        self->gbm_bo = nullptr;
        gbm_device_destroy(self->gbm_dev);
        self->gbm_dev = nullptr;
        close(self->drm_fd);
        self->drm_fd = -1;
        return FALSE;
    }

    hipExternalMemoryBufferDesc bdesc{};
    bdesc.offset = 0;
    bdesc.size = self->gpu_size;

    err = hipExternalMemoryGetMappedBuffer(&self->d_image, self->ext_mem, &bdesc);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipExternalMemoryGetMappedBuffer failed: %s", hipGetErrorString(err));
        (void)hipDestroyExternalMemory(self->ext_mem);
        self->ext_mem = nullptr;
        gbm_bo_destroy(self->gbm_bo);
        self->gbm_bo = nullptr;
        gbm_device_destroy(self->gbm_dev);
        self->gbm_dev = nullptr;
        close(self->drm_fd);
        self->drm_fd = -1;
        return FALSE;
    }

    self->gpu_ready = TRUE;
    GST_INFO_OBJECT(self, "GPU DMABuf ready: fd=%d mapped=%p", bo_fd, (void*)self->d_image);
    return TRUE;
}

/** --- PAD TEMPLATES --- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink",
                                                                    GST_PAD_SINK,
                                                                    GST_PAD_ALWAYS,
                                                                    GST_STATIC_CAPS("video/x-raw,format=(string)NV12;"
                                                                                    "video/x-raw,format=(string)I420;"
                                                                                    "video/x-raw(memory:DMABuf),format=(string)I420;"
                                                                                    "video/x-raw(memory:DMABuf),format=(string)NV12"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src",
                                                                   GST_PAD_SRC,
                                                                   GST_PAD_ALWAYS,
                                                                   GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12; "
                                                                                   "video/x-raw,format=(string)I420;"
                                                                                   "video/x-raw(memory:DMABuf),format=(string)I420;"
                                                                                   "video/x-raw,format=(string)NV12"));

/** --- INIT --- */
static void gst_magma_videoconvert_init(GstMagmaVideoConvert* self) {
    self->in_width = 0;
    self->in_height = 0;
    self->in_stride = 0;
    self->in_format = GST_VIDEO_FORMAT_UNKNOWN;
    self->out_format = GST_VIDEO_FORMAT_UNKNOWN;
    self->convert = nullptr;

    self->drm_fd = -1;
    self->gbm_dev = nullptr;
    self->gbm_bo = nullptr;
    self->gbm_stride = 0;

    self->ext_mem = nullptr;
    self->d_image = 0;
    self->gpu_size = 0;
    self->hip_stream = nullptr;
    self->gpu_ready = FALSE;
    self->kernel_module = nullptr;
    self->kernel_func = nullptr;
    self->kernel_ready = FALSE;

    if (hipStreamCreate(&self->hip_stream) != hipSuccess)
        self->hip_stream = nullptr;
}

/** --- FINALIZE --- */
static void gst_magma_videoconvert_finalize(GObject* object) {
    GstMagmaVideoConvert* self = GST_MAGMA_VIDEOCONVERT(object);

    if (self->hip_stream) {
        (void)hipStreamDestroy(self->hip_stream);
        self->hip_stream = nullptr;
    }
    if (self->ext_mem) {
        (void)hipDestroyExternalMemory(self->ext_mem);
        self->ext_mem = nullptr;
        self->d_image = 0;
    }
    if (self->gbm_bo) {
        gbm_bo_destroy(self->gbm_bo);
        self->gbm_bo = nullptr;
    }
    if (self->gbm_dev) {
        gbm_device_destroy(self->gbm_dev);
        self->gbm_dev = nullptr;
    }
    if (self->kernel_module) {
        (void)hipModuleUnload(self->kernel_module);
        self->kernel_module = nullptr;
        self->kernel_func = nullptr;
    }
    if (self->drm_fd >= 0) {
        close(self->drm_fd);
        self->drm_fd = -1;
    }

    G_OBJECT_CLASS(gst_magma_videoconvert_parent_class)->finalize(object);
}

/** --- PROPERTIES --- */
static void gst_magma_videoconvert_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
    switch (prop_id) {
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_magma_videoconvert_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
    switch (prop_id) {
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

// ─── Converter forward declarations ─────────────────────────────────
static GstFlowReturn conv_sys_nv12_to_dmabuf_nv12(GstMagmaVideoConvert*, GstBuffer*, GstBuffer*);
static GstFlowReturn conv_dmabuf_nv12_to_sys_nv12(GstMagmaVideoConvert*, GstBuffer*, GstBuffer*);
static GstFlowReturn conv_sys_i420_to_dmabuf_nv12(GstMagmaVideoConvert*, GstBuffer*, GstBuffer*);
static GstFlowReturn conv_sys_i420_to_sys_nv12(GstMagmaVideoConvert*, GstBuffer*, GstBuffer*);

// ─── Dispatch table ─────────────────────────────────────────────────
static const MgmConvertEntry convert_table[] = {
    {MGM_MEM_SYSTEM, GST_VIDEO_FORMAT_NV12, MGM_MEM_DMABUF, GST_VIDEO_FORMAT_NV12, conv_sys_nv12_to_dmabuf_nv12},
    {MGM_MEM_DMABUF, GST_VIDEO_FORMAT_NV12, MGM_MEM_SYSTEM, GST_VIDEO_FORMAT_NV12, conv_dmabuf_nv12_to_sys_nv12},
    {MGM_MEM_SYSTEM, GST_VIDEO_FORMAT_I420, MGM_MEM_DMABUF, GST_VIDEO_FORMAT_NV12, conv_sys_i420_to_dmabuf_nv12},
    {MGM_MEM_DMABUF, GST_VIDEO_FORMAT_I420, MGM_MEM_DMABUF, GST_VIDEO_FORMAT_NV12, conv_sys_i420_to_dmabuf_nv12},
    {MGM_MEM_SYSTEM, GST_VIDEO_FORMAT_I420, MGM_MEM_SYSTEM, GST_VIDEO_FORMAT_NV12, conv_sys_i420_to_sys_nv12},
};
static const int convert_table_count = sizeof(convert_table) / sizeof(convert_table[0]);

// ─── Helpers: detect memory type from caps ──────────────────────────
static MgmMemType mem_type_from_caps(GstCaps* caps) {
    GstCapsFeatures* f = gst_caps_get_features(caps, 0);
    if (gst_caps_features_contains(f, GST_CAPS_FEATURE_MEMORY_DMABUF))
        return MGM_MEM_DMABUF;
    return MGM_MEM_SYSTEM;
}

/** --- CAPS NEGOTIATION --- */
static GstCaps* gst_magma_videoconvert_transform_caps(GstBaseTransform* trans, GstPadDirection direction, GstCaps* caps, GstCaps* filter) {

    GstCaps* orig = gst_caps_copy(caps);
    GstCaps* other = gst_caps_copy(caps);

    // Toggle DMABuf/system memory type on 'other'
    for (guint i = 0; i < gst_caps_get_size(other); i++) {
        GstCapsFeatures* f = gst_caps_get_features(other, i);
        if (!f || gst_caps_features_is_any(f))
            continue;
        if (gst_caps_features_contains(f, GST_CAPS_FEATURE_MEMORY_DMABUF))
            gst_caps_features_remove(f, GST_CAPS_FEATURE_MEMORY_DMABUF);
        else {
            gst_caps_features_remove(f, GST_CAPS_FEATURE_MEMORY_SYSTEM_MEMORY);
            gst_caps_features_add(f, GST_CAPS_FEATURE_MEMORY_DMABUF);
        }
    }

    GstCaps* result = gst_caps_merge(orig, other);

    // Add format conversion variants (I420 ↔ NV12) with all fields preserved
    GstStructure* s = gst_caps_get_structure(caps, 0);
    const gchar* fmt = gst_structure_get_string(s, "format");
    if (fmt) {
        // I420 → add NV12 variants (system + DMABuf)
        if (g_strcmp0(fmt, "I420") == 0) {
            GstCaps* nv12_sys = gst_caps_new_empty();
            GstStructure* ns = gst_structure_copy(s);
            gst_structure_set(ns, "format", G_TYPE_STRING, "NV12", NULL);
            gst_caps_append_structure(nv12_sys, ns);
            result = gst_caps_merge(result, nv12_sys);

            GstCaps* nv12_dma = gst_caps_new_empty();
            GstStructure* nd = gst_structure_copy(s);
            gst_structure_set(nd, "format", G_TYPE_STRING, "NV12", NULL);
            gst_caps_append_structure(nv12_dma, nd);
            gst_caps_set_features(nv12_dma, 0, gst_caps_features_new(GST_CAPS_FEATURE_MEMORY_DMABUF, NULL));
            result = gst_caps_merge(result, nv12_dma);
        }
        // NV12 → add I420 variants (system + DMABuf)
        else if (g_strcmp0(fmt, "NV12") == 0) {
            GstCaps* i420_sys = gst_caps_new_empty();
            GstStructure* is = gst_structure_copy(s);
            gst_structure_set(is, "format", G_TYPE_STRING, "I420", NULL);
            gst_caps_append_structure(i420_sys, is);
            result = gst_caps_merge(result, i420_sys);

            GstCaps* i420_dma = gst_caps_new_empty();
            GstStructure* id = gst_structure_copy(s);
            gst_structure_set(id, "format", G_TYPE_STRING, "I420", NULL);
            gst_caps_append_structure(i420_dma, id);
            gst_caps_set_features(i420_dma, 0, gst_caps_features_new(GST_CAPS_FEATURE_MEMORY_DMABUF, NULL));
            result = gst_caps_merge(result, i420_dma);
        }
    }

    if (filter) {
        GstCaps* tmp = gst_caps_intersect_full(result, filter, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(result);
        result = tmp;
    }

    return result;
}

static gboolean gst_magma_videoconvert_set_caps(GstBaseTransform* trans, GstCaps* incaps, GstCaps* outcaps) {

    GstMagmaVideoConvert* self = GST_MAGMA_VIDEOCONVERT(trans);
    GstVideoInfo in_info, out_info;

    if (!gst_video_info_from_caps(&in_info, incaps) || !gst_video_info_from_caps(&out_info, outcaps)) {
        GST_ERROR_OBJECT(self, "Failed to parse caps");
        return FALSE;
    }

    self->in_width = GST_VIDEO_INFO_WIDTH(&in_info);
    self->in_height = GST_VIDEO_INFO_HEIGHT(&in_info);
    self->in_stride = GST_VIDEO_INFO_PLANE_STRIDE(&in_info, 0);
    self->in_format = GST_VIDEO_INFO_FORMAT(&in_info);
    self->out_format = GST_VIDEO_INFO_FORMAT(&out_info);

    MgmMemType in_mem = mem_type_from_caps(incaps);
    MgmMemType out_mem = mem_type_from_caps(outcaps);

    // If input is system memory but the buffer carries MagmaHipMeta,
    // override: the data is really on GPU.
    if (in_mem == MGM_MEM_SYSTEM) {
        // Peek at the first buffer to detect MagmaHipMeta
        // (We can't peek here in set_caps, so the transform will
        //  detect this case and re-dispatch.)
    }

    // Look up converter
    self->convert = nullptr;
    for (gint i = 0; i < convert_table_count; i++) {
        const MgmConvertEntry* e = &convert_table[i];
        if (e->in_mem == in_mem && e->in_fmt == self->in_format && e->out_mem == out_mem && e->out_fmt == self->out_format) {
            self->convert = e->func;
            break;
        }
    }

    GST_INFO_OBJECT(self,
                    "Input %dx%d fmt=%s mem=%d → fmt=%s mem=%d %s",
                    self->in_width,
                    self->in_height,
                    gst_video_format_to_string(self->in_format),
                    (int)in_mem,
                    gst_video_format_to_string(self->out_format),
                    (int)out_mem,
                    self->convert ? "converter found" : "passthrough");
    return TRUE;
}

static gboolean gst_magma_videoconvert_transform_size(GstBaseTransform* trans, GstPadDirection direction, GstCaps* caps, gsize size, GstCaps* othercaps, gsize* othersize) {

    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, othercaps)) {
        GST_ERROR_OBJECT(trans, "Failed to parse output caps for transform_size");
        return FALSE;
    }
    *othersize = GST_VIDEO_INFO_SIZE(&info);
    return TRUE;
}

// ─── Kernel directory resolution ──────────────────────────────────
static const char* find_kernel_dir(void) {
    const char* env = g_getenv("MAGMA_KERNEL_DIR");
    if (env)
        return env;
    return MAGMA_KERNEL_SRC_DIR;
}

// ─── Converter helpers ──────────────────────────────────────────────

static GstBuffer* dmabuf_from_gbm_bo(GstMagmaVideoConvert* self, gsize size, GstVideoFormat fmt, gint w, gint h) {
    int out_fd = gbm_bo_get_fd(self->gbm_bo);
    if (out_fd < 0) {
        GST_ERROR_OBJECT(self, "gbm_bo_get_fd failed");
        return nullptr;
    }

    GstAllocator* allocator = gst_dmabuf_allocator_new();
    if (!allocator) {
        close(out_fd);
        return nullptr;
    }
    GstMemory* mem = gst_dmabuf_allocator_alloc(allocator, out_fd, size);
    gst_object_unref(allocator);
    if (!mem) {
        close(out_fd);
        return nullptr;
    }

    GstBuffer* buf = gst_buffer_new();
    gst_buffer_append_memory(buf, mem);

    // Attach video meta so downstream knows the layout
    gint strides[GST_VIDEO_MAX_PLANES] = {};
    gsize offsets[GST_VIDEO_MAX_PLANES] = {};
    gint n_planes = GST_VIDEO_FORMAT_INFO_N_PLANES(gst_video_format_get_info(fmt));
    guint stride = self->gbm_stride;
    gsize plane_offset = 0;
    for (gint i = 0; i < n_planes; i++) {
        strides[i] = (gint)stride;
        offsets[i] = plane_offset;
        plane_offset += stride * GST_VIDEO_FORMAT_INFO_SCALE_HEIGHT(gst_video_format_get_info(fmt), i, h);
    }
    gst_buffer_add_video_meta_full(buf, GST_VIDEO_FRAME_FLAG_NONE, fmt, w, h, n_planes, offsets, strides);
    return buf;
}

// ─── Converter: System NV12 → DMABuf NV12 ──────────────────────────
static GstFlowReturn conv_sys_nv12_to_dmabuf_nv12(GstMagmaVideoConvert* self, GstBuffer* inbuf, GstBuffer* outbuf) {
    if (!self->gpu_ready && !create_gpu_dmabuf(self))
        return GST_FLOW_ERROR;
    gint w = self->in_width, h = self->in_height;
    gsize pitch = self->gbm_stride;

    // Check for MagmaHipMeta → GPU path
    MagmaHipMeta* hmeta = magma_buffer_get_hip_meta(inbuf);
    if (hmeta) {
        // GPU path: copy NV12 from contiguous GPU buffer to pitched DMABuf
        // hmeta->d_ptr layout: Y at 0, UV interleaved at w*h
        hipDeviceptr_t src_y = hmeta->d_ptr;
        hipDeviceptr_t src_uv = (hipDeviceptr_t)((uint8_t*)hmeta->d_ptr + (size_t)w * h);

        hipError_t herr = hipMemcpy2D((void*)self->d_image, pitch, (const void*)src_y, (size_t)w, (size_t)w, (size_t)h, hipMemcpyDeviceToDevice);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMemcpy2D(Y) failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }

        herr = hipMemcpy2D((void*)((uint8_t*)self->d_image + pitch * h), pitch, (const void*)src_uv, (size_t)w, (size_t)w, (size_t)h / 2, hipMemcpyDeviceToDevice);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMemcpy2D(UV) failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }

        (void)hipStreamSynchronize(self->hip_stream);

        GstBuffer* out = dmabuf_from_gbm_bo(self, self->gpu_size, GST_VIDEO_FORMAT_NV12, w, h);
        if (!out)
            return GST_FLOW_ERROR;
        gst_buffer_remove_all_memory(outbuf);
        gst_buffer_append_memory(outbuf, gst_buffer_get_memory(out, 0));
        {
            gsize o[GST_VIDEO_MAX_PLANES] = {0, (gsize)(pitch * h)};
            gint s[GST_VIDEO_MAX_PLANES] = {(gint)pitch, (gint)pitch};
            if (!gst_buffer_get_video_meta(outbuf))
                gst_buffer_add_video_meta_full(outbuf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12, w, h, 2, o, s);
        }
        gst_buffer_unref(out);
        return GST_FLOW_OK;
    }

    // ─── CPU upload path: copy host NV12 → GPU NV12 ────────────
    gint stride = self->in_stride;

    GstMapInfo in_map;
    if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ))
        return GST_FLOW_ERROR;

    if (stride == (gint)pitch) {
        hipMemcpyAsync(self->d_image, in_map.data, self->gpu_size, hipMemcpyHostToDevice, self->hip_stream);
    } else {
        for (gint y = 0; y < h; y++)
            hipMemcpyAsync((guint8*)self->d_image + y * pitch, in_map.data + y * stride, (gsize)w, hipMemcpyHostToDevice, self->hip_stream);
        const guint8* src_uv = in_map.data + stride * h;
        for (gint y = 0; y < h / 2; y++)
            hipMemcpyAsync((guint8*)self->d_image + pitch * h + y * pitch, src_uv + y * stride, (gsize)w, hipMemcpyHostToDevice, self->hip_stream);
    }
    gst_buffer_unmap(inbuf, &in_map);
    hipStreamSynchronize(self->hip_stream);

    GstBuffer* out = dmabuf_from_gbm_bo(self, self->gpu_size, GST_VIDEO_FORMAT_NV12, w, h);
    if (!out)
        return GST_FLOW_ERROR;
    gst_buffer_remove_all_memory(outbuf);
    gst_buffer_append_memory(outbuf, gst_buffer_get_memory(out, 0));
    {
        gsize offsets[GST_VIDEO_MAX_PLANES] = {0, (gsize)(pitch * h)};
        gint strides[GST_VIDEO_MAX_PLANES] = {(gint)pitch, (gint)pitch};
        if (!gst_buffer_get_video_meta(outbuf))
            gst_buffer_add_video_meta_full(outbuf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12, w, h, 2, offsets, strides);
    }
    gst_buffer_unref(out);
    return GST_FLOW_OK;
}

// ─── Converter: DMABuf NV12 → System NV12 ──────────────────────────
static GstFlowReturn conv_dmabuf_nv12_to_sys_nv12(GstMagmaVideoConvert* self, GstBuffer* inbuf, GstBuffer* outbuf) {
    GstMemory* in_mem = gst_buffer_peek_memory(inbuf, 0);
    if (!in_mem || !gst_is_dmabuf_memory(in_mem))
        return GST_FLOW_ERROR;

    int dma_fd = gst_dmabuf_memory_get_fd(in_mem);
    gsize buf_size = gst_memory_get_sizes(in_mem, NULL, NULL);
    gsize src_stride = self->in_width;
    GstVideoMeta* vmeta = gst_buffer_get_video_meta(inbuf);
    if (vmeta && vmeta->stride[0] > 0)
        src_stride = vmeta->stride[0];
    gint w = self->in_width, h = self->in_height, dst_stride = self->in_stride;

    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = dma_fd;
    desc.size = buf_size;
    hipExternalMemory_t ext_mem;
    hipError_t err = hipImportExternalMemory(&ext_mem, &desc);
    if (err != hipSuccess)
        return GST_FLOW_ERROR;

    hipExternalMemoryBufferDesc bdesc{};
    bdesc.offset = 0;
    bdesc.size = buf_size;
    hipDeviceptr_t d_ptr;
    err = hipExternalMemoryGetMappedBuffer(&d_ptr, ext_mem, &bdesc);
    if (err != hipSuccess) {
        hipDestroyExternalMemory(ext_mem);
        return GST_FLOW_ERROR;
    }
    hipStreamSynchronize(self->hip_stream);

    GstMapInfo out_map;
    if (!gst_buffer_map(outbuf, &out_map, GST_MAP_WRITE)) {
        hipDestroyExternalMemory(ext_mem);
        return GST_FLOW_ERROR;
    }

    if (dst_stride == (gint)src_stride) {
        err = hipMemcpy(out_map.data, d_ptr, buf_size, hipMemcpyDeviceToHost);
    } else {
        for (gint y = 0; y < h; y++)
            if ((err = hipMemcpy(out_map.data + y * dst_stride, (guint8*)d_ptr + y * src_stride, (gsize)w, hipMemcpyDeviceToHost)) != hipSuccess)
                break;
        if (err == hipSuccess) {
            const guint8* src_uv = (guint8*)d_ptr + src_stride * h;
            for (gint y = 0; y < h / 2; y++) {
                auto t = out_map.data + dst_stride * h + y * dst_stride;
                if ((err = hipMemcpy(t, src_uv + y * src_stride, (gsize)w, hipMemcpyDeviceToHost)) != hipSuccess)
                    break;
            }
        }
    }
    gst_buffer_unmap(outbuf, &out_map);
    hipDestroyExternalMemory(ext_mem);
    return err == hipSuccess ? GST_FLOW_OK : GST_FLOW_ERROR;
}

// ─── Converter: System I420 → System NV12 ──────────────────────────
static GstFlowReturn conv_sys_i420_to_sys_nv12(GstMagmaVideoConvert* self, GstBuffer* inbuf, GstBuffer* outbuf) {
    gint w = self->in_width, h = self->in_height;
    gint src_stride = self->in_stride > 0 ? self->in_stride : w;

    // Check for MagmaHipMeta → download from GPU, then interleave on CPU
    MagmaHipMeta* hmeta = magma_buffer_get_hip_meta(inbuf);
    if (hmeta) {
        // Download Y plane from GPU (stride=w) → host
        std::vector<uint8_t> y_host((size_t)w * h);
        hipError_t herr = hipMemcpy2D(y_host.data(), (size_t)w, hmeta->d_ptr, (size_t)w, (size_t)w, (size_t)h, hipMemcpyDeviceToHost);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMemcpy2D(Y) CPU path failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }

        // Download U plane (at offset w*h, stride=w/2)
        uint8_t* d_u = (uint8_t*)hmeta->d_ptr + (size_t)w * h;
        uint8_t* d_v = d_u + (size_t)(w / 2) * (h / 2);
        std::vector<uint8_t> u_host((size_t)(w / 2) * (h / 2));
        std::vector<uint8_t> v_host((size_t)(w / 2) * (h / 2));
        herr = hipMemcpy(u_host.data(), d_u, (size_t)(w / 2) * (h / 2), hipMemcpyDeviceToHost);
        if (herr == hipSuccess)
            herr = hipMemcpy(v_host.data(), d_v, (size_t)(w / 2) * (h / 2), hipMemcpyDeviceToHost);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMemcpy(UV) CPU path failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }

        // Write to output system buffer
        GstMapInfo out_map;
        if (!gst_buffer_map(outbuf, &out_map, GST_MAP_WRITE))
            return GST_FLOW_ERROR;

        GstVideoMeta* vmeta = gst_buffer_get_video_meta(outbuf);
        gint dst_stride = vmeta && vmeta->stride[0] > 0 ? vmeta->stride[0] : w;

        // Y plane
        for (gint y = 0; y < h; y++)
            memcpy(out_map.data + y * dst_stride, y_host.data() + y * w, (size_t)w);

        // Interleave U+V into NV12 UV
        gint dst_uv_off = dst_stride * h;
        for (gint y = 0; y < h / 2; y++)
            for (gint x = 0; x < w / 2; x++) {
                out_map.data[dst_uv_off + y * dst_stride + x * 2] = u_host[y * (w / 2) + x];
                out_map.data[dst_uv_off + y * dst_stride + x * 2 + 1] = v_host[y * (w / 2) + x];
            }

        gst_buffer_unmap(outbuf, &out_map);
        return GST_FLOW_OK;
    }

    // ─── Pure CPU path: map input system buffer directly ─────────
    GstMapInfo in_map;
    if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ))
        return GST_FLOW_ERROR;
    GstMapInfo out_map;
    if (!gst_buffer_map(outbuf, &out_map, GST_MAP_WRITE)) {
        gst_buffer_unmap(inbuf, &in_map);
        return GST_FLOW_ERROR;
    }

    GstVideoMeta* vmeta = gst_buffer_get_video_meta(outbuf);
    gint dst_stride = vmeta && vmeta->stride[0] > 0 ? vmeta->stride[0] : w;

    // Y plane
    for (gint y = 0; y < h; y++)
        memcpy(out_map.data + y * dst_stride, in_map.data + y * src_stride, (size_t)w);

    // Interleave U+V into NV12 UV
    const uint8_t* src_u = in_map.data + src_stride * h;
    const uint8_t* src_v = src_u + (size_t)(w / 2) * (h / 2);
    gint dst_uv_off = dst_stride * h;
    for (gint y = 0; y < h / 2; y++)
        for (gint x = 0; x < w / 2; x++) {
            out_map.data[dst_uv_off + y * dst_stride + x * 2] = src_u[y * (w / 2) + x];
            out_map.data[dst_uv_off + y * dst_stride + x * 2 + 1] = src_v[y * (w / 2) + x];
        }

    gst_buffer_unmap(outbuf, &out_map);
    gst_buffer_unmap(inbuf, &in_map);
    return GST_FLOW_OK;
}

// ─── Converter: I420 → DMABuf NV12 ─────────────────────────────────
// Handles both CPU (upload) and MagmaHip (GPU interleave) inputs.
static GstFlowReturn conv_sys_i420_to_dmabuf_nv12(GstMagmaVideoConvert* self, GstBuffer* inbuf, GstBuffer* outbuf) {
    if (!self->gpu_ready && !create_gpu_dmabuf(self))
        return GST_FLOW_ERROR;

    gint w = self->in_width, h = self->in_height;
    gsize pitch = self->gbm_stride;

    // Check for MagmaHipMeta → GPU path
    MagmaHipMeta* hmeta = magma_buffer_get_hip_meta(inbuf);
    if (hmeta) {
        // GPU path: interleave separate U/V planes into NV12 UV.
        // dptr layout: Y at 0, U at w*h, V at w*h + w*h/4
        if (!self->kernel_ready) {
            std::string kernel_dir(find_kernel_dir());
            std::string kernel_path = kernel_dir + "/videoconvert_kernels.hip";
            HipKernel kern = compile_kernel(kernel_path.c_str(), "i420_to_nv12_uv");
            if (!kern.func) {
                GST_ERROR_OBJECT(self, "Failed to compile i420_to_nv12_uv from %s", kernel_path.c_str());
                return GST_FLOW_ERROR;
            }
            self->kernel_module = kern.module;
            self->kernel_func = kern.func;
            self->kernel_ready = TRUE;
            GST_INFO_OBJECT(self, "Kernel compiled from %s", kernel_path.c_str());
        }

        hipDeviceptr_t pu = (hipDeviceptr_t)((uint8_t*)hmeta->d_ptr + (size_t)w * h);
        hipDeviceptr_t pv = (hipDeviceptr_t)((uint8_t*)hmeta->d_ptr + (size_t)w * h + (size_t)(w / 2) * (h / 2));
        hipDeviceptr_t dst_uv = (hipDeviceptr_t)((uint8_t*)self->d_image + pitch * h);
        int pitch_i = (int)pitch;

        // Copy Y plane: contiguous src (stride=w) → pitched dst (stride=pitch)
        hipError_t herr = hipMemcpy2D((void*)self->d_image, pitch, (const void*)hmeta->d_ptr, (size_t)w, (size_t)w, (size_t)h, hipMemcpyDeviceToDevice);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMemcpy2D(Y) failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }

        // Interleave U+V → NV12 UV via compiled kernel
        void* args[] = {&dst_uv, &pitch_i, &pu, &pv, &w, &h};
        int bx = 32, by = 16;
        dim3 grid((w / 2 + bx - 1) / bx, (h / 2 + by - 1) / by);
        herr = hipModuleLaunchKernel(self->kernel_func, grid.x, grid.y, 1, bx, by, 1, 0, self->hip_stream, args, nullptr);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "i420_nv12 kernel launch failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }
        (void)hipStreamSynchronize(self->hip_stream);

        GstBuffer* out = dmabuf_from_gbm_bo(self, self->gpu_size, GST_VIDEO_FORMAT_NV12, w, h);
        if (!out)
            return GST_FLOW_ERROR;
        gst_buffer_remove_all_memory(outbuf);
        gst_buffer_append_memory(outbuf, gst_buffer_get_memory(out, 0));
        {
            gsize o[GST_VIDEO_MAX_PLANES] = {0, (gsize)(pitch * h)};
            gint s[GST_VIDEO_MAX_PLANES] = {(gint)pitch, (gint)pitch};
            if (!gst_buffer_get_video_meta(outbuf))
                gst_buffer_add_video_meta_full(outbuf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12, w, h, 2, o, s);
        }
        gst_buffer_unref(out);
        return GST_FLOW_OK;
    }

    // ─── CPU upload path: copy host I420 → GPU NV12 ────────────
    GstMapInfo in_map;
    if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ))
        return GST_FLOW_ERROR;

    gint s0 = self->in_stride > 0 ? self->in_stride : w;
    gint su = w / 2; // U/V plane stride for I420

    // Y plane
    for (gint y = 0; y < h; y++)
        hipMemcpyAsync((guint8*)self->d_image + y * pitch, in_map.data + y * s0, (gsize)w, hipMemcpyHostToDevice, self->hip_stream);

    // Interleave U+V → NV12 UV on host then upload
    const uint8_t* src_u = in_map.data + s0 * h;
    const uint8_t* src_v = src_u + (size_t)(w / 2) * (h / 2);
    std::vector<uint8_t> uv_buf(pitch * (h / 2));
    for (gint y = 0; y < h / 2; y++)
        for (gint x = 0; x < w / 2; x++) {
            uv_buf[y * pitch + x * 2] = src_u[y * su + x];
            uv_buf[y * pitch + x * 2 + 1] = src_v[y * su + x];
        }
    hipMemcpyAsync((guint8*)self->d_image + pitch * h, uv_buf.data(), pitch * (h / 2), hipMemcpyHostToDevice, self->hip_stream);
    gst_buffer_unmap(inbuf, &in_map);
    hipStreamSynchronize(self->hip_stream);

    GstBuffer* out = dmabuf_from_gbm_bo(self, self->gpu_size, GST_VIDEO_FORMAT_NV12, w, h);
    if (!out)
        return GST_FLOW_ERROR;
    gst_buffer_remove_all_memory(outbuf);
    gst_buffer_append_memory(outbuf, gst_buffer_get_memory(out, 0));
    {
        gsize o[GST_VIDEO_MAX_PLANES] = {0, (gsize)(pitch * h)};
        gint s[GST_VIDEO_MAX_PLANES] = {(gint)pitch, (gint)pitch};
        if (!gst_buffer_get_video_meta(outbuf))
            gst_buffer_add_video_meta_full(outbuf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12, w, h, 2, o, s);
    }
    gst_buffer_unref(out);
    return GST_FLOW_OK;
}

// ─── transform ──────────────────────────────────────────────────────
static GstFlowReturn gst_magma_videoconvert_transform(GstBaseTransform* trans, GstBuffer* inbuf, GstBuffer* outbuf) {
    GstMagmaVideoConvert* self = GST_MAGMA_VIDEOCONVERT(trans);

    if (!self->convert) {
        // Passthrough — check for MagmaHipMeta GPU data
        MagmaHipMeta* hmeta = magma_buffer_get_hip_meta(inbuf);
        if (hmeta) {
            gsize total = (gsize)self->in_width * self->in_height * 3 / 2;
            GstBuffer* sys_buf = gst_buffer_new_and_alloc(total);
            GstMapInfo map;
            gst_buffer_map(sys_buf, &map, GST_MAP_WRITE);
            hipMemcpy(map.data, hmeta->d_ptr, total, hipMemcpyDeviceToHost);
            hipStreamSynchronize(self->hip_stream);
            gst_buffer_unmap(sys_buf, &map);

            gst_buffer_remove_all_memory(outbuf);
            gst_buffer_append_memory(outbuf, gst_buffer_get_memory(sys_buf, 0));
            gst_buffer_unref(sys_buf);
        } else {
            GstMemory* in_mem = gst_buffer_peek_memory(inbuf, 0);
            if (!in_mem)
                return GST_FLOW_ERROR;
            gst_buffer_remove_all_memory(outbuf);
            gst_buffer_append_memory(outbuf, gst_memory_ref(in_mem));
        }
    } else {
        GstFlowReturn ret = self->convert(self, inbuf, outbuf);
        if (ret != GST_FLOW_OK)
            return ret;
    }

    GST_BUFFER_PTS(outbuf) = GST_BUFFER_PTS(inbuf);
    GST_BUFFER_DURATION(outbuf) = GST_BUFFER_DURATION(inbuf);
    GST_BUFFER_OFFSET(outbuf) = GST_BUFFER_OFFSET(inbuf);
    GST_BUFFER_OFFSET_END(outbuf) = GST_BUFFER_OFFSET_END(inbuf);
    return GST_FLOW_OK;
}

/** --- CLASS INIT --- */
static void gst_magma_videoconvert_class_init(GstMagmaVideoConvertClass* klass) {
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GObjectClass* object_class = G_OBJECT_CLASS(klass);
    GstBaseTransformClass* trans_class = GST_BASE_TRANSFORM_CLASS(klass);

    object_class->set_property = gst_magma_videoconvert_set_property;
    object_class->get_property = gst_magma_videoconvert_get_property;
    object_class->finalize = gst_magma_videoconvert_finalize;

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    gst_element_class_set_static_metadata(element_class, "Magma Video Converter", "Filter/Converter/Video", "Upload CPU NV12 to GPU DMABuf NV12 using GBM+HIP", "Magma");

    trans_class->transform_caps = gst_magma_videoconvert_transform_caps;
    trans_class->set_caps = gst_magma_videoconvert_set_caps;
    trans_class->transform_size = gst_magma_videoconvert_transform_size;
    trans_class->transform = gst_magma_videoconvert_transform;

    GST_DEBUG_CATEGORY_INIT(magma_videoconvert_debug, "magma_videoconvert", 0, "Magma Video Converter");
}

/** --- PLUGIN REGISTRATION --- */
static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "mgmvideoconvert", GST_RANK_NONE, GST_TYPE_MAGMA_VIDEOCONVERT);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mgmvideoconvert, "Magma Video Converter Plugin", plugin_init, "0.1.0", "LGPL", "magma", "https://imeguras.eu.org")
