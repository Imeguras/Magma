#include "mgmvideoconvert.hpp"

#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <xf86drm.h>

/** --- GOBJECT / GSTREAMER STUFF --- */
GST_DEBUG_CATEGORY_STATIC(magma_videoconvert_debug);
#define GST_CAT_DEFAULT magma_videoconvert_debug

enum
{
    PROP_0,
};

G_DEFINE_TYPE(
    GstMagmaVideoConvert,
    gst_magma_videoconvert,
    GST_TYPE_BASE_TRANSFORM
)

/** --- GBM BACKED DMABUF ALLOCATOR --- */

static gboolean create_gpu_dmabuf(GstMagmaVideoConvert *self){
    // 1) Open DRM render node
    const char *drm_path = nullptr;
    for (int i = 0; i < 64; i++) {
        char path[64];
        snprintf(path, sizeof(path), "/dev/dri/renderD%d", 128 + i);
        int fd = open(path, O_RDWR);
        if (fd < 0) continue;
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
    self->gbm_bo = gbm_bo_create(self->gbm_dev,
                                  self->in_width, bo_height,
                                  GBM_FORMAT_R8,
                                  GBM_BO_USE_RENDERING);
    if (!self->gbm_bo) {
        GST_ERROR_OBJECT(self, "gbm_bo_create(%dx%d, R8) failed",
                         self->in_width, bo_height);
        gbm_device_destroy(self->gbm_dev);
        self->gbm_dev = nullptr;
        close(self->drm_fd);
        self->drm_fd = -1;
        return FALSE;
    }

    self->gbm_stride = gbm_bo_get_stride(self->gbm_bo);
    self->gpu_size = self->gbm_stride * self->in_height * 3 / 2;
    GST_INFO_OBJECT(self, "GBM BO created: %dx%d stride=%d size=%zu",
                    self->in_width, self->in_height,
                    self->gbm_stride, self->gpu_size);

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
        GST_ERROR_OBJECT(self, "hipImportExternalMemory failed: %s",
                         hipGetErrorString(err));
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
        GST_ERROR_OBJECT(self, "hipExternalMemoryGetMappedBuffer failed: %s",
                         hipGetErrorString(err));
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
    GST_INFO_OBJECT(self, "GPU DMABuf ready: fd=%d mapped=%p",
                    bo_fd, (void*)self->d_image);
    return TRUE;
}

/** --- PAD TEMPLATES --- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw,format=(string)NV12; "
        "video/x-raw(memory:DMABuf),format=(string)NV12"
    )
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw(memory:DMABuf),format=(string)NV12; "
        "video/x-raw,format=(string)NV12"
    )
);

/** --- INIT --- */
static void gst_magma_videoconvert_init(GstMagmaVideoConvert *self)
{
    self->in_width  = 0;
    self->in_height = 0;
    self->in_stride = 0;

    self->need_upload   = FALSE;
    self->need_download = FALSE;

    self->drm_fd   = -1;
    self->gbm_dev  = nullptr;
    self->gbm_bo   = nullptr;
    self->gbm_stride = 0;

    self->ext_mem   = nullptr;
    self->d_image   = 0;
    self->gpu_size  = 0;
    self->hip_stream = nullptr;
    self->gpu_ready = FALSE;

    if (hipStreamCreate(&self->hip_stream) != hipSuccess)
        self->hip_stream = nullptr;
}

/** --- FINALIZE --- */
static void gst_magma_videoconvert_finalize(GObject *object)
{
    GstMagmaVideoConvert *self = GST_MAGMA_VIDEOCONVERT(object);

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
    if (self->drm_fd >= 0) {
        close(self->drm_fd);
        self->drm_fd = -1;
    }

    G_OBJECT_CLASS(gst_magma_videoconvert_parent_class)->finalize(object);
}

/** --- PROPERTIES --- */
static void gst_magma_videoconvert_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec){
    switch (prop_id){
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_magma_videoconvert_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec){
    switch (prop_id){
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/** --- CAPS NEGOTIATION --- */
static GstCaps *gst_magma_videoconvert_transform_caps(
    GstBaseTransform *trans,
    GstPadDirection direction,
    GstCaps *caps,
    GstCaps *filter){

    GstCaps *orig = gst_caps_copy(caps);
    GstCaps *other = gst_caps_copy(caps);

    for (guint i = 0; i < gst_caps_get_size(other); i++) {
        GstCapsFeatures *f = gst_caps_get_features(other, i);
        if (gst_caps_features_is_any(f))
            continue;
        if (f) {
            if (gst_caps_features_contains(f, GST_CAPS_FEATURE_MEMORY_DMABUF))
                gst_caps_features_remove(f, GST_CAPS_FEATURE_MEMORY_DMABUF);
            else {
                gst_caps_features_remove(f, GST_CAPS_FEATURE_MEMORY_SYSTEM_MEMORY);
                gst_caps_features_add(f, GST_CAPS_FEATURE_MEMORY_DMABUF);
            }
        } else {
            f = gst_caps_features_new(GST_CAPS_FEATURE_MEMORY_DMABUF, NULL);
            gst_caps_set_features(other, i, f);
        }
    }

    GstCaps *result = gst_caps_merge(orig, other);

    if (filter) {
        GstCaps *tmp = gst_caps_intersect_full(
            result, filter, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(result);
        result = tmp;
    }

    return result;
}

static gboolean gst_magma_videoconvert_set_caps(
    GstBaseTransform *trans,
    GstCaps *incaps,
    GstCaps *outcaps){

    GstMagmaVideoConvert *self = GST_MAGMA_VIDEOCONVERT(trans);
    GstVideoInfo in_info, out_info;

    if (!gst_video_info_from_caps(&in_info, incaps) ||
        !gst_video_info_from_caps(&out_info, outcaps)) {
        GST_ERROR_OBJECT(self, "Failed to parse caps");
        return FALSE;
    }

    self->in_width  = GST_VIDEO_INFO_WIDTH(&in_info);
    self->in_height = GST_VIDEO_INFO_HEIGHT(&in_info);
    self->in_stride = GST_VIDEO_INFO_PLANE_STRIDE(&in_info, 0);

    // Determine direction from caps memory features
    gboolean in_dmabuf  = gst_caps_features_contains(
        gst_caps_get_features(incaps, 0), GST_CAPS_FEATURE_MEMORY_DMABUF);
    gboolean out_dmabuf = gst_caps_features_contains(
        gst_caps_get_features(outcaps, 0), GST_CAPS_FEATURE_MEMORY_DMABUF);
    self->need_upload   = !in_dmabuf && out_dmabuf;
    self->need_download =  in_dmabuf && !out_dmabuf;

    GST_INFO_OBJECT(self, "Input %dx%d stride=%d%s%s",
                    self->in_width, self->in_height, self->in_stride,
                    self->need_upload ? " UPLOAD" : "",
                    self->need_download ? " DOWNLOAD" : "");

    return TRUE;
}

static gboolean gst_magma_videoconvert_transform_size(
    GstBaseTransform *trans,
    GstPadDirection direction,
    GstCaps *caps,
    gsize size,
    GstCaps *othercaps,
    gsize *othersize){

    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, othercaps)) {
        GST_ERROR_OBJECT(trans, "Failed to parse output caps for transform_size");
        return FALSE;
    }
    *othersize = GST_VIDEO_INFO_SIZE(&info);
    return TRUE;
}

/** --- TRANSFORM (CPU NV12 -> GPU DMABuf NV12) --- */
static GstFlowReturn gst_magma_videoconvert_transform(
    GstBaseTransform *trans,
    GstBuffer *inbuf,
    GstBuffer *outbuf){

    GstMagmaVideoConvert *self = GST_MAGMA_VIDEOCONVERT(trans);

    if (self->need_upload) {
        // --- CPU -> GPU: upload ---

        if (!self->gpu_ready) {
            if (!create_gpu_dmabuf(self))
                return GST_FLOW_ERROR;
        }

        gint w = self->in_width;
        gint h = self->in_height;
        gsize pitch = self->gbm_stride;
        gint stride = self->in_stride;

        GstMapInfo in_map;
        if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ)) {
            GST_ERROR_OBJECT(self, "Failed to map input buffer");
            return GST_FLOW_ERROR;
        }

        if (stride == (gint)pitch) {
            hipError_t err = hipMemcpyAsync(self->d_image, in_map.data, self->gpu_size,
                                            hipMemcpyHostToDevice, self->hip_stream);
            if (err != hipSuccess) {
                GST_ERROR_OBJECT(self, "hipMemcpyAsync failed: %s", hipGetErrorString(err));
                gst_buffer_unmap(inbuf, &in_map);
                return GST_FLOW_ERROR;
            }
        } else {
            for (gint y = 0; y < h; y++) {
                hipError_t err = hipMemcpyAsync(
                    (guint8*)self->d_image + y * pitch,
                    in_map.data + y * stride, (gsize)w,
                    hipMemcpyHostToDevice, self->hip_stream);
                if (err != hipSuccess) {
                    GST_ERROR_OBJECT(self, "hipMemcpyAsync Y row %d failed", y);
                    gst_buffer_unmap(inbuf, &in_map);
                    return GST_FLOW_ERROR;
                }
            }
            const guint8 *src_uv = in_map.data + stride * h;
            for (gint y = 0; y < h / 2; y++) {
                hipError_t err = hipMemcpyAsync(
                    (guint8*)self->d_image + pitch * h + y * pitch,
                    src_uv + y * stride, (gsize)w,
                    hipMemcpyHostToDevice, self->hip_stream);
                if (err != hipSuccess) {
                    GST_ERROR_OBJECT(self, "hipMemcpyAsync UV row %d failed", y);
                    gst_buffer_unmap(inbuf, &in_map);
                    return GST_FLOW_ERROR;
                }
            }
        }
        gst_buffer_unmap(inbuf, &in_map);

        hipError_t err = hipStreamSynchronize(self->hip_stream);
        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipStreamSynchronize failed: %s", hipGetErrorString(err));
            return GST_FLOW_ERROR;
        }

        // Create output DMABuf from the GBM BO
        int out_fd = gbm_bo_get_fd(self->gbm_bo);
        if (out_fd < 0) {
            GST_ERROR_OBJECT(self, "gbm_bo_get_fd for output failed");
            return GST_FLOW_ERROR;
        }

        GstAllocator *allocator = gst_dmabuf_allocator_new();
        if (!allocator) {
            GST_ERROR_OBJECT(self, "DMABuf allocator not available");
            close(out_fd);
            return GST_FLOW_ERROR;
        }
        GstMemory *dmabuf_mem = gst_dmabuf_allocator_alloc(allocator, out_fd, self->gpu_size);
        gst_object_unref(allocator);
        if (!dmabuf_mem) {
            GST_ERROR_OBJECT(self, "gst_dmabuf_allocator_alloc failed");
            close(out_fd);
            return GST_FLOW_ERROR;
        }

        gst_buffer_remove_all_memory(outbuf);
        gst_buffer_append_memory(outbuf, dmabuf_mem);

        // Attach video meta with GBM stride for downstream
        gint strides[GST_VIDEO_MAX_PLANES] = {(gint)pitch, (gint)pitch};
        gsize offsets[GST_VIDEO_MAX_PLANES] = {0, (gsize)(pitch * h)};
        if (!gst_buffer_get_video_meta(outbuf)) {
            gst_buffer_add_video_meta_full(outbuf, GST_VIDEO_FRAME_FLAG_NONE,
                                           GST_VIDEO_FORMAT_NV12,
                                           w, h, 2, offsets, strides);
        }

    } else if (self->need_download) {
        // --- GPU -> CPU: download ---

        GstMemory *in_mem = gst_buffer_peek_memory(inbuf, 0);
        if (!in_mem || !gst_is_dmabuf_memory(in_mem)) {
            GST_ERROR_OBJECT(self, "Download requires DMABuf input");
            return GST_FLOW_ERROR;
        }

        int dma_fd = gst_dmabuf_memory_get_fd(in_mem);
        if (dma_fd < 0) {
            GST_ERROR_OBJECT(self, "Failed to get DMABuf fd");
            return GST_FLOW_ERROR;
        }

        // Get the DMABuf stride from video meta (set by upload path)
        gsize src_stride = self->in_width;
        GstVideoMeta *vmeta = gst_buffer_get_video_meta(inbuf);
        if (vmeta && vmeta->stride[0] > 0)
            src_stride = vmeta->stride[0];

        gsize buf_size = gst_memory_get_sizes(in_mem, NULL, NULL);
        gint w = self->in_width;
        gint h = self->in_height;
        gint dst_stride = self->in_stride;

        // Import incoming DMABuf into HIP
        hipExternalMemoryHandleDesc desc{};
        desc.type = hipExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = dma_fd;
        desc.size = buf_size;

        hipExternalMemory_t ext_mem;
        hipError_t err = hipImportExternalMemory(&ext_mem, &desc);
        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipImportExternalMemory failed: %s",
                             hipGetErrorString(err));
            return GST_FLOW_ERROR;
        }

        hipExternalMemoryBufferDesc bdesc{};
        bdesc.offset = 0;
        bdesc.size = buf_size;

        hipDeviceptr_t d_ptr;
        err = hipExternalMemoryGetMappedBuffer(&d_ptr, ext_mem, &bdesc);
        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipExternalMemoryGetMappedBuffer failed: %s",
                             hipGetErrorString(err));
            (void)hipDestroyExternalMemory(ext_mem);
            return GST_FLOW_ERROR;
        }

        err = hipStreamSynchronize(self->hip_stream);
        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipStreamSynchronize failed: %s",
                             hipGetErrorString(err));
            (void)hipDestroyExternalMemory(ext_mem);
            return GST_FLOW_ERROR;
        }

        GstMapInfo out_map;
        if (!gst_buffer_map(outbuf, &out_map, GST_MAP_WRITE)) {
            GST_ERROR_OBJECT(self, "Failed to map output buffer");
            (void)hipDestroyExternalMemory(ext_mem);
            return GST_FLOW_ERROR;
        }

        if (dst_stride == (gint)src_stride) {
            err = hipMemcpy(out_map.data, d_ptr, buf_size, hipMemcpyDeviceToHost);
        } else {
            for (gint y = 0; y < h; y++) {
                err = hipMemcpy(out_map.data + y * dst_stride,
                                (guint8*)d_ptr + y * src_stride, (gsize)w,
                                hipMemcpyDeviceToHost);
                if (err != hipSuccess) break;
            }
            if (err == hipSuccess) {
                const guint8 *src_uv = (guint8*)d_ptr + src_stride * h;
                for (gint y = 0; y < h / 2; y++) {
                    err = hipMemcpy(out_map.data + dst_stride * h + y * dst_stride,
                                    src_uv + y * src_stride, (gsize)w,
                                    hipMemcpyDeviceToHost);
                    if (err != hipSuccess) break;
                }
            }
        }
        gst_buffer_unmap(outbuf, &out_map);

        (void)hipDestroyExternalMemory(ext_mem);

        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMemcpy D2H failed: %s", hipGetErrorString(err));
            return GST_FLOW_ERROR;
        }

    } else {
        // Passthrough — just copy input memory to output
        GstMemory *in_mem = gst_buffer_peek_memory(inbuf, 0);
        if (!in_mem) return GST_FLOW_ERROR;
        gst_buffer_remove_all_memory(outbuf);
        gst_buffer_append_memory(outbuf, gst_memory_ref(in_mem));
    }

    GST_BUFFER_PTS(outbuf)       = GST_BUFFER_PTS(inbuf);
    GST_BUFFER_DURATION(outbuf)  = GST_BUFFER_DURATION(inbuf);
    GST_BUFFER_OFFSET(outbuf)    = GST_BUFFER_OFFSET(inbuf);
    GST_BUFFER_OFFSET_END(outbuf)= GST_BUFFER_OFFSET_END(inbuf);

    return GST_FLOW_OK;
}

/** --- CLASS INIT --- */
static void gst_magma_videoconvert_class_init(GstMagmaVideoConvertClass *klass){
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GObjectClass *object_class = G_OBJECT_CLASS(klass);
    GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS(klass);

    object_class->set_property = gst_magma_videoconvert_set_property;
    object_class->get_property = gst_magma_videoconvert_get_property;
    object_class->finalize     = gst_magma_videoconvert_finalize;

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    gst_element_class_set_static_metadata(
        element_class,
        "Magma Video Converter",
        "Filter/Converter/Video",
        "Upload CPU NV12 to GPU DMABuf NV12 using GBM+HIP",
        "Magma"
    );

    trans_class->transform_caps  = gst_magma_videoconvert_transform_caps;
    trans_class->set_caps        = gst_magma_videoconvert_set_caps;
    trans_class->transform_size  = gst_magma_videoconvert_transform_size;
    trans_class->transform       = gst_magma_videoconvert_transform;

    GST_DEBUG_CATEGORY_INIT(magma_videoconvert_debug, "magma_videoconvert", 0,
                            "Magma Video Converter");
}

/** --- PLUGIN REGISTRATION --- */
static gboolean plugin_init(GstPlugin *plugin){
    return gst_element_register(
        plugin,
        "mgmvideoconvert",
        GST_RANK_NONE,
        GST_TYPE_MAGMA_VIDEOCONVERT
    );
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mgmvideoconvert,
    "Magma Video Converter Plugin",
    plugin_init,
    "0.1.0",
    "LGPL",
    "magma",
    "https://imeguras.eu.org"
)
