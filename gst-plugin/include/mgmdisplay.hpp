#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasesink.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <gbm.h>
#include "magma-meta.h"

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_DISPLAY (gst_magma_display_get_type())
G_DECLARE_FINAL_TYPE(GstMagmaDisplay, gst_magma_display, GST, MAGMA_DISPLAY, GstBaseSink)

struct _GstMagmaDisplay {
    GstBaseSink parent;

    gint in_width;
    gint in_height;

    gboolean sync;

    // DRM state
    int drm_fd;
    uint32_t crtc_id;
    uint32_t connector_id;
    uint32_t current_fb_id;
    drmModeModeInfo mode;
    drmModeCrtcPtr saved_crtc;
    gboolean drm_initialized;
    volatile gboolean flip_pending;

    // GBM (for MagmaHipMeta→DMABuf conversion)
    struct gbm_device* gbm_dev;
    struct gbm_bo* gbm_bo;
    uint32_t gbm_stride;
    uint32_t gbm_fourcc;

    // HIP import of GBM BO (for HIP copy fallback)
    hipExternalMemory_t ext_mem;
    hipDeviceptr_t d_image;
    gsize gpu_size;
    gboolean gpu_ready;

    hipStream_t hip_stream;

    // JIT-compiled NV12→RGB kernel
    hipModule_t rgb_module;
    hipFunction_t nv12_to_rgb_func;

    // Stats
    guint64 frames_rendered;
};

G_END_DECLS
