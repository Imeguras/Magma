#pragma once

/**
 * @file mgmdisplay.hpp
 * @brief DRM/KMS display sink element — renders NV12 video to a connected monitor.
 */

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

/**
 * @brief Magma DRM/KMS display sink.
 *
 * Renders NV12 video frames to a directly-connected display using
 * DRM/KMS and GBM. Supports three input paths:
 *   - DMABuf: imported to HIP, converted NV12→XRGB8888 via GPU kernel
 *   - MagmaHipMeta: same conversion from HIP device pointer
 *   - System memory: CPU→GPU copy then conversion
 *
 * The CRTC always receives an XRGB8888 framebuffer for maximum
 * hardware compatibility.
 *
 * @property sync  Whether to wait for VBlank sync (default: true)
 */
struct _GstMagmaDisplay {
    GstBaseSink parent;

    gint in_width;
    gint in_height;

    gboolean sync;

    int drm_fd;
    uint32_t crtc_id;
    uint32_t connector_id;
    uint32_t current_fb_id;
    drmModeModeInfo mode;
    drmModeCrtcPtr saved_crtc;
    gboolean drm_initialized;
    volatile gboolean flip_pending;

    struct gbm_device* gbm_dev;
    struct gbm_bo* gbm_bo;
    uint32_t gbm_stride;
    uint32_t gbm_fourcc;

    hipExternalMemory_t ext_mem;
    hipDeviceptr_t d_image;
    gsize gpu_size;
    gboolean gpu_ready;

    hipStream_t hip_stream;

    hipModule_t rgb_module;
    hipFunction_t nv12_to_rgb_func;

    guint64 frames_rendered;
};

G_END_DECLS
