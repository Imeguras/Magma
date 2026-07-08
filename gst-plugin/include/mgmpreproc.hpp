#pragma once

/**
 * @file mgmpreproc.hpp
 * @brief Preprocessing element — crops, resizes, and normalizes video frames to tensors.
 */

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_PREPROC (gst_magma_preproc_get_type())

G_DECLARE_FINAL_TYPE(GstMagmaPreproc, gst_magma_preproc, GST, MAGMA_PREPROC, GstBaseTransform)

/**
 * @brief Magma preprocessing element.
 *
 * Accepts NV12 video and produces a normalized float32 RGB tensor
 * attached as MagmaTensorMeta. Supports configurable ROI cropping
 * and letterbox resizing.
 *
 * @property net-width  Target tensor width
 * @property net-height Target tensor height
 * @property scale-factor  Pixel value scale factor
 * @property enable-roi Enable region-of-interest cropping
 * @property roi-x,roi-y,roi-w,roi-h  ROI rectangle within the source frame
 */
struct _GstMagmaPreproc {
    GstBaseTransform parent;

    gint net_width;
    gint net_height;
    gfloat scale_factor;

    gboolean enable_roi;
    gint roi_x;
    gint roi_y;
    gint roi_w;
    gint roi_h;

    gint in_width;
    gint in_height;
    GstVideoFormat in_format;

    hipStream_t hip_stream;

    hipExternalMemory_t external_memory;
    hipDeviceptr_t d_image;
    hipDeviceptr_t d_input_upload;

    hipModule_t kernel_module;
    hipFunction_t kernel_nv12_to_rgb;
    gboolean kernel_ready;

    int drm_fd;
    struct gbm_device* gbm;
    gboolean gbm_ready;

    int tensor_dmabuf_fd;
    hipExternalMemory_t tensor_ext_mem;
    float* d_tensor_output;
    GstMemory* tensor_mem;
    gsize tensor_alloc_size;

    gboolean imported;
};

G_END_DECLS
