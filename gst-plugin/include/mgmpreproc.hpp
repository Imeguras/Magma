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

    /* ── Properties ──────────────────────────────────────────── */
    gint net_width;      /**< Target tensor width (pixels) */
    gint net_height;     /**< Target tensor height (pixels) */
    gfloat scale_factor; /**< Pixel multiplier applied after normalisation */

    gboolean enable_roi; /**< If TRUE, crop source to ROI before resize */
    gint roi_x;          /**< ROI left coordinate in source frame */
    gint roi_y;          /**< ROI top coordinate in source frame */
    gint roi_w;          /**< ROI width */
    gint roi_h;          /**< ROI height */

    /* ── Source frame info (set by set_caps) ─────────────────── */
    gint in_width;       /**< Source frame width */
    gint in_height;      /**< Source frame height */
    GstVideoFormat in_format; /**< Source pixel format (must be NV12) */

    /* ── HIP stream ──────────────────────────────────────────── */
    hipStream_t hip_stream; /**< Shared HIP stream for all GPU ops */

    /* ── Input frame (DMABuf import or H2D upload) ───────────── */
    hipExternalMemory_t external_memory; /**< External mem handle for imported DMABuf */
    hipDeviceptr_t d_image;              /**< GPU pointer to NV12 input frame */
    hipDeviceptr_t d_input_upload;       /**< Fallback H2D upload buffer */

    /* ── Preprocessing kernel ────────────────────────────────── */
    hipModule_t kernel_module;     /**< JIT-compiled HIP module (preproc_kernels.hip) */
    hipFunction_t kernel_nv12_to_rgb; /**< nv12_to_rgb_normalised kernel function */
    gboolean kernel_ready;         /**< TRUE after first successful compile */

    /* ── Tensor output (DMABuf-backed) ───────────────────────── */
    int tensor_dmabuf_fd;          /**< DMABuf FD for the output tensor */
    hipExternalMemory_t tensor_ext_mem; /**< Not used (legacy placeholder) */
    float* d_tensor_output;        /**< GPU pointer to float32 RGB tensor */
    GstMemory* tensor_mem;         /**< GstMemory wrapping the DMABuf tensor */
    gsize tensor_alloc_size;       /**< Allocated byte size of the tensor buffer */

    /* ── State ───────────────────────────────────────────────── */
    gboolean imported; /**< TRUE when input DMABuf is imported to HIP */
};

G_END_DECLS
