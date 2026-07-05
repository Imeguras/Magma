#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>
#include <gbm.h>
#include "kernel_utils.hpp"

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_VIDEOCONVERT (gst_magma_videoconvert_get_type())

G_DECLARE_FINAL_TYPE(
    GstMagmaVideoConvert, gst_magma_videoconvert, GST, MAGMA_VIDEOCONVERT, GstBaseTransform)

// Memory type tag for each pad in a conversion
typedef enum {
    MGM_MEM_SYSTEM,   // plain system-memory GstBuffer
    MGM_MEM_DMABUF,   // DMABuf-backed GstBuffer
    MGM_MEM_MAGMAHIP, // GstBuffer carrying MagmaHipMeta (HIP device pointer)
} MgmMemType;

// Converter function: reads/writes inbuf/outbuf according to the pair
typedef GstFlowReturn (*MgmConvertFunc)(GstMagmaVideoConvert*, GstBuffer* inbuf, GstBuffer* outbuf);

// One entry in the converter dispatch table
typedef struct {
    MgmMemType in_mem;
    GstVideoFormat in_fmt;
    MgmMemType out_mem;
    GstVideoFormat out_fmt;
    MgmConvertFunc func;
} MgmConvertEntry;

struct _GstMagmaVideoConvert {
    GstBaseTransform parent;

    gint in_width;
    gint in_height;
    gint in_stride;
    GstVideoFormat in_format;
    GstVideoFormat out_format;

    // Converter selected in set_caps
    MgmConvertFunc convert;

    // DRM/GBM state
    int drm_fd;
    struct gbm_device* gbm_dev;
    struct gbm_bo* gbm_bo;
    guint gbm_stride;

    // HIP import of the GBM BO
    hipExternalMemory_t ext_mem;
    hipDeviceptr_t d_image;
    gsize gpu_size;
    gboolean gpu_ready;

    hipStream_t hip_stream;

    // Compiled I420→NV12 kernel
    hipModule_t kernel_module;
    hipFunction_t kernel_func;
    gboolean kernel_ready;
};

G_END_DECLS
