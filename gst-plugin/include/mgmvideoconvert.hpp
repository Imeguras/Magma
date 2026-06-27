#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>
#include <gbm.h>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_VIDEOCONVERT (gst_magma_videoconvert_get_type())

G_DECLARE_FINAL_TYPE(
    GstMagmaVideoConvert,
    gst_magma_videoconvert,
    GST,
    MAGMA_VIDEOCONVERT,
    GstBaseTransform
)

struct _GstMagmaVideoConvert {
    GstBaseTransform parent;

    gint in_width;
    gint in_height;
    gint in_stride;

    gboolean need_upload;   // CPU→GPU
    gboolean need_download; // GPU→CPU

    // DRM/GBM state
    int drm_fd;
    struct gbm_device *gbm_dev;
    struct gbm_bo *gbm_bo;
    guint gbm_stride;

    // HIP import of the GBM BO
    hipExternalMemory_t ext_mem;
    hipDeviceptr_t d_image;
    gsize gpu_size;
    gboolean gpu_ready;

    hipStream_t hip_stream;
};

G_END_DECLS
