#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_PREPROC (gst_magma_preproc_get_type())

G_DECLARE_FINAL_TYPE(GstMagmaPreproc, gst_magma_preproc, GST, MAGMA_PREPROC, GstBaseTransform)

struct _GstMagmaPreproc {
    GstBaseTransform parent;

    gint net_width;
    gint net_height;
    gfloat scale_factor;

    gint in_width;
    gint in_height;
    GstVideoFormat in_format;

    hipStream_t hip_stream;

    // imported DMABUF (input NV12)
    hipExternalMemory_t external_memory;
    hipDeviceptr_t d_image;

    // compiled GPU module
    hipModule_t kernel_module;
    hipFunction_t kernel_nv12_to_rgb;
    gboolean kernel_ready;

    // tensor output — backed by DMABuf for zero-copy
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
