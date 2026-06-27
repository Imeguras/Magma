#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_PREPROC (gst_magma_preproc_get_type())

G_DECLARE_FINAL_TYPE(
    GstMagmaPreproc,
    gst_magma_preproc,
    GST,
    MAGMA_PREPROC,
    GstBaseTransform
)

struct _GstMagmaPreproc {
    GstBaseTransform parent;

    gint net_width;
    gint net_height;
    gfloat scale_factor;

    gint in_width;
    gint in_height;
    GstVideoFormat in_format;

    hipStream_t hip_stream;

    // imported DMABUF
    hipExternalMemory_t external_memory;
    hipDeviceptr_t d_image;

    // compiled GPU kernel
    hipModule_t kernel_module;
    hipFunction_t kernel_func;
    gboolean kernel_ready;

    // tensor output
    float *d_tensor_output;

    gboolean imported;
};

G_END_DECLS
