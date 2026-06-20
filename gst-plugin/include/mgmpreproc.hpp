#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <hip/hip_runtime.h>

G_BEGIN_DECLS

#define SINK_CAPS \
"video/x-raw, format=(string){NV12, RGBA}"

#define SRC_CAPS \
"application/x-tensor, type=(string)float32"


#define GST_TYPE_MAGMA_PREPROC (gst_magma_preproc_get_type())

G_DECLARE_FINAL_TYPE(
    GstMagmaPreproc,
    gst_magma_preproc,
    GST,
    MAGMA_PREPROC,
    GstBaseTransform
)

struct _GstMagmaPreproc{
    GstBaseTransform parent;

    gint net_width;
    gint net_height;
    gfloat scale_factor;

    hipStream_t hip_stream;
    void *d_tensor_output;
};


G_END_DECLS