#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>

#include "magma-infer-meta.h"

G_BEGIN_DECLS

/* ---------- caps ---------- */

#define INFER_SINK_CAPS "video/x-raw(memory:DMABuf),format=(string)NV12"
#define INFER_SRC_CAPS "video/x-raw(memory:DMABuf),format=(string)NV12"

/* ---------- element type ---------- */

#define GST_TYPE_MAGMA_INFER (gst_magma_infer_get_type())

G_DECLARE_FINAL_TYPE(GstMagmaInfer, gst_magma_infer, GST, MAGMA_INFER, GstBaseTransform)

struct _GstMagmaInfer {
    GstBaseTransform parent;

    gchar* model_path;

    guint inference_interval;
    guint frame_counter;

    gint in_width;
    gint in_height;

    // GBM for output objects DMABuf
    int drm_fd;
    struct gbm_device* gbm;
    gboolean gbm_ready;

    // Output objects DMABuf (GPU-resident detections)
    int objects_dmabuf_fd;
    hipExternalMemory_t objects_ext_mem;
    MagmaInferObjectGPU* d_objects;
    GstMemory* objects_mem;
    guint max_objects;

    // HIP stream
    hipStream_t hip_stream;

    // Compiled dummy kernel
    hipModule_t kernel_module;
    hipFunction_t kernel_dummy;
    gboolean kernel_ready;

    // MIGraphX model (opaque C++ wrapper, nullptr until loaded)
    void* migraphx_model;
    gboolean model_loaded;
};

G_END_DECLS
