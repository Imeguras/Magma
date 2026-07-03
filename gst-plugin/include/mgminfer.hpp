#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <hip/hip_runtime.h>

#include "magma-infer-meta.h"
#include "magma_parser_api.h"

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

    // Output objects (GPU-resident detections)
    MagmaInferObjectGPU* d_objects;
    guint max_objects;

    // GPU output count
    int* d_num_det;

    // HIP stream
    hipStream_t hip_stream;

    // Compiled dummy kernel
    hipModule_t kernel_module;
    hipFunction_t kernel_dummy;
    gboolean kernel_ready;

    // MIGraphX model (opaque C++ wrapper, nullptr until loaded)
    void* migraphx_model;
    gboolean model_loaded;

    // Parser plugin
    gchar* parser_plugin_path;
    gchar* parser_func_name;
    void*  parser_handle;       /* dlopen handle */
    MagmaParseFunc parser_func; /* dlsym'd */

    // Thresholds (passed to parser)
    float confidence_thresh;
    float nms_thresh;
    guint max_detections;

    // Class filter (-1 = all, 0+ = only this class)
    gint class_filter;
};

G_END_DECLS
