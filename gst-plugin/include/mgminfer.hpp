#pragma once

/**
 * @file mgminfer.hpp
 * @brief Inference element — runs MIGraphX models and attaches detection results.
 */

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <hip/hip_runtime.h>
#include "magma-infer-meta.h"
#include "magma_parser_api.h"

G_BEGIN_DECLS

#define INFER_SINK_CAPS "video/x-raw(memory:DMABuf),format=(string)NV12"
#define INFER_SRC_CAPS "video/x-raw(memory:DMABuf),format=(string)NV12"

#define GST_TYPE_MAGMA_INFER (gst_magma_infer_get_type())

G_DECLARE_FINAL_TYPE(GstMagmaInfer, gst_magma_infer, GST, MAGMA_INFER, GstBaseTransform)

/**
 * @brief Magma inference element.
 *
 * Runs a MIGraphX compiled model on preprocessed RGB tensors.
 * Attaches MagmaInferenceMeta with detection objects to output buffers.
 * Supports model-parser plugins (YOLOv8, BiFormer, etc.) loaded at runtime.
 *
 * @property model-onnx-file Path to ONNX model file
 * @property model-mxr-file  Path to MIGraphX compiled model (.mxr)
 * @property inference-interval  Run inference every N frames (default 1)
 * @property parser-plugin   Path to parser plugin .so
 * @property confidence-threshold  Detection confidence threshold
 * @property nms-threshold    NMS IoU threshold
 * @property max-detections   Maximum detections per frame
 */
struct _GstMagmaInfer {
    GstBaseTransform parent;
    gchar* onnx_model_path;
    gchar* mxr_model_path;

    guint inference_interval;
    guint frame_counter;

    gint in_width;
    gint in_height;

    MagmaInferObjectGPU* d_objects;
    guint max_objects;

    int* d_num_det;

    hipStream_t hip_stream;

    hipModule_t kernel_module;
    hipFunction_t kernel_dummy;
    gboolean kernel_ready;

    void* migraphx_model;
    gboolean model_loaded;

    gchar* parser_plugin_path;
    gchar* parser_func_name;
    void* parser_handle;
    MagmaParseFunc parser_func;

    float confidence_thresh;
    float nms_thresh;
    guint max_detections;

    gint class_filter;

    GstMemory* cached_tensor_mem;
    hipExternalMemory_t cached_tensor_ext;
    hipDeviceptr_t cached_tensor_dptr;
    void* gpu_ctx;
};

G_END_DECLS
