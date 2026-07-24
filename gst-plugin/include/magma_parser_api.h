#pragma once

/**
 * @file magma_parser_api.h
 * @brief Parser plugin C API for model output post-processing.
 *
 * Parser plugins (YOLOv8, BiFormer, etc.) are loaded at runtime by
 * mgminfer via dlopen. They implement MagmaParseFunc to convert
 * raw model output tensors into MagmaParsedObject arrays.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A single parsed detection object.
 *
 * Coordinates are normalized 0..1 relative to the model input dimensions.
 */
typedef struct {
    int    class_id;
    float  confidence;
    float  x, y, w, h;       /**< normalized 0..1 relative to input dimensions */
} MagmaParsedObject;

/**
 * @brief Parameters passed from mgminfer to the parser plugin.
 *
 * Valid only during the parser function call -- the parser must copy
 * any data it needs to retain.
 */
typedef struct {
    /** ── Legacy single-output fields ────────────────────────────────
     *  If num_raw_outputs == 0 these are the only outputs available.
     *  If num_raw_outputs  > 0 they mirror output index 0 and the
     *  multi-output arrays below should be preferred.
     */
    const void*    d_raw_output;       /**< raw GPU pointer (first output) */
    const int64_t* output_shape;        /**< shape (first output) */
    int            num_dims;            /**< num dims (first output) */

    /** ── Multi-output support ───────────────────────────────────────
     *  num_raw_outputs == 0 → legacy mode (arrays below may be NULL).
     *  num_raw_outputs  > 0 → use the arrays below for all outputs.
     */
    int            num_raw_outputs;     /**< number of model outputs */
    const void**   d_raw_outputs;       /**< array of GPU ptrs, size num_raw_outputs */
    const int64_t** output_shapes;      /**< array of shape ptrs, size num_raw_outputs */
    const int*     num_dims_list;       /**< array of ndims,  size num_raw_outputs */

    /** Model input width (for box denormalization) */
    int            net_width;

    /** Model input height (for box denormalization) */
    int            net_height;

    /** Confidence threshold */
    float          confidence_thresh;

    /** NMS IoU threshold */
    float          nms_thresh;

    /** Maximum number of detections to return */
    int            max_detections;

    /** Pre-allocated GPU DMABuf for output objects (max_detections entries) */
    void*          d_objects;

    /** GPU pointer to int — parser writes detected count here */
    int*           d_num_detected;

    /** HIP stream for kernel launches (opaque handle) */
    void*          stream;
} MagmaParseParams;

/**
 * @brief Parser plugin entry point.
 *
 * @param params Input/output parameters (see MagmaParseParams)
 * @return 0 on success, nonzero on error
 */
typedef int (*MagmaParseFunc)(MagmaParseParams* params);

#ifdef __cplusplus
}
#endif
