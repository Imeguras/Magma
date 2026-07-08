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
    /** Raw MIGraphX model output (GPU pointer, valid only during the call) */
    const void*    d_raw_output;

    /** Shape of the raw output tensor */
    const int64_t* output_shape;

    /** Number of dimensions in output_shape */
    int            num_dims;

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
