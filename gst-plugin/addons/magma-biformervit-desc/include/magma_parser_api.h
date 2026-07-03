#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- detection output ---------- */

typedef struct {
    int    class_id;
    float  confidence;
    float  x, y, w, h;       /* normalized 0..1 relative to input dimensions */
} MagmaParsedObject;

/* ---------- parser params (mgminfer → parser) ---------- */

typedef struct {
    /* raw MIGraphX output (GPU pointer, valid only during the call) */
    const void*    d_raw_output;
    const int64_t* output_shape;
    int            num_dims;

    /* model input dimensions (for box denormalization) */
    int            net_width;
    int            net_height;

    /* thresholds */
    float          confidence_thresh;
    float          nms_thresh;
    int            max_detections;

    /* GPU output buffer: pre-allocated DMABuf-backed, max_detections * sizeof(MagmaParsedObject) */
    void*          d_objects;
    /* GPU output count: single int on device, parser sets via hipMemcpy or atomicAdd */
    int*           d_num_detected;

    /* HIP stream for kernel launches (opaque — cast in impl) */
    void*          stream;
} MagmaParseParams;

/* return value: 0 = success, nonzero = error */
typedef int (*MagmaParseFunc)(MagmaParseParams* params);

#ifdef __cplusplus
}
#endif
