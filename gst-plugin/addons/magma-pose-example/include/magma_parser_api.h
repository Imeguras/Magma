#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int    class_id;
    float  confidence;
    float  x, y, w, h;
} MagmaParsedObject;

typedef struct {
    const void*    d_raw_output;
    const int64_t* output_shape;
    int            num_dims;
    int            num_raw_outputs;
    const void**   d_raw_outputs;
    const int64_t** output_shapes;
    const int*     num_dims_list;
    int            net_width;
    int            net_height;
    float          confidence_thresh;
    float          nms_thresh;
    int            max_detections;
    void*          d_objects;
    int*           d_num_detected;
    void*          stream;
} MagmaParseParams;

typedef int (*MagmaParseFunc)(MagmaParseParams* params);

typedef struct {
    const char*  onnx_path;
    const char*  mxr_output_path;
    int          device_id;
    const char*  precision;
    const char*  calib_data_path;
    int          batch_size;
    const char*  const* extras;
} MagmaCompileParams;

typedef int (*MagmaCompileFunc)(const MagmaCompileParams* params);

struct _MagmaPrimitiveList;
struct _MagmaToPrimitivesParams;
typedef int (*MagmaToPrimitivesFunc)(const struct _MagmaToPrimitivesParams* params,
                                     struct _MagmaPrimitiveList* out);

#ifdef __cplusplus
}
#endif
