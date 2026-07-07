#include "magma_parser_api.h"

#include <algorithm>
#include <cstring>
#include <hip/hip_runtime.h>

/* kernel declarations */
extern "C" __global__ void transpose_col_to_row_kernel(const float*, float*, int, int);

extern "C" __global__ void decode_filter_kernel(const float*, float*, int*, int, int, int, float, float, int, int);

extern "C" __global__ void nms_compact_kernel(const float*, float*, int*, const int*, int, float, int, int, int);

static inline void safe_free_device(void* p) {
    if (p)
        (void)hipFree(p);
}

extern "C" int magma_parse(MagmaParseParams* p) {
    if (!p || !p->d_raw_output || !p->d_objects || !p->d_num_detected)
        return 1;

    hipStream_t stream = (hipStream_t)p->stream;

    int N = 1, stride = 1, num_classes = 80;
    bool col_major = false;
    float* d_work = (float*)p->d_raw_output;

    if (p->num_dims == 3) {
        int d1 = (int)p->output_shape[1];
        int d2 = (int)p->output_shape[2];
        if (d1 > d2) {
            N = d1;
            stride = d2;
        } else {
            N = d2;
            stride = d1;
            col_major = true;
        }
    } else if (p->num_dims == 2) {
        N = (int)p->output_shape[0];
        stride = (int)p->output_shape[1];
    } else if (p->num_dims == 1) {
        N = (int)p->output_shape[0];
        stride = 1;
    } else {
        return 1;
    }

    num_classes = stride - 4;
    if (num_classes < 1)
        return 1;

    int max_out = p->max_detections > 0 ? p->max_detections : 100;
    int net_w = p->net_width > 0 ? p->net_width : 640;
    int net_h = p->net_height > 0 ? p->net_height : 640;
    int block = 256;
    int grid = (N + block - 1) / block;

    float* d_transposed = nullptr;
    float* d_intermediate = nullptr;
    int* d_counter = nullptr;
    int* d_out_counter = nullptr;
    size_t inter_bytes = 0;
    int padded = 1, nms_grid = 0;
    size_t shared_bytes = 0;

    hipError_t e;

    if (col_major) {
        size_t total = (size_t)N * stride;
        e = hipMalloc(&d_transposed, total * sizeof(float));
        if (e != hipSuccess)
            goto fail;
        int tgrid = (total + block - 1) / block;
        transpose_col_to_row_kernel<<<tgrid, block, 0, stream>>>((const float*)p->d_raw_output, d_transposed, N, stride);
        d_work = d_transposed;
    }

    inter_bytes = (size_t)N * 7 * sizeof(float);
    e = hipMalloc(&d_intermediate, inter_bytes);
    if (e != hipSuccess)
        goto fail;

    e = hipMalloc(&d_counter, sizeof(int));
    if (e != hipSuccess)
        goto fail;

    e = hipMalloc(&d_out_counter, sizeof(int));
    if (e != hipSuccess)
        goto fail;

    e = hipMemsetAsync(d_counter, 0, sizeof(int), stream);
    if (e != hipSuccess)
        goto fail;

    decode_filter_kernel<<<grid, block, 0, stream>>>((const float*)d_work, d_intermediate, d_counter, N, stride, num_classes, p->confidence_thresh, (float)max_out, net_w, net_h);

    /* No sync between decode_filter and nms_compact — same stream guarantees ordering.
       nms_compact reads d_counter (M survivors) from device, no host round-trip. */

    e = hipMemsetAsync(d_out_counter, 0, sizeof(int), stream);
    if (e != hipSuccess)
        goto fail;

    padded = 1;
    while (padded < max_out) padded <<= 1;
    if (padded > 1024)
        goto fail;
    shared_bytes = (size_t)padded * (sizeof(float) + sizeof(int));
    nms_compact_kernel<<<1, padded, shared_bytes, stream>>>(
        d_intermediate,
        (float*)p->d_objects,
        d_out_counter,
        d_counter,
        max_out,
        p->nms_thresh,
        net_w, net_h,
        padded);

    e = hipMemcpyDtoHAsync(p->d_num_detected, d_out_counter, sizeof(int), stream);
    if (e != hipSuccess)
        goto fail;

    e = hipStreamSynchronize(stream);
    if (e != hipSuccess)
        goto fail;

done:
    safe_free_device(d_transposed);
    safe_free_device(d_intermediate);
    safe_free_device(d_counter);
    safe_free_device(d_out_counter);
    return (e == hipSuccess) ? 0 : 1;

fail:
    safe_free_device(d_transposed);
    safe_free_device(d_intermediate);
    safe_free_device(d_counter);
    safe_free_device(d_out_counter);
    return 1;
}
