#include "magma_parser_api.h"

#include <algorithm>
#include <cstring>
#include <hip/hip_runtime.h>

extern "C" __global__ void decode_filter_kernel(const float*, float*, int*, int, int, float, float, int, int);

extern "C" __global__ void nms_compact_kernel(const float*, float*, int*, const int*, int, float, int, int);

struct Proposal {
    float x1, y1, x2, y2;
    float class_id;
    float score;
    float orig_idx;
};

static inline void safe_free_device(void* p) {
    if (p)
        (void)hipFree(p);
}

extern "C" int magma_parse(MagmaParseParams* p) {
    if (!p || !p->d_raw_output || !p->d_objects || !p->d_num_detected)
        return 1;

    hipStream_t stream = (hipStream_t)p->stream;

    int N = 0, stride = 0;

    if (p->num_dims == 3) {
        N = (int)p->output_shape[1];
        stride = (int)p->output_shape[2];
    } else if (p->num_dims == 2) {
        int d0 = (int)p->output_shape[0];
        int d1 = (int)p->output_shape[1];
        if (d0 == 5) {
            N = d1;
            stride = d0;
        } else {
            N = d0;
            stride = d1;
        }
    } else if (p->num_dims == 1) {
        int total = (int)p->output_shape[0];
        N = total / 5;
        stride = 5;
    } else {
        return 1;
    }

    if (stride != 5 || N < 1)
        return 1;

    int max_out = p->max_detections > 0 ? p->max_detections : 100;
    int net_w = p->net_width > 0 ? p->net_width : 800;
    int net_h = p->net_height > 0 ? p->net_height : 800;
    int block = 256;
    int grid = (N + block - 1) / block;
    int num_survivors = 0;

    float* d_intermediate = nullptr;
    int* d_decode_count = nullptr;
    int* d_nms_counter = nullptr;
    float* d_sorted = nullptr;

    hipError_t e;

    e = hipMalloc(&d_intermediate, (size_t)N * 7 * sizeof(float));
    if (e != hipSuccess) goto fail;

    e = hipMalloc(&d_decode_count, sizeof(int));
    if (e != hipSuccess) goto fail;

    e = hipMalloc(&d_nms_counter, sizeof(int));
    if (e != hipSuccess) goto fail;

    e = hipMemsetAsync(d_decode_count, 0, sizeof(int), stream);
    if (e != hipSuccess) goto fail;

    decode_filter_kernel<<<grid, block, 0, stream>>>(
        (const float*)p->d_raw_output, d_intermediate, d_decode_count,
        N, stride, p->confidence_thresh, (float)max_out, net_w, net_h);

    e = hipStreamSynchronize(stream);
    if (e != hipSuccess) goto fail;

    e = hipMemcpyDtoH(&num_survivors, d_decode_count, sizeof(int));
    if (e != hipSuccess) goto fail;

    if (num_survivors > 0) {
        if (num_survivors > max_out) num_survivors = max_out;

        size_t surv_bytes = (size_t)num_survivors * 7 * sizeof(float);
        float* host = (float*)malloc(surv_bytes);
        if (!host) goto fail;

        e = hipMemcpyDtoH(host, d_intermediate, surv_bytes);
        if (e != hipSuccess) { free(host); goto fail; }

        std::sort((Proposal*)host, (Proposal*)host + num_survivors,
                  [](const Proposal& a, const Proposal& b) { return a.score > b.score; });

        e = hipMalloc(&d_sorted, surv_bytes);
        if (e != hipSuccess) { free(host); goto fail; }

        e = hipMemcpyHtoDAsync(d_sorted, host, surv_bytes, stream);
        free(host);
        if (e != hipSuccess) goto fail;

        e = hipMemsetAsync(d_nms_counter, 0, sizeof(int), stream);
        if (e != hipSuccess) goto fail;
        e = hipMemcpyHtoDAsync(d_decode_count, &num_survivors, sizeof(int), stream);
        if (e != hipSuccess) goto fail;

        {
            int ng = (num_survivors + block - 1) / block;
            nms_compact_kernel<<<ng, block, 0, stream>>>(
                d_sorted, (float*)p->d_objects, d_nms_counter, d_decode_count,
                max_out, p->nms_thresh, net_w, net_h);
        }

        e = hipMemcpyDtoHAsync(p->d_num_detected, d_nms_counter, sizeof(int), stream);
        if (e != hipSuccess) goto fail;
    } else {
        e = hipMemsetAsync(p->d_num_detected, 0, sizeof(int), stream);
        if (e != hipSuccess) goto fail;
    }

    e = hipStreamSynchronize(stream);

done:
    safe_free_device(d_intermediate);
    safe_free_device(d_decode_count);
    safe_free_device(d_nms_counter);
    safe_free_device(d_sorted);
    return (e == hipSuccess) ? 0 : 1;

fail:
    safe_free_device(d_intermediate);
    safe_free_device(d_decode_count);
    safe_free_device(d_nms_counter);
    safe_free_device(d_sorted);
    return 1;
}
