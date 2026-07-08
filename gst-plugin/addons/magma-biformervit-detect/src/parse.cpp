#include "magma_parser_api.h"

#include <algorithm>
#include <cstring>
#include <hip/hip_runtime.h>

extern "C" __global__ void decode_filter_kernel(const float*, float*, int*, int, int, int, float, float, int, int);

extern "C" __global__ void nms_suppress_kernel(const float*, uint8_t*, int, float);

extern "C" __global__ void compact_kernel(const float*, const uint8_t*, float*, int*, int, int, int);

struct Proposal {
    float x1, y1, x2, y2;
    int class_id;
    float score;
    int orig_idx;
};

static inline void safe_free_device(void* p) {
    if (p)
        (void)hipFree(p);
}

extern "C" int magma_parse(MagmaParseParams* p) {
    if (!p || !p->d_raw_output || !p->d_objects || !p->d_num_detected)
        return 1;

    hipStream_t stream = (hipStream_t)p->stream;

    int N = 1, stride = 1, num_classes = 80;

    if (p->num_dims == 3) {
        N = (int)p->output_shape[1];
        stride = (int)p->output_shape[2];
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
    int net_w = p->net_width > 0 ? p->net_width : 224;
    int net_h = p->net_height > 0 ? p->net_height : 224;
    int block = 256;
    int grid = (N + block - 1) / block;
    int num_survivors = 0;

    float* d_intermediate = nullptr;
    int* d_counter = nullptr;
    uint8_t* d_suppressed = nullptr;
    float* d_sorted = nullptr;
    size_t inter_bytes = 0;

    hipError_t e;

    inter_bytes = (size_t)N * 7 * sizeof(float);
    e = hipMalloc(&d_intermediate, inter_bytes);
    if (e != hipSuccess)
        goto fail;

    e = hipMalloc(&d_counter, sizeof(int));
    if (e != hipSuccess)
        goto fail;

    e = hipMemsetAsync(d_counter, 0, sizeof(int), stream);
    if (e != hipSuccess)
        goto fail;

    decode_filter_kernel<<<grid, block, 0, stream>>>((const float*)p->d_raw_output, d_intermediate, d_counter, N, stride, num_classes, p->confidence_thresh, (float)max_out, net_w, net_h);

    e = hipStreamSynchronize(stream);
    if (e != hipSuccess)
        goto fail;

    e = hipMemcpyDtoHAsync(&num_survivors, d_counter, sizeof(int), stream);
    if (e != hipSuccess)
        goto fail;
    e = hipStreamSynchronize(stream);
    if (e != hipSuccess)
        goto fail;
    if (num_survivors > max_out)
        num_survivors = max_out;

    if (num_survivors <= 0)
        goto done;
    if (num_survivors > N)
        num_survivors = N;

    {
        size_t surv_bytes = (size_t)num_survivors * 7 * sizeof(float);
        float* host_survivors = (float*)malloc(surv_bytes);
        if (!host_survivors)
            goto fail;

        e = hipMemcpyDtoH(host_survivors, d_intermediate, surv_bytes);
        if (e != hipSuccess) {
            free(host_survivors);
            goto fail;
        }

        std::sort((Proposal*)host_survivors, (Proposal*)host_survivors + num_survivors, [](const Proposal& a, const Proposal& b) { return a.score > b.score; });

        e = hipMalloc(&d_sorted, surv_bytes);
        if (e != hipSuccess) {
            free(host_survivors);
            goto fail;
        }
        e = hipMalloc(&d_suppressed, (size_t)num_survivors);
        if (e != hipSuccess) {
            free(host_survivors);
            goto fail;
        }

        e = hipMemcpyHtoDAsync(d_sorted, host_survivors, surv_bytes, stream);
        free(host_survivors);
        if (e != hipSuccess)
            goto fail;
    }

    e = hipMemsetAsync(d_suppressed, 0, (size_t)num_survivors, stream);
    if (e != hipSuccess)
        goto fail;

    nms_suppress_kernel<<<grid, block, 0, stream>>>(d_sorted, d_suppressed, num_survivors, p->nms_thresh);

    e = hipMemsetAsync(d_counter, 0, sizeof(int), stream);
    if (e != hipSuccess)
        goto fail;

    compact_kernel<<<grid, block, 0, stream>>>(d_sorted, d_suppressed, (float*)p->d_objects, d_counter, num_survivors, net_w, net_h);

    e = hipMemcpyDtoHAsync(p->d_num_detected, d_counter, sizeof(int), stream);
    if (e != hipSuccess)
        goto fail;
    e = hipStreamSynchronize(stream);

done:
    safe_free_device(d_intermediate);
    safe_free_device(d_counter);
    safe_free_device(d_sorted);
    safe_free_device(d_suppressed);
    return (e == hipSuccess) ? 0 : 1;

fail:
    safe_free_device(d_intermediate);
    safe_free_device(d_counter);
    safe_free_device(d_sorted);
    safe_free_device(d_suppressed);
    return 1;
}
