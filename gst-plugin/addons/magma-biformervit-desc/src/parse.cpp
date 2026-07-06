/**
 * @file parse.cpp
 * @brief Implementation of the magma_parse function for post-processing YOLOv8 model outputs.
 *
 */
// MAJOR TODO: fprintf is ultra annoying on gstreamer and apropriate calls should be pipebacked someway but im to lazy and im out of credits!

#include "magma_parser_api.h"

#include <algorithm>
#include <cstring>
#include <hip/hip_runtime.h>

/* kernel declarations */
extern "C" __global__ void classify_threshold_filter_kernel(const float*, float*, int*, int, float, int, int, int);

extern "C" __global__ void compact_topk_classes_kernel(const float*, const uint8_t*, float*, int*, int);

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

    int num_classes = (p->num_dims == 2) ? (int)p->output_shape[1] : (int)p->output_shape[0];

    if (num_classes < 1)
        return 1;

    // for ViT the size is always
    int net_w = 224, net_h = 224, block = 256;

    int max_out = p->max_detections > 0 ? p->max_detections : 100;

    int grid = (num_classes + block - 1) / block;

    // Buffer pointers for intermediate GPU processing
    float* d_intermediate = nullptr;
    int* d_counter = nullptr;
    float* d_sorted = nullptr;
    uint8_t* d_suppressed = nullptr;
    int num_survivors = 0;
    // 0->5 fill so * 7 so sequentially we have x1,y1,x2,y2,class_id,score,orig_idx
    size_t inter_bytes = (size_t)max_out * 7 * sizeof(float);

    hipError_t e;

    // GST_INFO_OBJECT(self, "PARSE: inter_bytes=%zu\n", inter_bytes);
    e = hipMalloc(&d_intermediate, inter_bytes);
    if (e != hipSuccess)
        goto fail;

    e = hipMalloc(&d_counter, sizeof(int));
    if (e != hipSuccess)
        goto fail;

    e = hipMemsetAsync(d_counter, 0, sizeof(int), stream);

    if (e != hipSuccess)
        goto fail;

    // __global__ void classify_threshold_filter_kernel(const float* __restrict__ data,float* __restrict__ out,int* __restrict__ counter, int num_classes, float conf_thresh, int max_out, int net_w,
    // int net_h)
    classify_threshold_filter_kernel<<<grid, block, 0, stream>>>((const float*)p->d_raw_output, d_intermediate, d_counter, num_classes, p->confidence_thresh, (int)max_out, net_w, net_h);
    // hipMemcpyDtoHAsync
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

    {
        size_t surv_bytes = (size_t)num_survivors * 7 * sizeof(float);
        float* host_survivors = (float*)malloc(surv_bytes);
        if (!host_survivors)
            goto fail;

        // 3. Synchronously copy the data from the GPU's d_intermediate to your new CPU host pointer
        e = hipMemcpyDtoH(host_survivors, d_intermediate, surv_bytes);
        if (e != hipSuccess) {
            free(host_survivors);
            goto fail;
        }
        // pulling the good ones upward!
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

        e = hipMemsetAsync(d_suppressed, 0, (size_t)num_survivors, stream);
        if (e != hipSuccess)
            goto fail;
    }
    e = hipMemsetAsync(d_counter, 0, sizeof(int), stream);
    if (e != hipSuccess)
        goto fail;

    compact_topk_classes_kernel<<<grid, block, 0, stream>>>(d_sorted, d_suppressed, (float*)p->d_objects, d_counter, num_survivors);
    e = hipMemcpyDtoHAsync(p->d_num_detected, d_counter, sizeof(int), stream);
    if (e != hipSuccess)
        goto fail;

    e = hipStreamSynchronize(stream);
    if (e != hipSuccess)
        goto fail;

done:

    safe_free_device(d_intermediate);
    safe_free_device(d_counter);
    safe_free_device(d_sorted);
    safe_free_device(d_suppressed);
    return (e == hipSuccess) ? 0 : 1;

fail:
    fprintf(stderr, "PARSE: magma_parse failed: %s\n", hipGetErrorString(e));

    safe_free_device(d_intermediate);
    safe_free_device(d_counter);
    safe_free_device(d_sorted);
    safe_free_device(d_suppressed);
    return 1;
}
