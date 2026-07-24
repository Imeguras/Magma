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

/* ── new-detection path kernels ──────────────────────────────────── */

__global__ void filter_normalize_kernel(
    const float* __restrict__ d_dets,
    const void*  __restrict__ d_labels,
    float*       __restrict__ d_out,
    int*         __restrict__ d_counter,
    int num_dets, int max_out, float conf_thresh,
    float inv_w, float inv_h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_dets) return;

    float score = d_dets[i * 5 + 4];
    if (score < conf_thresh) return;

    float x1 = d_dets[i * 5 + 0];
    float y1 = d_dets[i * 5 + 1];
    float x2 = d_dets[i * 5 + 2];
    float y2 = d_dets[i * 5 + 3];
    /* labels are int64 from ONNX; cast via int64_t and truncate */
    int64_t label_raw = static_cast<const int64_t*>(d_labels)[i];
    int cls = (int)label_raw;

    int idx = atomicAdd(d_counter, 1);
    if (idx >= max_out) return;

    float* o = d_out + idx * 6;
    o[0] = (float)cls;
    o[1] = score;
    o[2] = x1 * inv_w;
    o[3] = y1 * inv_h;
    o[4] = fmaxf(x2 - x1, 0.0f) * inv_w;
    o[5] = fmaxf(y2 - y1, 0.0f) * inv_h;
}

/* ── RPN kernels (existing) ──────────────────────────────────────── */

static inline void safe_free_device(void* p) {
    if (p)
        (void)hipFree(p);
}

static int parse_detection(MagmaParseParams* p) {
    /* multi-output path: dets[0] = [batch, N, 5], labels[1] = [batch, N] */
    auto* dets_shape = p->output_shapes[0];
    int dets_ndim = p->num_dims_list[0];
    int N = 0;
    if (dets_ndim == 3) {
        N = (int)dets_shape[1];
    } else if (dets_ndim == 2) {
        N = (int)dets_shape[0];
    } else {
        return 1;
    }
    if (N < 1) return 0;

    const float* d_dets = (const float*)p->d_raw_outputs[0];
    const void* d_labels = p->d_raw_outputs[1];

    int max_out = p->max_detections > 0 ? p->max_detections : 100;
    float conf_thresh = p->confidence_thresh;
    float inv_w = 1.0f / (p->net_width > 0 ? (float)p->net_width : 800.0f);
    float inv_h = 1.0f / (p->net_height > 0 ? (float)p->net_height : 800.0f);

    int block = 256;
    int grid = (N + block - 1) / block;

    hipError_t e;
    e = hipMemsetAsync(p->d_num_detected, 0, sizeof(int), (hipStream_t)p->stream);
    if (e != hipSuccess) return 1;

    filter_normalize_kernel<<<grid, block, 0, (hipStream_t)p->stream>>>(
        d_dets, d_labels, (float*)p->d_objects, p->d_num_detected,
        N, max_out, conf_thresh, inv_w, inv_h);

    e = hipStreamSynchronize((hipStream_t)p->stream);
    return (e == hipSuccess) ? 0 : 1;
}

static int parse_rpn(MagmaParseParams* p) {
    /* legacy single-output RPN path: output = [N, 5] with stride=5 */
    hipStream_t stream = (hipStream_t)p->stream;

    int N = 0, stride = 0;
    if (p->num_dims == 3) {
        N = (int)p->output_shape[1];
        stride = (int)p->output_shape[2];
    } else if (p->num_dims == 2) {
        int d0 = (int)p->output_shape[0];
        int d1 = (int)p->output_shape[1];
        if (d0 == 5) {
            N = d1; stride = d0;
        } else {
            N = d0; stride = d1;
        }
    } else if (p->num_dims == 1) {
        int total = (int)p->output_shape[0];
        N = total / 5; stride = 5;
    } else {
        return 1;
    }
    if (stride != 5 || N < 1) return 1;

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

extern "C" int magma_parse(MagmaParseParams* p) {
    if (!p || !p->d_objects || !p->d_num_detected)
        return 1;

    if (p->num_raw_outputs >= 2) {
        /* detection path: dets + labels */
        if (!p->d_raw_outputs || !p->output_shapes || !p->num_dims_list)
            return 1;
        if (!p->d_raw_outputs[0] || !p->d_raw_outputs[1])
            return 1;
        return parse_detection(p);
    }

    /* legacy single-output RPN path */
    if (!p->d_raw_output)
        return 1;
    return parse_rpn(p);
}
