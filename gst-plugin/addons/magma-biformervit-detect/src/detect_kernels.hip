#include <hip/hip_runtime.h>
#include <stdint.h>

/*
 * BiFormerDetector output layout:
 *   Input shape:  [1, 3, 224, 224]
 *   Output shape: [1, N, 4 + num_classes] row-major
 *
 *   Each query: [cx, cy, w, h, class_score_0, class_score_1, ...]
 *   - cx, cy: center coordinates (absolute pixels, 0..net_w/net_h)
 *   - w, h:   box dimensions in pixels
 *   - class_scores: raw logits (sigmoid → probability)
 */

extern "C" {

/* ---- Kernel 1: decode + threshold filter ---- */
__global__ void decode_filter_kernel(
    const float* __restrict__ data,    /* [1, N, stride] row-major raw output */
    float*       __restrict__ out,     /* [max_out * 7] = {x1,y1,x2,y2,class_id,score,orig_idx} */
    int*         __restrict__ counter, /* atomic output count */
    int N, int stride, int num_classes,
    float conf_thresh, float max_out,
    int net_w, int net_h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int base = i * stride;

    float cx = data[base + 0];
    float cy = data[base + 1];
    float w  = fmaxf(data[base + 2], 0.0f);
    float h  = fmaxf(data[base + 3], 0.0f);

    float max_score = -1e10f;
    int max_class = -1;
    for (int c = 0; c < num_classes; c++) {
        float s = data[base + 4 + c];
        float sig = 1.0f / (1.0f + __expf(-s));
        if (sig > max_score) {
            max_score = sig;
            max_class = c;
        }
    }

    if (max_score < conf_thresh || max_class < 0)
        return;

    float x1 = (cx - w / 2.0f);
    float y1 = (cy - h / 2.0f);
    float x2 = (cx + w / 2.0f);
    float y2 = (cy + h / 2.0f);

    int idx = atomicAdd(counter, 1);
    if (idx >= max_out) return;

    out[idx * 7 + 0] = x1;
    out[idx * 7 + 1] = y1;
    out[idx * 7 + 2] = x2;
    out[idx * 7 + 3] = y2;
    out[idx * 7 + 4] = (float)max_class;
    out[idx * 7 + 5] = max_score;
    out[idx * 7 + 6] = (float)i;
}

/* ---- device helper: IoU ---- */
__device__ static float box_iou(float ax1, float ay1, float ax2, float ay2,
                                 float bx1, float by1, float bx2, float by2)
{
    float ix1 = fmaxf(ax1, bx1);
    float iy1 = fmaxf(ay1, by1);
    float ix2 = fminf(ax2, bx2);
    float iy2 = fminf(ay2, by2);
    float iw = fmaxf(ix2 - ix1, 0.0f);
    float ih = fmaxf(iy2 - iy1, 0.0f);
    float inter = iw * ih;
    float area_a = (ax2 - ax1) * (ay2 - ay1);
    float area_b = (bx2 - bx1) * (by2 - by1);
    float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

/* ---- Kernel 2: per-class NMS ---- */
__global__ void nms_suppress_kernel(
    const float* __restrict__ sorted,  /* [M * 7] host-sorted by score */
    uint8_t*     __restrict__ suppressed,
    int M, float nms_thresh)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    int ci = (int)sorted[i * 7 + 4];
    float xi1 = sorted[i * 7 + 0];
    float yi1 = sorted[i * 7 + 1];
    float xi2 = sorted[i * 7 + 2];
    float yi2 = sorted[i * 7 + 3];

    for (int j = 0; j < i; j++) {
        int cj = (int)sorted[j * 7 + 4];
        if (ci != cj) continue;

        float xj1 = sorted[j * 7 + 0];
        float yj1 = sorted[j * 7 + 1];
        float xj2 = sorted[j * 7 + 2];
        float yj2 = sorted[j * 7 + 3];

        float iou = box_iou(xi1, yi1, xi2, yi2, xj1, yj1, xj2, yj2);
        if (iou > nms_thresh) {
            suppressed[i] = 1;
            return;
        }
    }
    suppressed[i] = 0;
}

/* ---- Kernel 3: compact survivors → MagmaParsedObject ---- */
__global__ void compact_kernel(
    const float* __restrict__ sorted,
    const uint8_t* __restrict__ suppressed,
    float*       __restrict__ objects,    /* output buffer as float[6] per entry */
    int*         __restrict__ out_counter,
    int M, int net_w, int net_h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    if (suppressed[i]) return;

    int idx = atomicAdd(out_counter, 1);

    float x1 = sorted[i * 7 + 0];
    float y1 = sorted[i * 7 + 1];
    float x2 = sorted[i * 7 + 2];
    float y2 = sorted[i * 7 + 3];
    int class_id = (int)sorted[i * 7 + 4];
    float conf   = sorted[i * 7 + 5];

    float* o = objects + idx * 6;
    o[0] = __int_as_float(class_id);
    o[1] = conf;
    o[2] = x1 / (float)net_w;
    o[3] = y1 / (float)net_h;
    o[4] = fmaxf(x2 - x1, 0.0f) / (float)net_w;
    o[5] = fmaxf(y2 - y1, 0.0f) / (float)net_h;
}

} /* extern "C" */
