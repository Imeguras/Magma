#include <hip/hip_runtime.h>
#include <stdint.h>

/* 
 * YOLOv8 output layout:
 *   Default (post-ultralytics v8.1): [1, N, 84] row-major — stride = last dim
 *   Legacy:                          [1, 84, N] col-major — stride = middle dim (84)
 *   num_classes = stride - 4
 */

extern "C" {

/* ---- Kernel 0: col-major → row-major transpose (legacy) ---- */
__global__ void transpose_col_to_row_kernel(
    const float* __restrict__ src,   /* [stride, N] col-major */
    float*       __restrict__ dst,   /* [N, stride] row-major */
    int N, int stride)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * stride;
    if (flat >= total) return;
    int row = flat / stride;
    int col = flat % stride;
    dst[row * stride + col] = src[col * N + row];
}

/* ---- Kernel 1: decode + filter ---- */
__global__ void decode_filter_kernel(
    const float* __restrict__ data,    /* raw output */
    float*       __restrict__ out,      /* [max_out * 7] = {x1,y1,x2,y2,class_id,score,orig_idx} */
    int*         __restrict__ counter,  /* atomic output count */
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

/* ---- Kernel 2: sort + NMS + compact (single kernel) ---- */
/*
 * Bitonic sort by score descending makes upper-triangular NMS correct.
 * padded = next power of 2 >= max_out.  Single-block launch.
 */
__global__ void nms_compact_kernel(
    const float* __restrict__ detections,  /* [M * 7] row-major */
    float*       __restrict__ objects,     /* [max_out * 6] output buffer */
    int*         __restrict__ out_counter, /* output count */
    const int*   __restrict__ d_num_survivors,  /* M = number of decode survivors */
    int max_out, float nms_thresh,
    int net_w, int net_h, int padded)
{
    extern __shared__ char shared[];
    float* s_score = (float*)shared;
    int* s_idx = (int*)(shared + padded * sizeof(float));

    int tid = threadIdx.x;
    int M = min(*d_num_survivors, max_out);
    if (M == 0) return;

    if (tid < M) {
        s_score[tid] = detections[tid * 7 + 5];
        s_idx[tid] = tid;
    } else if (tid < padded) {
        s_score[tid] = -1e30f;
        s_idx[tid] = tid;
    }
    __syncthreads();

    /* Bitonic sort by score DESCENDING (highest first) */
    for (int k = 2; k <= padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (s_score[tid] < s_score[ixj]) {
                        float ts = s_score[tid]; s_score[tid] = s_score[ixj]; s_score[ixj] = ts;
                        int ti = s_idx[tid]; s_idx[tid] = s_idx[ixj]; s_idx[ixj] = ti;
                    }
                } else {
                    if (s_score[tid] > s_score[ixj]) {
                        float ts = s_score[tid]; s_score[tid] = s_score[ixj]; s_score[ixj] = ts;
                        int ti = s_idx[tid]; s_idx[tid] = s_idx[ixj]; s_idx[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (tid >= M) return;

    int src_idx = s_idx[tid];
    float x1 = detections[src_idx * 7 + 0];
    float y1 = detections[src_idx * 7 + 1];
    float x2 = detections[src_idx * 7 + 2];
    float y2 = detections[src_idx * 7 + 3];
    int class_id = (int)detections[src_idx * 7 + 4];
    float score = detections[src_idx * 7 + 5];

    /* Upper-triangular NMS (correct because sorted by score descending) */
    for (int j = 0; j < tid; j++) {
        int jdx = s_idx[j];
        int cj = (int)detections[jdx * 7 + 4];
        if (class_id != cj) continue;

        float jx1 = detections[jdx * 7 + 0];
        float jy1 = detections[jdx * 7 + 1];
        float jx2 = detections[jdx * 7 + 2];
        float jy2 = detections[jdx * 7 + 3];

        float iou = box_iou(x1, y1, x2, y2, jx1, jy1, jx2, jy2);
        if (iou > nms_thresh)
            return;
    }

    /* Survivor: compact to output */
    int idx = atomicAdd(out_counter, 1);
    if (idx >= max_out) return;

    float* o = objects + idx * 6;
    o[0] = __int_as_float(class_id);
    o[1] = score;
    o[2] = x1 / (float)net_w;
    o[3] = y1 / (float)net_h;
    o[4] = fmaxf(x2 - x1, 0.0f) / (float)net_w;
    o[5] = fmaxf(y2 - y1, 0.0f) / (float)net_h;
}

} /* extern "C" */
