#include <hip/hip_runtime.h>
#include <stdint.h>

/*
 * InternImage RPN output layout:
 *   Output shape: [1, N, 5] where N depends on input size
 *   stride = 5: [obj_logit, dx, dy, dw, dh]
 *
 * Anchor layout (4 FPN levels, strides 4/8/16/32, 3 ratios per position):
 *   N = sum over levels of (W_l * H_l * 3)
 *   where W_l = net_w / stride_l, H_l = net_h / stride_l
 *
 * Within each level: row-major grid (y,x), 3 anchors per position:
 *   ratio 0.5 → w=11.31, h=5.66
 *   ratio 1.0 → w=8.0,  h=8.0
 *   ratio 2.0 → w=5.66, h=11.31
 */

extern "C" {

__constant__ float c_base_wh[3][2] = {
    {11.313708f, 5.656854f},
    {8.0f, 8.0f},
    {5.656854f, 11.313708f},
};

__constant__ int c_strides[4] = {4, 8, 16, 32};

__global__ void decode_filter_kernel(
    const float* __restrict__ data,
    float*       __restrict__ out,
    int*         __restrict__ counter,
    int N, int stride,
    float conf_thresh, float max_out,
    int net_w, int net_h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int base = i * stride;

    float obj_logit = data[base + 0];
    float obj_score = 1.0f / (1.0f + __expf(-obj_logit));
    if (obj_score < conf_thresh)
        return;

    float dx = data[base + 1];
    float dy = data[base + 2];
    float dw = data[base + 3];
    float dh = data[base + 4];

    int level_starts[5];
    int level_widths[4] = {};
    int level_heights[4] = {};
    level_starts[0] = 0;
    for (int l = 0; l < 4; l++) {
        level_widths[l] = net_w / c_strides[l];
        level_heights[l] = net_h / c_strides[l];
        level_starts[l+1] = level_starts[l] + level_widths[l] * level_heights[l] * 3;
    }

    int level = 0;
    while (level < 4 && i >= level_starts[level+1]) level++;

    int stride_l = c_strides[level];
    int w_l = level_widths[level];
    int level_start = level_starts[level];

    int pos_in_level = i - level_start;
    int h_l = level_heights[level];
    int num_grid = w_l * h_l;
    int ratio_idx = pos_in_level / num_grid;
    int grid_idx = pos_in_level % num_grid;
    int gy = grid_idx / w_l;
    int gx = grid_idx % w_l;

    float cx = (gx + 0.5f) * stride_l;
    float cy = (gy + 0.5f) * stride_l;

    float aw = c_base_wh[ratio_idx][0];
    float ah = c_base_wh[ratio_idx][1];

    float box_cx = dx * aw + cx;
    float box_cy = dy * ah + cy;
    float box_w  = __expf(dw) * aw;
    float box_h  = __expf(dh) * ah;

    float x1 = box_cx - box_w / 2.0f;
    float y1 = box_cy - box_h / 2.0f;
    float x2 = box_cx + box_w / 2.0f;
    float y2 = box_cy + box_h / 2.0f;

    int idx = atomicAdd(counter, 1);
    if (idx >= max_out) return;

    out[idx * 7 + 0] = x1;
    out[idx * 7 + 1] = y1;
    out[idx * 7 + 2] = x2;
    out[idx * 7 + 3] = y2;
    out[idx * 7 + 4] = 0.0f;
    out[idx * 7 + 5] = obj_score;
    out[idx * 7 + 6] = (float)i;
}

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

__global__ void nms_compact_kernel(
    const float* __restrict__ detections,
    float*       __restrict__ objects,
    int*         __restrict__ out_counter,
    const int*   __restrict__ d_num_survivors,
    int max_out, float nms_thresh,
    int net_w, int net_h)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int M = min(*d_num_survivors, max_out);
    if (tid >= M) return;

    float x1 = detections[tid * 7 + 0];
    float y1 = detections[tid * 7 + 1];
    float x2 = detections[tid * 7 + 2];
    float y2 = detections[tid * 7 + 3];
    float score = detections[tid * 7 + 5];

    for (int j = 0; j < tid; j++) {
        float jx1 = detections[j * 7 + 0];
        float jy1 = detections[j * 7 + 1];
        float jx2 = detections[j * 7 + 2];
        float jy2 = detections[j * 7 + 3];

        float iou = box_iou(x1, y1, x2, y2, jx1, jy1, jx2, jy2);
        if (iou > nms_thresh)
            return;
    }

    int idx = atomicAdd(out_counter, 1);
    if (idx >= max_out) return;

    float* o = objects + idx * 6;
    o[0] = 0.0f;
    o[1] = score;
    o[2] = x1 / (float)net_w;
    o[3] = y1 / (float)net_h;
    o[4] = fmaxf(x2 - x1, 0.0f) / (float)net_w;
    o[5] = fmaxf(y2 - y1, 0.0f) / (float)net_h;
}

}
