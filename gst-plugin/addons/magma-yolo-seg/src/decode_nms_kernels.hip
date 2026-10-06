#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" {

__global__ void transpose_col_to_row_kernel(const float* __restrict__ src, float* __restrict__ dst, int stride, int N) {
	int flat = blockIdx.x * blockDim.x + threadIdx.x;
	int total = stride * N;
	if (flat >= total)
		return;
	int detection = flat / stride;
	int channel = flat % stride;
	dst[detection * stride + channel] = src[channel * N + detection];
}

__global__ void decode_filter_kernel(
    const float* __restrict__ data, float* __restrict__ out, int* __restrict__ counter, int N, int stride, int num_classes, int num_coeffs, float conf_thresh, float max_out, int net_w, int net_h) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N)
		return;

	int base = i * stride;

	float cx = data[base + 0];
	float cy = data[base + 1];
	float w = fmaxf(data[base + 2], 0.0f);
	float h = fmaxf(data[base + 3], 0.0f);

	/* Ultralytics YOLO-seg exports already apply sigmoid to the class
	 * channels (Detect._inference returns cat(dbox, cls.sigmoid())).
	 * Applying it a second time squashes every score into [0.5, 0.73]. */
	float max_score = -1e10f;
	int max_class = -1;
	for (int c = 0; c < num_classes; c++) {
		float s = data[base + 4 + c];
		if (s > max_score) {
			max_score = s;
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
	if (idx >= max_out)
		return;

	int es = 7 + num_coeffs;
	float* o = out + (size_t)idx * es;
	o[0] = x1;
	o[1] = y1;
	o[2] = x2;
	o[3] = y2;
	o[4] = (float)max_class;
	o[5] = max_score;
	o[6] = (float)i;

	int coeff_base = i * stride + 4 + num_classes;
	for (int c = 0; c < num_coeffs; c++) {
		o[7 + c] = data[coeff_base + c];
	}
}

__device__ static float box_iou(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2) {
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

__global__ void nms_mask_kernel(const float* __restrict__ filtered,
                                const float* __restrict__ d_rowmajor,
                                const float* __restrict__ d_proto,
                                float* __restrict__ objects,
                                int* __restrict__ out_counter,
                                const int* __restrict__ d_num_filtered,
                                float* __restrict__ d_masks,
                                int max_out,
                                float nms_thresh,
                                int stride,
                                int num_classes,
                                int num_coeffs,
                                int proto_c,
                                int proto_h,
                                int proto_w,
                                int net_w,
                                int net_h) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int M = min(*d_num_filtered, max_out);
	if (tid >= M)
		return;

	int es = 7 + num_coeffs;
	float x1 = filtered[tid * es + 0];
	float y1 = filtered[tid * es + 1];
	float x2 = filtered[tid * es + 2];
	float y2 = filtered[tid * es + 3];
	int class_id = (int)filtered[tid * es + 4];
	float score = filtered[tid * es + 5];

	for (int j = 0; j < tid; j++) {
		int cj = (int)filtered[j * es + 4];
		if (class_id != cj)
			continue;
		float jx1 = filtered[j * es + 0];
		float jy1 = filtered[j * es + 1];
		float jx2 = filtered[j * es + 2];
		float jy2 = filtered[j * es + 3];
		float iou = box_iou(x1, y1, x2, y2, jx1, jy1, jx2, jy2);
		if (iou > nms_thresh)
			return;
	}

	int idx = atomicAdd(out_counter, 1);
	if (idx >= max_out)
		return;

	float* o = objects + idx * 6;
	o[0] = __int_as_float(class_id);
	o[1] = score;
	o[2] = x1 / (float)net_w;
	o[3] = y1 / (float)net_h;
	o[4] = fmaxf(x2 - x1, 0.0f) / (float)net_w;
	o[5] = fmaxf(y2 - y1, 0.0f) / (float)net_h;

	int orig_idx = (int)filtered[tid * es + 6];
	int coeff_base = orig_idx * stride + 4 + num_classes;
	int mask_pixels = proto_h * proto_w;

	/* Box in proto-mask coordinates — the proto basis spans the whole net
	 * input, so a detection's mask is only valid inside its own box
	 * (ultralytics crop_mask).  Everything outside is forced negative. */
	int mx1 = (int)floorf(x1 * (float)proto_w / (float)net_w);
	int my1 = (int)floorf(y1 * (float)proto_h / (float)net_h);
	int mx2 = (int)ceilf(x2 * (float)proto_w / (float)net_w);
	int my2 = (int)ceilf(y2 * (float)proto_h / (float)net_h);

	for (int p = 0; p < mask_pixels; p++) {
		int r = p / proto_w;
		int c0 = p - r * proto_w;
		if (c0 < mx1 || c0 >= mx2 || r < my1 || r >= my2) {
			d_masks[(size_t)idx * mask_pixels + p] = -1e10f;
			continue;
		}
		float sum = 0.0f;
		for (int c = 0; c < num_coeffs; c++) {
			sum += d_rowmajor[coeff_base + c] * d_proto[(size_t)c * mask_pixels + p];
		}
		d_masks[(size_t)idx * mask_pixels + p] = sum;
	}
}
}
