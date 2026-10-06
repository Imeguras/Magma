// SPDX-FileCopyrightText: 2026 João Vieira <joaodavid2001@gmail.com>
// SPDX-License-Identifier: LGPL-3.0-or-later

/* ─── Example pose estimation parser ─────────────────────────────────
 *
 * Demonstrates a custom semantic type ("magma.pose.v1") with both
 * magma_parse and magma_to_primitives.
 *
 * Model output format (GPU, float32):
 *   [batch, num_keypoints, 3] = (x, y, confidence) per keypoint
 *
 * The parser:
 *   1. magma_parse: reads keypoints from model output, writes to
 *      the custom GPU output buffer as:
 *        [num_persons: int32]
 *        [person_0_keypoint_0_x, y, conf,  person_0_keypoint_1_x, y, conf, ...]
 *        [person_1_keypoint_0_x, y, conf, ...]
 *      where coordinates are normalised 0..1.
 *
 *   2. magma_to_primitives: reads the custom GPU buffer, converts to
 *      point + polyline primitives.
 *
 * The parser also exports magma_semantic_type = "magma.pose.v1" so
 * mgminfer can register the converter with the correct type_id.
 */

#include "magma_parser_api.h"
#include "magma-primitives.h"

#include <hip/hip_runtime.h>
#include <cstring>
#include <cstdio>

/* ─── COCO skeleton definition (17 keypoints, 16 edges) ──────────── */
#define NUM_KEYPOINTS 17
#define NUM_EDGES 16

static const int skeleton_edges[NUM_EDGES][2] = {
    {0, 1},   /* nose → left_eye */
    {0, 2},   /* nose → right_eye */
    {1, 3},   /* left_eye → left_ear */
    {2, 4},   /* right_eye → right_ear */
    {5, 6},   /* left_shoulder → right_shoulder */
    {5, 7},   /* left_shoulder → left_elbow */
    {7, 9},   /* left_elbow → left_wrist */
    {6, 8},   /* right_shoulder → right_elbow */
    {8, 10},  /* right_elbow → right_wrist */
    {5, 11},  /* left_shoulder → left_hip */
    {6, 12},  /* right_shoulder → right_hip */
    {11, 12}, /* left_hip → right_hip */
    {11, 13}, /* left_hip → left_knee */
    {13, 15}, /* left_knee → left_ankle */
    {12, 14}, /* right_hip → right_knee */
    {14, 16}, /* right_knee → right_ankle */
};

/* ─── GPU kernel: filter + normalise keypoints ────────────────────── */

__global__ void extract_keypoints_kernel(const float* __restrict__ d_input, /* [batch, N, 3] */
                                         float* __restrict__ d_output,      /* flattened per-person keypoints */
                                         int* __restrict__ d_counter,
                                         int num_persons,
                                         int num_kpts,
                                         float conf_thresh) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_persons)
		return;

	int n_kept = 0;
	for (int k = 0; k < num_kpts; k++) {
		float x = d_input[idx * num_kpts * 3 + k * 3 + 0];
		float y = d_input[idx * num_kpts * 3 + k * 3 + 1];
		float conf = d_input[idx * num_kpts * 3 + k * 3 + 2];
		if (conf >= conf_thresh)
			n_kept++;
	}
	if (n_kept < 2)
		return;

	int out_idx = atomicAdd(d_counter, 1);
	/* Output layout per person:
	 *   [num_valid_kpts, kpt0_x, kpt0_y, kpt0_conf, kpt1_x, ...] */
	float* out = d_output + (size_t)out_idx * (1 + num_kpts * 3);
	out[0] = (float)num_kpts;
	for (int k = 0; k < num_kpts; k++) {
		out[1 + k * 3 + 0] = d_input[idx * num_kpts * 3 + k * 3 + 0];
		out[1 + k * 3 + 1] = d_input[idx * num_kpts * 3 + k * 3 + 1];
		out[1 + k * 3 + 2] = d_input[idx * num_kpts * 3 + k * 3 + 2];
	}
}

/* ─── magma_parse: extract keypoints from model output ───────────── */

extern "C" int magma_parse(MagmaParseParams* p) {
	if (!p || !p->d_raw_output || !p->d_num_detected)
		return 1;

	hipStream_t stream = (hipStream_t)p->stream;

	/* Determine number of persons and keypoints from output shape */
	int num_persons = 0, num_kpts = NUM_KEYPOINTS;
	if (p->num_dims == 3) {
		num_persons = (int)p->output_shape[1];
	} else if (p->num_dims == 2) {
		num_persons = (int)p->output_shape[0];
	} else {
		return 1;
	}
	if (num_persons < 1)
		return 0;

	/* Allocate custom GPU output:
	 *   d_num_detected = number of valid persons
	 *   d_objects = per-person keypoint data */
	int max_out = p->max_detections > 0 ? p->max_detections : 100;
	size_t person_bytes = (size_t)(1 + num_kpts * 3) * sizeof(float);

	hipError_t e;
	e = hipMemsetAsync(p->d_num_detected, 0, sizeof(int), stream);
	if (e != hipSuccess)
		return 1;

	int block = 256;
	int grid = (num_persons + block - 1) / block;
	extract_keypoints_kernel<<<grid, block, 0, stream>>>((const float*)p->d_raw_output, (float*)p->d_objects, p->d_num_detected, num_persons, num_kpts, p->confidence_thresh);

	e = hipStreamSynchronize(stream);
	return (e == hipSuccess) ? 0 : 1;
}

/* ─── magma_to_primitives: keypoints → point + polyline primitives ── */

extern "C" const char* magma_semantic_type = "magma.pose.v1";

extern "C" int magma_to_primitives(const MagmaToPrimitivesParams* params, MagmaPrimitiveList* out) {
	if (!params || !out)
		return -1;

	const void* data = params->cpu_data ? params->cpu_data : params->d_data;
	int data_size = params->cpu_data ? params->cpu_data_size : params->data_size;
	if (!data || data_size < (int)sizeof(float))
		return 0;

	int num_kpts = (data_size / (int)sizeof(float) < 52) ? 0 : NUM_KEYPOINTS;
	/* The data layout is [num_persons][1 + num_kpts*3] floats.
	 * Find how many persons are in the buffer. */
	size_t person_floats = (size_t)(1 + num_kpts * 3);
	int num_persons = data_size / (int)(person_floats * sizeof(float));
	if (num_persons > 20)
		num_persons = 20;

	const float* fdata = (const float*)data;

	uint32_t joint_color = 0xFFFF0000; /* red */
	uint32_t bone_color = 0xFF00FF00;  /* green */

	int src_w = params->source_width;
	int src_h = params->source_height;

	for (int person = 0; person < num_persons; person++) {
		const float* kp = fdata + person * person_floats;
		int valid = (int)kp[0];
		if (valid < 2)
			continue;

		/* Allocate temporary buffer for joint screen coordinates */
		int joints_xy[NUM_KEYPOINTS * 2];

		/* Draw keypoints as circles */
		for (int k = 0; k < NUM_KEYPOINTS && k < valid; k++) {
			float nx = kp[1 + k * 3 + 0];
			float ny = kp[1 + k * 3 + 1];
			float conf = kp[1 + k * 3 + 2];
			if (conf < 0.5f)
				continue;

			int sx = (int)(nx * src_w);
			int sy = (int)(ny * src_h);
			joints_xy[k * 2 + 0] = sx;
			joints_xy[k * 2 + 1] = sy;

			magma_primitive_list_add_point(out, sx, sy, joint_color, 4);
		}

		/* Draw skeleton edges as polylines */
		for (int e = 0; e < NUM_EDGES; e++) {
			int k0 = skeleton_edges[e][0];
			int k1 = skeleton_edges[e][1];
			int verts[4] = {
			    joints_xy[k0 * 2 + 0],
			    joints_xy[k0 * 2 + 1],
			    joints_xy[k1 * 2 + 0],
			    joints_xy[k1 * 2 + 1],
			};
			if (verts[0] == 0 && verts[1] == 0)
				continue;
			if (verts[2] == 0 && verts[3] == 0)
				continue;
			magma_primitive_list_add_polyline(out, verts, 2, bone_color, 2);
		}
	}

	return 0;
}
