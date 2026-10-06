// SPDX-FileCopyrightText: 2026 João Vieira <joaodavid2001@gmail.com>
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "magma_parser_api.h"
#include "magma-primitives.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>
#include <hip/hip_runtime.h>
#include <migraphx/migraphx.hpp>

extern "C" __global__ void decode_filter_kernel(const float*, float*, int*, int, int, float, float, int, int);

extern "C" __global__ void nms_compact_kernel(const float*, float*, int*, const int*, int, float, int, int);

extern "C" __global__ void cascade_filter_kernel(const float*, float*, int*, int, int, float, float, int, int);

extern "C" __global__ void cascade_nms_kernel(const float*, float*, int*, const int*, int, float, int, int);

struct Proposal {
	float x1, y1, x2, y2;
	float class_id;
	float score;
	float orig_idx;
};

/* ── new-detection path kernels ──────────────────────────────────── */

__global__ void filter_normalize_kernel(const float* __restrict__ d_dets,
                                        const void* __restrict__ d_labels,
                                        float* __restrict__ d_out,
                                        int* __restrict__ d_counter,
                                        int num_dets,
                                        int max_out,
                                        float conf_thresh,
                                        float inv_w,
                                        float inv_h,
                                        const float* __restrict__ d_masks_in,
                                        float* __restrict__ d_masks_out,
                                        int mask_h,
                                        int mask_w) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_dets)
		return;

	float score = d_dets[i * 5 + 4];
	if (score < conf_thresh)
		return;

	float x1 = d_dets[i * 5 + 0];
	float y1 = d_dets[i * 5 + 1];
	float x2 = d_dets[i * 5 + 2];
	float y2 = d_dets[i * 5 + 3];
	/* labels are int64 from ONNX; cast via int64_t and truncate */
	int64_t label_raw = static_cast<const int64_t*>(d_labels)[i];
	int cls = (int)label_raw;

	int idx = atomicAdd(d_counter, 1);
	if (idx >= max_out)
		return;

	float* o = d_out + idx * 6;
	o[0] = (float)cls;
	o[1] = score;
	o[2] = x1 * inv_w;
	o[3] = y1 * inv_h;
	o[4] = fmaxf(x2 - x1, 0.0f) * inv_w;
	o[5] = fmaxf(y2 - y1, 0.0f) * inv_h;

	/* Copy mask data for this detection (if masks are present) */
	if (d_masks_in && d_masks_out && mask_h > 0 && mask_w > 0) {
		int mask_pixels = mask_h * mask_w;
		for (int p = 0; p < mask_pixels; p++) {
			d_masks_out[(size_t)idx * mask_pixels + p] = d_masks_in[(size_t)i * mask_pixels + p];
		}
	}
}

/* ── decode_masks_kernel: GPU-only mask sigmoid+threshold+polygon ── */

__device__ __constant__ unsigned int mask_palette[16] = {
    0xFF0000FF,
    0xFF00FF00,
    0xFFFF0000,
    0xFFFFFF00,
    0xFFFF00FF,
    0xFF00FFFF,
    0xFFFFA500,
    0xFF800080,
    0xFF87CEEB,
    0xFFFFC0CB,
    0xFF00FF80,
    0xFF7FFF00,
    0xFFFF1493,
    0xFF00BFFF,
    0xFF98FB98,
    0xFFDDA0DD,
};

extern "C" __global__ void decode_masks_kernel(const float* __restrict__ d_masks,
                                               const float* __restrict__ d_objects,
                                               int num_masks,
                                               int mask_h,
                                               int mask_w,
                                               int source_w,
                                               int source_h,
                                               int roi_x,
                                               int roi_y,
                                               int roi_w,
                                               int roi_h,
                                               int* __restrict__ d_vertex_out,
                                               unsigned int* __restrict__ d_fill_out,
                                               unsigned int* __restrict__ d_border_out,
                                               int* __restrict__ d_num_out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_masks)
		return;

	const float* det = d_objects + i * 6;
	float bx = det[2], by = det[3], bw = det[4], bh = det[5];
	if (bw < 0.001f || bh < 0.001f)
		return;

	int mask_pixels = mask_h * mask_w;
	int min_r = mask_h, max_r = -1, min_c = mask_w, max_c = -1;
	for (int p = 0; p < mask_pixels; p++) {
		float v = 1.0f / (1.0f + expf(-d_masks[(size_t)i * mask_pixels + p]));
		if (v > 0.5f) {
			int r = p / mask_w, c = p % mask_w;
			if (r < min_r)
				min_r = r;
			if (r > max_r)
				max_r = r;
			if (c < min_c)
				min_c = c;
			if (c > max_c)
				max_c = c;
		}
	}
	if (min_r > max_r)
		return;

	float sx = (float)roi_x + bx * (float)source_w;
	float sy = (float)roi_y + by * (float)source_h;
	float sw = bw * (float)source_w;
	float sh = bh * (float)source_h;

	float rx0 = (float)min_c / (float)mask_w, ry0 = (float)min_r / (float)mask_h;
	float rx1 = (float)(max_c + 1) / (float)mask_w, ry1 = (float)(max_r + 1) / (float)mask_h;

	int x0 = (int)(sx + rx0 * sw), y0 = (int)(sy + ry0 * sh);
	int x1 = (int)(sx + rx1 * sw), y1 = (int)(sy + ry1 * sh);
	if (x0 < 0)
		x0 = 0;
	if (y0 < 0)
		y0 = 0;
	if (x1 < 0)
		x1 = 0;
	if (y1 < 0)
		y1 = 0;

	int idx = atomicAdd(d_num_out, 1);
	int* v = d_vertex_out + idx * 8;
	v[0] = x0;
	v[1] = y0;
	v[2] = x1;
	v[3] = y0;
	v[4] = x1;
	v[5] = y1;
	v[6] = x0;
	v[7] = y1;

	unsigned int cid = (unsigned int)det[0];
	unsigned int color = mask_palette[cid % 16];
	d_fill_out[idx] = (color & 0xFFFFFF00) | 0x1A; /* ~10% alpha */
	d_border_out[idx] = color;                     /* solid */
}

/* ── RPN kernels (existing) ──────────────────────────────────────── */

static inline void safe_free_device(void* p) {
	if (p)
		(void)hipFree(p);
}

static int parse_detection(MagmaParseParams* p) {
	/* multi-output path: dets[0] = [batch, N, 5], labels[1] = [batch, N], masks[2] = [batch, N, H, W] */
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
	if (N < 1)
		return 0;

	const float* d_dets = (const float*)p->d_raw_outputs[0];
	const void* d_labels = p->d_raw_outputs[1];

	int max_out = p->max_detections > 0 ? p->max_detections : 100;
	float conf_thresh = p->confidence_thresh;
	float inv_w = 1.0f / (p->net_width > 0 ? (float)p->net_width : 800.0f);
	float inv_h = 1.0f / (p->net_height > 0 ? (float)p->net_height : 800.0f);

	/* Check for mask output */
	const float* d_masks_in = nullptr;
	int mask_h = 0, mask_w = 0;
	float* d_masks_out = (float*)p->d_masks;
	if (p->num_raw_outputs >= 3 && p->d_raw_outputs[2] && p->d_masks) {
		d_masks_in = (const float*)p->d_raw_outputs[2];
		if (p->num_dims_list[2] >= 4) {
			auto* mshape = p->output_shapes[2];
			mask_h = (int)mshape[2];
			mask_w = (int)mshape[3];
		}
	}

	int block = 256;
	int grid = (N + block - 1) / block;

	hipError_t e;
	e = hipMemsetAsync(p->d_num_detected, 0, sizeof(int), (hipStream_t)p->stream);
	if (e != hipSuccess)
		return 1;

	filter_normalize_kernel<<<grid, block, 0, (hipStream_t)p->stream>>>(
	    d_dets, d_labels, (float*)p->d_objects, p->d_num_detected, N, max_out, conf_thresh, inv_w, inv_h, d_masks_in, d_masks_out, mask_h, mask_w);

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
	if (e != hipSuccess)
		goto fail;
	e = hipMalloc(&d_decode_count, sizeof(int));
	if (e != hipSuccess)
		goto fail;
	e = hipMalloc(&d_nms_counter, sizeof(int));
	if (e != hipSuccess)
		goto fail;

	e = hipMemsetAsync(d_decode_count, 0, sizeof(int), stream);
	if (e != hipSuccess)
		goto fail;

	decode_filter_kernel<<<grid, block, 0, stream>>>((const float*)p->d_raw_output, d_intermediate, d_decode_count, N, stride, p->confidence_thresh, (float)max_out, net_w, net_h);

	e = hipStreamSynchronize(stream);
	if (e != hipSuccess)
		goto fail;

	e = hipMemcpyDtoH(&num_survivors, d_decode_count, sizeof(int));
	if (e != hipSuccess)
		goto fail;

	if (num_survivors > 0) {
		if (num_survivors > max_out)
			num_survivors = max_out;

		size_t surv_bytes = (size_t)num_survivors * 7 * sizeof(float);
		float* host = (float*)malloc(surv_bytes);
		if (!host)
			goto fail;

		e = hipMemcpyDtoH(host, d_intermediate, surv_bytes);
		if (e != hipSuccess) {
			free(host);
			goto fail;
		}

		std::sort((Proposal*)host, (Proposal*)host + num_survivors, [](const Proposal& a, const Proposal& b) { return a.score > b.score; });

		e = hipMalloc(&d_sorted, surv_bytes);
		if (e != hipSuccess) {
			free(host);
			goto fail;
		}

		e = hipMemcpyHtoDAsync(d_sorted, host, surv_bytes, stream);
		free(host);
		if (e != hipSuccess)
			goto fail;

		e = hipMemsetAsync(d_nms_counter, 0, sizeof(int), stream);
		if (e != hipSuccess)
			goto fail;
		e = hipMemcpyHtoDAsync(d_decode_count, &num_survivors, sizeof(int), stream);
		if (e != hipSuccess)
			goto fail;

		{
			int ng = (num_survivors + block - 1) / block;
			nms_compact_kernel<<<ng, block, 0, stream>>>(d_sorted, (float*)p->d_objects, d_nms_counter, d_decode_count, max_out, p->nms_thresh, net_w, net_h);
		}

		e = hipMemcpyDtoHAsync(p->d_num_detected, d_nms_counter, sizeof(int), stream);
		if (e != hipSuccess)
			goto fail;
	} else {
		e = hipMemsetAsync(p->d_num_detected, 0, sizeof(int), stream);
		if (e != hipSuccess)
			goto fail;
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

/* ─── Compile hook — overrides default MIGraphX compilation ──────── */

extern "C" int magma_compile(const MagmaCompileParams* params) {
	if (!params || !params->onnx_path || !params->mxr_output_path)
		return 1;

	const char* precision = params->precision ? params->precision : "FP32";

	fprintf(stderr,
	        "[magma_compile] ONNX=%s MXR=%s precision=%s calib=%s batch=%d\n",
	        params->onnx_path,
	        params->mxr_output_path,
	        precision,
	        params->calib_data_path ? params->calib_data_path : "(none)",
	        params->batch_size);

	try {
		migraphx::onnx_options opts;
		/* parse extras for overrides */
		if (params->extras) {
			for (int i = 0; params->extras[i]; i++) {
				std::string kv(params->extras[i]);
				auto eq = kv.find('=');
				if (eq != std::string::npos && kv.substr(0, eq) == "input-shape") {
					std::string dims = kv.substr(eq + 1);
					std::vector<std::size_t> shape;
					size_t pos = 0;
					while (pos < dims.size()) {
						size_t next = dims.find(',', pos);
						shape.push_back(std::stoul(dims.substr(pos, next - pos)));
						if (next == std::string::npos)
							break;
						pos = next + 1;
					}
					if (shape.size() >= 2) {
						opts.set_input_parameter_shape("input", shape);
						fprintf(stderr, "[magma_compile] set input shape: [");
						for (auto d : shape)
							fprintf(stderr, "%zu,", d);
						fprintf(stderr, "]\n");
					}
				}
			}
		}

		migraphx::program prog = migraphx::parse_onnx(params->onnx_path, opts);
		fprintf(stderr, "[magma_compile] ONNX parsed OK\n");

		if (strcmp(precision, "FP16") == 0) {
			migraphx::quantize_fp16(prog);
			fprintf(stderr, "[magma_compile] Quantized to FP16\n");
		} else if (strcmp(precision, "INT8") == 0) {
			/* Run AMD Quark PTQ offline, then feed the quantized ONNX
			   through normal MIGraphX compilation (FP32 path — the QDQ
			   nodes handle the INT8 arithmetic at runtime). */
			if (!params->calib_data_path) {
				fprintf(stderr, "[magma_compile] INT8 requested but no calib-data-path\n");
				return 1;
			}

			/* Build temporary path for quantized ONNX */
			std::string tmp_onnx = std::string(params->mxr_output_path) + ".quant.onnx";

			/* Locate the quark_quantize.py script — try source tree first
			   (during development), then installed location, then cwd. */
			std::string script_path;
			const char* search_dirs[] = {"/usr/share/magma/scripts", ".", nullptr};
			for (const char** d = search_dirs; *d; ++d) {
				std::string candidate = std::string(*d) + "/quark_quantize.py";
				FILE* f = fopen(candidate.c_str(), "r");
				if (f) {
					fclose(f);
					script_path = candidate;
					break;
				}
			}

			if (script_path.empty()) {
				fprintf(stderr,
				        "[magma_compile] quark_quantize.py not found "
				        "(looked in /usr/share/magma/scripts and cwd)\n");
				return 1;
			}

			/* Build input-shape string from cached parse info if available.
			   Default: 1,3,640,640 — typical for InternImage family. */
			std::string shape_str = "1,3,640,640";
			if (params->extras) {
				for (int i = 0; params->extras[i]; i++) {
					std::string kv(params->extras[i]);
					auto eq = kv.find('=');
					if (eq != std::string::npos && kv.substr(0, eq) == "input-shape") {
						shape_str = kv.substr(eq + 1);
						break;
					}
				}
			}

			/* Find a Quark-capable Python interpreter */
			std::string quark_python;
			const char* qp_env = std::getenv("QUARK_PYTHON");
			if (qp_env && qp_env[0]) {
				quark_python = qp_env;
			} else {
				const char* py_search[] = {"/opt/magma-quark/bin/python3", "quark-env/bin/python3", "python3", nullptr};
				for (const char** p = py_search; *p; ++p) {
					FILE* f = fopen(*p, "r");
					if (f) {
						fclose(f);
						quark_python = *p;
						break;
					}
				}
			}
			if (quark_python.empty())
				quark_python = "python3";

			std::string cache_dir;
			const char* qcd_env = std::getenv("QUARK_CACHE_DIR");
			if (qcd_env && qcd_env[0])
				cache_dir = std::string(" --cache-dir '") + qcd_env + "'";

			std::string cmd = std::string("'") + quark_python + "' '" + script_path +
			                  "'"
			                  " --onnx '" +
			                  params->onnx_path +
			                  "'"
			                  " --output '" +
			                  tmp_onnx +
			                  "'"
			                  " --calib-dir '" +
			                  params->calib_data_path +
			                  "'"
			                  " --input-shape " +
			                  shape_str + cache_dir + " 2>&1";

			fprintf(stderr, "[magma_compile] Running Quark PTQ: %s\n", cmd.c_str());
			int ret = std::system(cmd.c_str());
			if (ret != 0) {
				fprintf(stderr, "[magma_compile] Quark quantization failed (exit=%d)\n", ret);
				return 1;
			}

			/* Check quantized ONNX was produced */
			{
				FILE* f = fopen(tmp_onnx.c_str(), "r");
				if (!f) {
					fprintf(stderr, "[magma_compile] Quark output not found: %s\n", tmp_onnx.c_str());
					return 1;
				}
				fclose(f);
			}

			/* Parse the quantized ONNX (QDQ nodes → MIGraphX handles them natively) */
			migraphx::onnx_options qopts;
			if (params->extras) {
				for (int i = 0; params->extras[i]; i++) {
					std::string kv(params->extras[i]);
					auto eq = kv.find('=');
					if (eq != std::string::npos && kv.substr(0, eq) == "input-shape") {
						std::string dims = kv.substr(eq + 1);
						std::vector<std::size_t> shape;
						size_t pos = 0;
						while (pos < dims.size()) {
							size_t next = dims.find(',', pos);
							shape.push_back(std::stoul(dims.substr(pos, next - pos)));
							if (next == std::string::npos)
								break;
							pos = next + 1;
						}
						if (shape.size() >= 2) {
							qopts.set_input_parameter_shape("input", shape);
						}
					}
				}
			}
			prog = migraphx::parse_onnx(tmp_onnx.c_str(), qopts);
			fprintf(stderr, "[magma_compile] Quantized ONNX parsed OK\n");

			/* Clean up temp file */
			std::remove(tmp_onnx.c_str());
		}

		migraphx::target tgt("gpu");
		migraphx::compile_options copts;
		prog.compile(tgt, copts);
		fprintf(stderr, "[magma_compile] Compiled for GPU OK\n");

		migraphx::save(prog, params->mxr_output_path);
		fprintf(stderr, "[magma_compile] Saved to %s\n", params->mxr_output_path);

		return 0;

	} catch (const std::exception& e) {
		fprintf(stderr, "[magma_compile] FAILED: %s\n", e.what());
		return 1;
	}
}

static int parse_cascade(MagmaParseParams* p) {
	/* Single output: [batch, N, 6] = [x1, y1, x2, y2, score, class_id] — pre-NMS proposals. */
	int ndim = p->num_dims;
	const int64_t* shape = p->output_shape;
	int N = 0, stride = 0;
	if (ndim == 3) {
		N = (int)shape[1];
		stride = (int)shape[2];
	} else if (ndim == 2) {
		N = (int)shape[0];
		stride = (int)shape[1];
	} else {
		return 1;
	}
	if (stride != 6 || N < 1)
		return 1;

	const float* d_data = (const float*)p->d_raw_output;
	int max_out = p->max_detections > 0 ? p->max_detections : 100;
	float conf_thresh = p->confidence_thresh;
	float nms_thresh = p->nms_thresh;
	int net_w = p->net_width > 0 ? p->net_width : 640;
	int net_h = p->net_height > 0 ? p->net_height : 640;
	int block = 256;
	int grid = (N + block - 1) / block;
	int num_survivors = 0;

	float* d_intermediate = nullptr;
	int* d_filter_count = nullptr;
	int* d_nms_counter = nullptr;
	float* d_sorted = nullptr;

	hipError_t e;

	e = hipMalloc(&d_intermediate, (size_t)N * 7 * sizeof(float));
	if (e != hipSuccess)
		goto fail;
	e = hipMalloc(&d_filter_count, sizeof(int));
	if (e != hipSuccess)
		goto fail;

	e = hipMemsetAsync(d_filter_count, 0, sizeof(int), (hipStream_t)p->stream);
	if (e != hipSuccess)
		goto fail;

	cascade_filter_kernel<<<grid, block, 0, (hipStream_t)p->stream>>>(d_data, d_intermediate, d_filter_count, N, stride, conf_thresh, (float)max_out, net_w, net_h);

	e = hipStreamSynchronize((hipStream_t)p->stream);
	if (e != hipSuccess)
		goto fail;

	e = hipMemcpyDtoH(&num_survivors, d_filter_count, sizeof(int));
	if (e != hipSuccess)
		goto fail;

	if (num_survivors > 0) {
		if (num_survivors > max_out)
			num_survivors = max_out;

		size_t surv_bytes = (size_t)num_survivors * 7 * sizeof(float);
		float* host = (float*)malloc(surv_bytes);
		if (!host)
			goto fail;

		e = hipMemcpyDtoH(host, d_intermediate, surv_bytes);
		if (e != hipSuccess) {
			free(host);
			goto fail;
		}

		std::sort((Proposal*)host, (Proposal*)host + num_survivors, [](const Proposal& a, const Proposal& b) { return a.score > b.score; });

		e = hipMalloc(&d_sorted, surv_bytes);
		if (e != hipSuccess) {
			free(host);
			goto fail;
		}

		e = hipMemcpyHtoDAsync(d_sorted, host, surv_bytes, (hipStream_t)p->stream);
		free(host);
		if (e != hipSuccess)
			goto fail;

		e = hipMalloc(&d_nms_counter, sizeof(int));
		if (e != hipSuccess)
			goto fail;
		e = hipMemsetAsync(d_nms_counter, 0, sizeof(int), (hipStream_t)p->stream);
		if (e != hipSuccess)
			goto fail;
		e = hipMemcpyHtoDAsync(d_filter_count, &num_survivors, sizeof(int), (hipStream_t)p->stream);
		if (e != hipSuccess)
			goto fail;

		{
			int ng = (num_survivors + block - 1) / block;
			cascade_nms_kernel<<<ng, block, 0, (hipStream_t)p->stream>>>(d_sorted, (float*)p->d_objects, d_nms_counter, d_filter_count, max_out, nms_thresh, net_w, net_h);
		}

		e = hipMemcpyDtoHAsync(p->d_num_detected, d_nms_counter, sizeof(int), (hipStream_t)p->stream);
		if (e != hipSuccess)
			goto fail;
	} else {
		e = hipMemsetAsync(p->d_num_detected, 0, sizeof(int), (hipStream_t)p->stream);
		if (e != hipSuccess)
			goto fail;
	}

	e = hipStreamSynchronize((hipStream_t)p->stream);

done:
	safe_free_device(d_intermediate);
	safe_free_device(d_filter_count);
	safe_free_device(d_nms_counter);
	safe_free_device(d_sorted);
	return (e == hipSuccess) ? 0 : 1;

fail:
	safe_free_device(d_intermediate);
	safe_free_device(d_filter_count);
	safe_free_device(d_nms_counter);
	safe_free_device(d_sorted);
	return 1;
}

extern "C" int magma_parse(MagmaParseParams* p) {
	if (!p || !p->d_objects || !p->d_num_detected)
		return 1;

	if (p->num_raw_outputs >= 2) {
		/* detection path: dets + labels (already NMS'd) */
		if (!p->d_raw_outputs || !p->output_shapes || !p->num_dims_list)
			return 1;
		if (!p->d_raw_outputs[0] || !p->d_raw_outputs[1])
			return 1;
		return parse_detection(p);
	}

	/* single-output path */
	if (!p->d_raw_output)
		return 1;

	/* Detect cascade format: last dim == 6 (x1,y1,x2,y2,score,class_id) */
	if (p->num_dims >= 2) {
		int last_dim = (int)p->output_shape[p->num_dims - 1];
		if (last_dim == 6)
			return parse_cascade(p);
	}

	/* legacy RPN path (stride 5) */
	return parse_rpn(p);
}

/* ─── Semantic type string ─────────────────────────────────────────
 * mgminfer reads this symbol at dlopen time and uses it as the
 * registry key when registering magma_to_primitives.
 */
extern "C" const char* magma_semantic_type = "magma.detection.internimage.v1";

/* ─── Inline labels lookup (self-contained, no libmagma-meta dep) ───
 *
 * Reads COCO-format label file (one name per line) and caches by
 * realpath + mtime.  Used by magma_to_primitives below.
 */

struct LabelCache {
	char path[4096];
	time_t mtime;
	char** names;
	int count;
	int valid;
};

static struct LabelCache g_label_cache = {{0}, 0, NULL, 0, 0};

static const char* internimage_lookup_label(unsigned int class_id, const char* labels_path) {
	if (!labels_path || !labels_path[0])
		return NULL;

	struct stat st;
	if (stat(labels_path, &st) != 0)
		return NULL;

	/* Check cache validity */
	if (g_label_cache.valid) {
		if (strcmp(g_label_cache.path, labels_path) == 0 && g_label_cache.mtime == st.st_mtime) {
			if ((int)class_id < g_label_cache.count)
				return g_label_cache.names[class_id] ? g_label_cache.names[class_id] : "";
			return NULL;
		}
		/* Invalidate */
		for (int i = 0; i < g_label_cache.count; i++)
			free(g_label_cache.names[i]);
		free(g_label_cache.names);
		g_label_cache.names = NULL;
		g_label_cache.count = 0;
		g_label_cache.valid = 0;
	}

	/* Load */
	strncpy(g_label_cache.path, labels_path, sizeof(g_label_cache.path) - 1);
	g_label_cache.path[sizeof(g_label_cache.path) - 1] = '\0';
	g_label_cache.mtime = st.st_mtime;

	FILE* f = fopen(labels_path, "r");
	if (!f)
		return NULL;

	int count = 0;
	int ch;
	while ((ch = fgetc(f)) != EOF) {
		if (ch == '\n')
			count++;
	}
	rewind(f);

	g_label_cache.names = (char**)calloc((size_t)count, sizeof(char*));
	g_label_cache.count = count;

	char buf[1024];
	int idx = 0;
	while (idx < count && fgets(buf, sizeof(buf), f)) {
		size_t len = strlen(buf);
		while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
			buf[--len] = '\0';
		if (len > 0)
			g_label_cache.names[idx] = strdup(buf);
		else
			g_label_cache.names[idx] = NULL;
		idx++;
	}
	fclose(f);
	g_label_cache.valid = 1;

	if ((int)class_id < g_label_cache.count)
		return g_label_cache.names[class_id] ? g_label_cache.names[class_id] : "";
	return NULL;
}

/* ─── GPU-accelerated mask polygon generation ───────────────────────
 *
 * Cached GPU buffers for decode_masks_kernel output. None of this
 * touches the full mask tensor on CPU — only the small resulting
 * polygon vertices cross the PCIe bus.
 */
struct MaskDecodeBuffers {
	int* d_vertex;          /* int[num_detections * 8] */
	unsigned int* d_fill;   /* unsigned int[num_detections] */
	unsigned int* d_border; /* unsigned int[num_detections] */
	int* d_count;           /* int[1] */
	int capacity;
};

static MaskDecodeBuffers g_mask_buf = {nullptr, nullptr, nullptr, nullptr, 0};

static int ensure_mask_decode_buffers(int max_detections) {
	if (max_detections <= g_mask_buf.capacity)
		return 0;
	auto safe_free = [](void* p) {
		if (p)
			(void)hipFree(p);
	};
	safe_free(g_mask_buf.d_vertex);
	safe_free(g_mask_buf.d_fill);
	safe_free(g_mask_buf.d_border);
	safe_free(g_mask_buf.d_count);
	g_mask_buf.capacity = 0;

	int cap = max_detections + 64;
	if (hipMalloc(&g_mask_buf.d_vertex, (size_t)cap * 8 * sizeof(int)) != hipSuccess || hipMalloc(&g_mask_buf.d_fill, (size_t)cap * sizeof(unsigned int)) != hipSuccess ||
	    hipMalloc(&g_mask_buf.d_border, (size_t)cap * sizeof(unsigned int)) != hipSuccess || hipMalloc(&g_mask_buf.d_count, sizeof(int)) != hipSuccess) {
		fprintf(stderr, "[magma_internimage] mask decode buffer allocation failed\n");
		return 1;
	}
	g_mask_buf.capacity = cap;
	return 0;
}

/* ─── to_primitives converter ──────────────────────────────────────
 *
 * Reads MagmaInferObjectGPU[] from params and produces rect + text
 * primitives.  The data layout is 6 floats per object:
 *   [class_id, score, x, y, width, height]
 *
 * When d_masks_gpu is present, launches decode_masks_kernel on GPU
 * to produce filled mask polygons — the full mask tensor stays on
 * GPU, only the resulting polygon vertices cross to CPU.
 */
extern "C" int magma_to_primitives(const MagmaToPrimitivesParams* params, MagmaPrimitiveList* out) {
	if (!params || !out)
		return -1;

	const void* data = params->cpu_data ? params->cpu_data : params->d_data;
	int data_size = params->cpu_data ? params->cpu_data_size : params->data_size;
	if (!data || data_size < 6 * (int)sizeof(float))
		return 0;

	int num = data_size / (6 * (int)sizeof(float));
	if (num > 500)
		num = 500;

	int has_gpu_masks = (params->d_masks_gpu != NULL && params->d_masks_bytes > 0 && params->d_objects_gpu != NULL);

	static const unsigned int palette[] = {
	    0xFF0000FF,
	    0xFF00FF00,
	    0xFFFF0000,
	    0xFFFFFF00,
	    0xFFFF00FF,
	    0xFF00FFFF,
	    0xFFFFA500,
	    0xFF800080,
	    0xFF87CEEB,
	    0xFFFFC0CB,
	    0xFF00FF80,
	    0xFF7FFF00,
	    0xFFFF1493,
	    0xFF00BFFF,
	    0xFF98FB98,
	    0xFFDDA0DD,
	};

	float sx = params->roi_x;
	float sy = params->roi_y;
	float sw = params->roi_w > 0 ? (float)params->roi_w : (float)params->source_width;
	float sh = params->roi_h > 0 ? (float)params->roi_h : (float)params->source_height;

	/* ─── GPU mask decode path ─────────────────────────────────────── */
	if (has_gpu_masks && num > 0) {
		int mask_pixels = params->mask_h * params->mask_w;
		if (mask_pixels > 0 && ensure_mask_decode_buffers(num) == 0) {
			int mask_h = params->mask_h;
			int mask_w = params->mask_w;

			/* Reset GPU counter */
			(void)hipMemsetAsync(g_mask_buf.d_count, 0, sizeof(int), (hipStream_t)params->stream);

			/* Launch decode kernel */
			int block = 128;
			int grid = (num + block - 1) / block;
			decode_masks_kernel<<<grid, block, 0, (hipStream_t)params->stream>>>((const float*)params->d_masks_gpu,
			                                                                     (const float*)params->d_objects_gpu,
			                                                                     num,
			                                                                     mask_h,
			                                                                     mask_w,
			                                                                     (int)params->source_width,
			                                                                     (int)params->source_height,
			                                                                     params->roi_x,
			                                                                     params->roi_y,
			                                                                     params->roi_w,
			                                                                     params->roi_h,
			                                                                     g_mask_buf.d_vertex,
			                                                                     g_mask_buf.d_fill,
			                                                                     g_mask_buf.d_border,
			                                                                     g_mask_buf.d_count);

			(void)hipStreamSynchronize((hipStream_t)params->stream);

			int num_generated = 0;
			(void)hipMemcpyDtoH(&num_generated, g_mask_buf.d_count, sizeof(int));
			if (num_generated > 0) {
				if (num_generated > num)
					num_generated = num;

				/* Read back polygon data */
				int* host_verts = (int*)malloc((size_t)num_generated * 8 * sizeof(int));
				unsigned int* host_fill = (unsigned int*)malloc((size_t)num_generated * sizeof(unsigned int));
				unsigned int* host_border = (unsigned int*)malloc((size_t)num_generated * sizeof(unsigned int));
				if (host_verts && host_fill && host_border) {
					(void)hipMemcpyDtoH(host_verts, g_mask_buf.d_vertex, (size_t)num_generated * 8 * sizeof(int));
					(void)hipMemcpyDtoH(host_fill, g_mask_buf.d_fill, (size_t)num_generated * sizeof(unsigned int));
					(void)hipMemcpyDtoH(host_border, g_mask_buf.d_border, (size_t)num_generated * sizeof(unsigned int));

					for (int gi = 0; gi < num_generated; gi++) {
						magma_primitive_list_add_polygon(out, host_verts + gi * 8, 4, host_fill[gi], host_border[gi]);
					}
				}
				free(host_verts);
				free(host_fill);
				free(host_border);

				/* Skip CPU rect generation below — masks already rendered */
				/* Still need labels though */
				for (int i = 0; i < num; i++) {
					const float* o = (const float*)data + i * 6;
					unsigned int cid = (unsigned int)o[0];
					float score = o[1];
					float x = o[2], y = o[3], w = o[4], h = o[5];
					if (w < 0.001f || h < 0.001f)
						continue;

					unsigned int color = palette[cid % 16];
					int rx = (int)(sx + x * sw);
					int ry = (int)(sy + y * sh);
					int rw = (int)(w * sw);
					int rh = (int)(h * sh);
					if (rw < 2 || rh < 2)
						continue;

					char label[MAGMA_LABEL_MAX];
					const char* name = internimage_lookup_label(cid, params->labels_path);
					int n;
					if (name) {
						n = snprintf(label, sizeof(label), "%s %.0f%%", name, score * 100.0f);
					} else {
						n = snprintf(label, sizeof(label), "%u %.0f%%", cid, score * 100.0f);
					}
					if (n > 0)
						magma_primitive_list_add_text(out, rx, ry - 12, color, label, n);
				}
				return 0;
			}
		}
	}

	/* ─── CPU fallback path (no GPU masks) ──────────────────────────── */
	for (int i = 0; i < num; i++) {
		const float* o = (const float*)data + i * 6;
		unsigned int cid = (unsigned int)o[0]; /* class_id stored as float */
		float score = o[1];
		float x = o[2];
		float y = o[3];
		float w = o[4];
		float h = o[5];

		if (w < 0.001f || h < 0.001f)
			continue;

		unsigned int color = palette[cid % 16];
		int rx = (int)(sx + x * sw);
		int ry = (int)(sy + y * sh);
		int rw = (int)(w * sw);
		int rh = (int)(h * sh);
		if (rw < 2 || rh < 2)
			continue;

		magma_primitive_list_add_rect(out, rx, ry, rw, rh, color, 0);

		char label[MAGMA_LABEL_MAX];
		const char* name = internimage_lookup_label(cid, params->labels_path);
		int n;
		if (name) {
			n = snprintf(label, sizeof(label), "%s %.0f%%", name, score * 100.0f);
		} else {
			n = snprintf(label, sizeof(label), "%u %.0f%%", cid, score * 100.0f);
		}
		if (n > 0)
			magma_primitive_list_add_text(out, rx, ry - 12, color, label, n);
	}
	return 0;
}
