#include "magma_parser_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <climits>
#include <strings.h>
#include <sys/stat.h>
#include <dirent.h>
#include <hip/hip_runtime.h>
#include <migraphx/migraphx.hpp>

/* kernel declarations */
extern "C" __global__ void transpose_col_to_row_kernel(const float*, float*, int, int);

extern "C" __global__ void decode_filter_kernel(const float*, float*, int*, int, int, int, float, float, int, int);

extern "C" __global__ void nms_compact_kernel(const float*, float*, int*, const int*, int, float, int, int);

static inline void safe_free_device(void* p) {
	if (p)
		(void)hipFree(p);
}

extern "C" int magma_parse(MagmaParseParams* p) {
	if (!p || !p->d_raw_output || !p->d_objects || !p->d_num_detected)
		return 1;

	hipStream_t stream = (hipStream_t)p->stream;

	int N = 1, stride = 1, num_classes = 80;
	bool col_major = false;
	float* d_work = (float*)p->d_raw_output;

	if (p->num_dims == 3) {
		int d1 = (int)p->output_shape[1];
		int d2 = (int)p->output_shape[2];
		if (d1 > d2) {
			N = d1;
			stride = d2;
		} else {
			N = d2;
			stride = d1;
			col_major = true;
		}
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
	int net_w = p->net_width > 0 ? p->net_width : 640;
	int net_h = p->net_height > 0 ? p->net_height : 640;
	int block = 256;
	int grid = (N + block - 1) / block;

	float* d_transposed = nullptr;
	float* d_intermediate = nullptr;
	int* d_counter = nullptr;
	int* d_out_counter = nullptr;
	size_t inter_bytes = 0;
	int nms_grid = 0;

	hipError_t e;

	if (col_major) {
		size_t total = (size_t)N * stride;
		e = hipMalloc(&d_transposed, total * sizeof(float));
		if (e != hipSuccess)
			goto fail;
		int tgrid = (total + block - 1) / block;
		transpose_col_to_row_kernel<<<tgrid, block, 0, stream>>>((const float*)p->d_raw_output, d_transposed, N, stride);
		d_work = d_transposed;
	}

	inter_bytes = (size_t)N * 7 * sizeof(float);
	e = hipMalloc(&d_intermediate, inter_bytes);
	if (e != hipSuccess)
		goto fail;

	e = hipMalloc(&d_counter, sizeof(int));
	if (e != hipSuccess)
		goto fail;

	e = hipMalloc(&d_out_counter, sizeof(int));
	if (e != hipSuccess)
		goto fail;

	e = hipMemsetAsync(d_counter, 0, sizeof(int), stream);
	if (e != hipSuccess)
		goto fail;

	decode_filter_kernel<<<grid, block, 0, stream>>>((const float*)d_work, d_intermediate, d_counter, N, stride, num_classes, p->confidence_thresh, (float)max_out, net_w, net_h);

	/* No sync between decode_filter and nms_compact — same stream guarantees ordering.
	   nms_compact reads d_counter (M survivors) from device, no host round-trip. */

	e = hipMemsetAsync(d_out_counter, 0, sizeof(int), stream);
	if (e != hipSuccess)
		goto fail;

	nms_grid = (max_out + block - 1) / block;
	nms_compact_kernel<<<nms_grid, block, 0, stream>>>(d_intermediate, (float*)p->d_objects, d_out_counter, d_counter, max_out, p->nms_thresh, net_w, net_h);

	e = hipMemcpyDtoHAsync(p->d_num_detected, d_out_counter, sizeof(int), stream);
	if (e != hipSuccess)
		goto fail;

	e = hipStreamSynchronize(stream);
	if (e != hipSuccess)
		goto fail;

done:
	safe_free_device(d_transposed);
	safe_free_device(d_intermediate);
	safe_free_device(d_counter);
	safe_free_device(d_out_counter);
	return (e == hipSuccess) ? 0 : 1;

fail:
	safe_free_device(d_transposed);
	safe_free_device(d_intermediate);
	safe_free_device(d_counter);
	safe_free_device(d_out_counter);
	return 1;
}

extern "C" int magma_compile(const MagmaCompileParams* params) {
	if (!params || !params->onnx_path || !params->mxr_output_path)
		return 1;

	const char* precision = params->precision ? params->precision : "FP32";

	fprintf(stderr,
	        "[yolov8_compile] ONNX=%s MXR=%s precision=%s calib=%s batch=%d\n",
	        params->onnx_path,
	        params->mxr_output_path,
	        precision,
	        params->calib_data_path ? params->calib_data_path : "(none)",
	        params->batch_size);

	try {
		migraphx::onnx_options opts;
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
						opts.set_input_parameter_shape("images", shape);
						fprintf(stderr, "[yolov8_compile] set input shape: [");
						for (auto d : shape)
							fprintf(stderr, "%zu,", d);
						fprintf(stderr, "]\n");
					}
				}
			}
		}

		migraphx::program prog = migraphx::parse_onnx(params->onnx_path, opts);
		fprintf(stderr, "[yolov8_compile] ONNX parsed OK\n");

		if (strcmp(precision, "FP16") == 0) {
			migraphx::quantize_fp16(prog);
			fprintf(stderr, "[yolov8_compile] Quantized to FP16\n");
		} else if (strcmp(precision, "INT8") == 0) {
			if (!params->calib_data_path) {
				fprintf(stderr, "[yolov8_compile] INT8 requested but no calib-data-path\n");
				return 1;
			}

			std::string tmp_onnx = std::string(params->mxr_output_path) + ".quant.onnx";

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
				fprintf(stderr, "[yolov8_compile] quark_quantize.py not found\n");
				return 1;
			}

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

			/* Find a Python interpreter that has Quark installed */
			std::string quark_python;
			const char* qp_env = std::getenv("QUARK_PYTHON");
			if (qp_env && qp_env[0]) {
				quark_python = qp_env;
			}
			if (quark_python.empty()) {
				/* Check conda/venv environments from env vars first */
				const char* conda_prefix = std::getenv("CONDA_PREFIX");
				if (conda_prefix && conda_prefix[0]) {
					std::string cp = std::string(conda_prefix) + "/bin/python3";
					FILE* f = fopen(cp.c_str(), "r");
					if (f) {
						fclose(f);
						quark_python = cp;
					}
				}
			}
			if (quark_python.empty()) {
				const char* venv_dir = std::getenv("VIRTUAL_ENV");
				if (venv_dir && venv_dir[0]) {
					std::string vp = std::string(venv_dir) + "/bin/python3";
					FILE* f = fopen(vp.c_str(), "r");
					if (f) {
						fclose(f);
						quark_python = vp;
					}
				}
			}
			if (quark_python.empty()) {
				const char* fallback_search[] = {"/opt/magma-quark/bin/python3", "quark-env/bin/python3", nullptr};
				for (const char** p = fallback_search; *p; ++p) {
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

			/* Verify this Python can import Quark */
			{
				std::string check_cmd = std::string("'") + quark_python + "' -c \"import quark.onnx\" 2>/dev/null";
				int check_ret = std::system(check_cmd.c_str());
				if (check_ret != 0) {
					fprintf(stderr, "[yolov8_compile] Quark not available in '%s'\n", quark_python.c_str());
					quark_python.clear();
				}
			}

			if (!quark_python.empty()) {
				/* ── Quark PTQ path ── */
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
				                  shape_str + " --input-name images" + cache_dir + " 2>&1";

				fprintf(stderr, "[yolov8_compile] Running Quark PTQ: %s\n", cmd.c_str());
				int ret = std::system(cmd.c_str());
				if (ret != 0) {
					fprintf(stderr, "[yolov8_compile] Quark quantization failed (exit=%d)\n", ret);
					return 1;
				}

				{
					FILE* f = fopen(tmp_onnx.c_str(), "r");
					if (!f) {
						fprintf(stderr, "[yolov8_compile] Quark output not found: %s\n", tmp_onnx.c_str());
						return 1;
					}
					fclose(f);
				}

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
								qopts.set_input_parameter_shape("images", shape);
							}
						}
					}
				}
				prog = migraphx::parse_onnx(tmp_onnx.c_str(), qopts);
				fprintf(stderr, "[yolov8_compile] Quantized ONNX parsed OK\n");
				std::remove(tmp_onnx.c_str());
			} else {
				/* ── MIGraphX native INT8 path (no Quark needed) ── */
				fprintf(stderr, "[yolov8_compile] Using MIGraphX native INT8 quantization\n");

				DIR* dir = opendir(params->calib_data_path);
				if (!dir) {
					fprintf(stderr, "[yolov8_compile] Cannot open calib dir: %s\n", params->calib_data_path);
					return 1;
				}

				int model_h = 640, model_w = 640;
				if (params->extras) {
					for (int i = 0; params->extras[i]; i++) {
						std::string kv(params->extras[i]);
						auto eq = kv.find('=');
						if (eq != std::string::npos && kv.substr(0, eq) == "input-shape") {
							std::string dims = kv.substr(eq + 1);
							size_t pos = 0;
							std::vector<int> shape;
							while (pos < dims.size()) {
								size_t next = dims.find(',', pos);
								shape.push_back(std::stoi(dims.substr(pos, next - pos)));
								if (next == std::string::npos)
									break;
								pos = next + 1;
							}
							if (shape.size() >= 4) {
								model_h = shape[2];
								model_w = shape[3];
							}
							break;
						}
					}
				}

				int tensor_size = 3 * model_h * model_w;
				int max_samples = 64;
				std::vector<std::vector<float>> samples;
				int img_count = 0;

				struct dirent* entry;
				while ((entry = readdir(dir)) != NULL && img_count < max_samples) {
					const char* name = entry->d_name;
					const char* ext = strrchr(name, '.');
					if (!ext)
						continue;
					if (strcasecmp(ext, ".png") != 0 && strcasecmp(ext, ".jpg") != 0 && strcasecmp(ext, ".jpeg") != 0 && strcasecmp(ext, ".bmp") != 0)
						continue;

					std::string full_path = std::string(params->calib_data_path) + "/" + name;
					int w, h, ch;
					unsigned char* img = stbi_load(full_path.c_str(), &w, &h, &ch, 3);
					if (!img)
						continue;

					std::vector<float> tensor(tensor_size);
					float sx = (float)model_w / (float)w;
					float sy = (float)model_h / (float)h;
					float scale = std::max(sx, sy);
					int new_w = (int)(w * scale);
					int new_h = (int)(h * scale);
					int ox = (new_w - model_w) / 2;
					int oy = (new_h - model_h) / 2;

					/* Simple bilinear resize + letterbox + NCHW */
					for (int c = 0; c < 3; c++) {
						for (int r = 0; r < model_h; r++) {
							for (int col = 0; col < model_w; col++) {
								float src_r = ((float)(r + oy) / (float)new_h) * h;
								float src_c = ((float)(col + ox) / (float)new_w) * w;
								int sr = (int)src_r, sc = (int)src_c;
								if (sr >= h - 1)
									sr = h - 2;
								if (sc >= w - 1)
									sc = w - 2;
								float fr = src_r - sr, fc = src_c - sc;
								float v = (1 - fr) * (1 - fc) * img[(sr * w + sc) * 3 + c] + fr * (1 - fc) * img[((sr + 1) * w + sc) * 3 + c] + (1 - fr) * fc * img[(sr * w + sc + 1) * 3 + c] +
								          fr * fc * img[((sr + 1) * w + sc + 1) * 3 + c];
								tensor[c * model_h * model_w + r * model_w + col] = v / 255.0f;
							}
						}
					}
					stbi_image_free(img);
					samples.push_back(std::move(tensor));
					img_count++;
					fprintf(stderr, "[yolov8_compile] Loaded calib image %d: %s (%dx%d)\n", img_count, name, w, h);
				}
				closedir(dir);

				if (samples.empty()) {
					fprintf(stderr, "[yolov8_compile] No calibration images found in %s\n", params->calib_data_path);
					return 1;
				}
				fprintf(stderr, "[yolov8_compile] Loaded %zu calibration images\n", samples.size());

				/* Build calibration data for MIGraphX */
				migraphx::target tgt("gpu");
				migraphx::quantize_int8_options qopts;
				migraphx::shape input_shape(migraphx_shape_float_type, {1, 3, (std::size_t)model_h, (std::size_t)model_w});
				for (auto& s : samples) {
					migraphx::program_parameters pp;
					pp.add("images", migraphx::argument(input_shape, s.data()));
					qopts.add_calibration_data(pp);
				}

				migraphx::quantize_int8(prog, tgt, qopts);
				fprintf(stderr, "[yolov8_compile] MIGraphX native INT8 quantize OK\n");
			}
		}

		migraphx::target tgt("gpu");
		migraphx::compile_options copts;
		prog.compile(tgt, copts);
		fprintf(stderr, "[yolov8_compile] Compiled for GPU OK\n");

		migraphx::save(prog, params->mxr_output_path);
		fprintf(stderr, "[yolov8_compile] Saved to %s\n", params->mxr_output_path);

		return 0;

	} catch (const std::exception& e) {
		fprintf(stderr, "[yolov8_compile] FAILED: %s\n", e.what());
		return 1;
	}
}
