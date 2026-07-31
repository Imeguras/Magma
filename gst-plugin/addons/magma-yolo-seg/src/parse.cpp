#include "magma_parser_api.h"
#include "magma-primitives.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>
#include <hip/hip_runtime.h>
#include <migraphx/migraphx.hpp>

extern "C" __global__ void transpose_col_to_row_kernel(const float*, float*, int, int);

extern "C" __global__ void decode_filter_kernel(
    const float*, float*, int*, int, int, int, int, float, float, int, int);

extern "C" __global__ void nms_mask_kernel(
    const float*, const float*, const float*,
    float*, int*, const int*, float*,
    int, float, int, int, int, int, int, int, int, int);

static inline void safe_free_device(void* p) {
    if (p) (void)hipFree(p);
}

extern "C" int magma_parse(MagmaParseParams* p) {
    if (!p || !p->d_objects || !p->d_num_detected)
        return 1;
    if (p->num_raw_outputs < 2)
        return 1;
    if (!p->d_raw_outputs || !p->output_shapes || !p->num_dims_list)
        return 1;
    if (!p->d_raw_outputs[0] || !p->d_raw_outputs[1])
        return 1;

    auto* shape0 = p->output_shapes[0];
    int ndim0 = p->num_dims_list[0];
    int N = 0, stride = 0;
    if (ndim0 == 3) {
        N = (int)shape0[2];
        stride = (int)shape0[1];
    } else {
        return 1;
    }

    auto* shape1 = p->output_shapes[1];
    int ndim1 = p->num_dims_list[1];
    int proto_c = 0, proto_h = 0, proto_w = 0;
    if (ndim1 >= 4) {
        proto_c = (int)shape1[1];
        proto_h = (int)shape1[2];
        proto_w = (int)shape1[3];
    } else if (ndim1 == 3) {
        proto_c = (int)shape1[0];
        proto_h = (int)shape1[1];
        proto_w = (int)shape1[2];
    } else {
        return 1;
    }

    int num_classes = stride - 4 - proto_c;
    int num_coeffs = proto_c;
    int max_out = p->max_detections > 0 ? p->max_detections : 100;
    int net_w = p->net_width > 0 ? p->net_width : 640;
    int net_h = p->net_height > 0 ? p->net_height : 640;
    hipStream_t stream = (hipStream_t)p->stream;

    float* d_rowmajor = nullptr;
    float* d_filtered = nullptr;
    int* d_counter = nullptr;
    float* d_masks = nullptr;
    int es = 7 + num_coeffs;

    hipError_t e;

    e = hipMalloc(&d_rowmajor, (size_t)N * stride * sizeof(float));
    if (e != hipSuccess) goto fail;

    {
        int total = stride * N;
        int block = 256;
        int grid = (total + block - 1) / block;
        transpose_col_to_row_kernel<<<grid, block, 0, stream>>>(
            (const float*)p->d_raw_outputs[0], d_rowmajor, stride, N);
    }

    e = hipMalloc(&d_filtered, (size_t)max_out * es * sizeof(float));
    if (e != hipSuccess) goto fail;

    e = hipMalloc(&d_counter, sizeof(int));
    if (e != hipSuccess) goto fail;
    e = hipMemsetAsync(d_counter, 0, sizeof(int), stream);
    if (e != hipSuccess) goto fail;

    {
        int block = 256;
        int grid = (N + block - 1) / block;
        decode_filter_kernel<<<grid, block, 0, stream>>>(
            d_rowmajor, d_filtered, d_counter,
            N, stride, num_classes, num_coeffs,
            p->confidence_thresh, (float)max_out, net_w, net_h);
    }

    e = hipMemsetAsync(p->d_num_detected, 0, sizeof(int), stream);
    if (e != hipSuccess) goto fail;

    p->mask_h = proto_h;
    p->mask_w = proto_w;

    d_masks = (float*)p->d_masks;

    {
        int block = 256;
        int grid = (max_out + block - 1) / block;
        nms_mask_kernel<<<grid, block, 0, stream>>>(
            d_filtered, d_rowmajor,
            (const float*)p->d_raw_outputs[1],
            (float*)p->d_objects, p->d_num_detected, d_counter,
            d_masks,
            max_out, p->nms_thresh,
            stride, num_classes, num_coeffs,
            proto_c, proto_h, proto_w,
            net_w, net_h);
    }

    e = hipStreamSynchronize(stream);
    if (e != hipSuccess) goto fail;

done:
    safe_free_device(d_rowmajor);
    safe_free_device(d_filtered);
    safe_free_device(d_counter);
    return (e == hipSuccess) ? 0 : 1;

fail:
    safe_free_device(d_rowmajor);
    safe_free_device(d_filtered);
    safe_free_device(d_counter);
    return 1;
}

extern "C" int magma_compile(const MagmaCompileParams* params) {
    if (!params || !params->onnx_path || !params->mxr_output_path)
        return 1;

    const char* precision = params->precision ? params->precision : "FP32";

    fprintf(stderr, "[magma_yolo_seg_compile] ONNX=%s MXR=%s precision=%s calib=%s batch=%d\n",
            params->onnx_path, params->mxr_output_path,
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
                        if (next == std::string::npos) break;
                        pos = next + 1;
                    }
                    if (shape.size() >= 2) {
                        opts.set_input_parameter_shape("images", shape);
                        fprintf(stderr, "[magma_yolo_seg_compile] set input shape: [");
                        for (auto d : shape) fprintf(stderr, "%zu,", d);
                        fprintf(stderr, "]\n");
                    }
                }
            }
        }

        migraphx::program prog = migraphx::parse_onnx(params->onnx_path, opts);
        fprintf(stderr, "[magma_yolo_seg_compile] ONNX parsed OK\n");

        if (strcmp(precision, "FP16") == 0) {
            migraphx::quantize_fp16(prog);
            fprintf(stderr, "[magma_yolo_seg_compile] Quantized to FP16\n");
        } else if (strcmp(precision, "INT8") == 0) {
            if (!params->calib_data_path) {
                fprintf(stderr, "[magma_yolo_seg_compile] INT8 requested but no calib-data-path\n");
                return 1;
            }

            std::string tmp_onnx = std::string(params->mxr_output_path) + ".quant.onnx";

            std::string script_path;
            const char* search_dirs[] = {
                "/usr/share/magma/scripts",
                ".",
                nullptr};
            for (const char** d = search_dirs; *d; ++d) {
                std::string candidate = std::string(*d) + "/quark_quantize.py";
                FILE* f = fopen(candidate.c_str(), "r");
                if (f) { fclose(f); script_path = candidate; break; }
            }

            if (script_path.empty()) {
                fprintf(stderr, "[magma_yolo_seg_compile] quark_quantize.py not found\n");
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

            std::string quark_python;
            const char* qp_env = std::getenv("QUARK_PYTHON");
            if (qp_env && qp_env[0]) {
                quark_python = qp_env;
            } else {
                const char* py_search[] = {
                    "/opt/magma-quark/bin/python3",
                    "quark-env/bin/python3",
                    "python3",
                    nullptr};
                for (const char** p = py_search; *p; ++p) {
                    FILE* f = fopen(*p, "r");
                    if (f) { fclose(f); quark_python = *p; break; }
                }
            }
            if (quark_python.empty()) quark_python = "python3";

            std::string cache_dir;
            const char* qcd_env = std::getenv("QUARK_CACHE_DIR");
            if (qcd_env && qcd_env[0]) cache_dir = std::string(" --cache-dir '") + qcd_env + "'";

            std::string cmd =
                std::string("'") + quark_python + "' '" + script_path + "'"
                " --onnx '" + params->onnx_path + "'"
                " --output '" + tmp_onnx + "'"
                " --calib-dir '" + params->calib_data_path + "'"
                " --input-shape " + shape_str +
                " --input-name images" +
                cache_dir +
                " 2>&1";

            fprintf(stderr, "[magma_yolo_seg_compile] Running Quark PTQ: %s\n", cmd.c_str());
            int ret = std::system(cmd.c_str());
            if (ret != 0) {
                fprintf(stderr, "[magma_yolo_seg_compile] Quark quantization failed (exit=%d)\n", ret);
                return 1;
            }

            {
                FILE* f = fopen(tmp_onnx.c_str(), "r");
                if (!f) {
                    fprintf(stderr, "[magma_yolo_seg_compile] Quark output not found: %s\n", tmp_onnx.c_str());
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
                            if (next == std::string::npos) break;
                            pos = next + 1;
                        }
                        if (shape.size() >= 2) {
                            qopts.set_input_parameter_shape("images", shape);
                        }
                    }
                }
            }
            prog = migraphx::parse_onnx(tmp_onnx.c_str(), qopts);
            fprintf(stderr, "[magma_yolo_seg_compile] Quantized ONNX parsed OK\n");
            std::remove(tmp_onnx.c_str());
        }

        migraphx::target tgt("gpu");
        migraphx::compile_options copts;
        prog.compile(tgt, copts);
        fprintf(stderr, "[magma_yolo_seg_compile] Compiled for GPU OK\n");

        migraphx::save(prog, params->mxr_output_path);
        fprintf(stderr, "[magma_yolo_seg_compile] Saved to %s\n", params->mxr_output_path);

        return 0;

    } catch (const std::exception& e) {
        fprintf(stderr, "[magma_yolo_seg_compile] FAILED: %s\n", e.what());
        return 1;
    }
}

extern "C" const char* magma_semantic_type = "magma.detection.yolo-seg.v1";

static const unsigned int palette[] = {
    0xFF0000FF, 0xFF00FF00, 0xFFFF0000, 0xFFFFFF00,
    0xFFFF00FF, 0xFF00FFFF, 0xFFFFA500, 0xFF800080,
    0xFF87CEEB, 0xFFFFC0CB, 0xFF00FF80, 0xFF7FFF00,
    0xFFFF1493, 0xFF00BFFF, 0xFF98FB98, 0xFFDDA0DD,
};

struct LabelCache {
    char path[4096];
    time_t mtime;
    char** names;
    int count;
    int valid;
};

static struct LabelCache g_label_cache = {{0}, 0, NULL, 0, 0};

static const char* yolo_lookup_label(unsigned int class_id,
                                      const char* labels_path) {
    if (!labels_path || !labels_path[0]) return NULL;

    struct stat st;
    if (stat(labels_path, &st) != 0) return NULL;

    if (g_label_cache.valid) {
        if (strcmp(g_label_cache.path, labels_path) == 0 &&
            g_label_cache.mtime == st.st_mtime) {
            if ((int)class_id < g_label_cache.count)
                return g_label_cache.names[class_id] ? g_label_cache.names[class_id] : "";
            return NULL;
        }
        for (int i = 0; i < g_label_cache.count; i++)
            free(g_label_cache.names[i]);
        free(g_label_cache.names);
        g_label_cache.names = NULL;
        g_label_cache.count = 0;
        g_label_cache.valid = 0;
    }

    strncpy(g_label_cache.path, labels_path, sizeof(g_label_cache.path) - 1);
    g_label_cache.path[sizeof(g_label_cache.path) - 1] = '\0';
    g_label_cache.mtime = st.st_mtime;

    FILE* f = fopen(labels_path, "r");
    if (!f) return NULL;

    int count = 0;
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        if (ch == '\n') count++;
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

struct MaskDecodeBuffers {
    int*        d_vertex;
    unsigned int* d_fill;
    unsigned int* d_border;
    int*        d_count;
    int         capacity;
};

static MaskDecodeBuffers g_mask_buf = {nullptr, nullptr, nullptr, nullptr, 0};

static int ensure_mask_decode_buffers(int max_detections) {
    if (max_detections <= g_mask_buf.capacity)
        return 0;
    auto sf = [](void* p) { if (p) (void)hipFree(p); };
    sf(g_mask_buf.d_vertex);
    sf(g_mask_buf.d_fill);
    sf(g_mask_buf.d_border);
    sf(g_mask_buf.d_count);
    g_mask_buf.capacity = 0;

    int cap = max_detections + 64;
    if (hipMalloc(&g_mask_buf.d_vertex, (size_t)cap * 8 * sizeof(int)) != hipSuccess ||
        hipMalloc(&g_mask_buf.d_fill,   (size_t)cap * sizeof(unsigned int)) != hipSuccess ||
        hipMalloc(&g_mask_buf.d_border, (size_t)cap * sizeof(unsigned int)) != hipSuccess ||
        hipMalloc(&g_mask_buf.d_count,  sizeof(int)) != hipSuccess) {
        fprintf(stderr, "[yolo_seg] mask decode buffer allocation failed\n");
        return 1;
    }
    g_mask_buf.capacity = cap;
    return 0;
}

extern "C" __global__ void decode_masks_kernel(
    const float* __restrict__ d_masks,
    const float* __restrict__ d_objects,
    int num_masks, int mask_h, int mask_w,
    int source_w, int source_h,
    int roi_x, int roi_y, int roi_w, int roi_h,
    int* __restrict__ d_vertex_out,
    unsigned int* __restrict__ d_fill_out,
    unsigned int* __restrict__ d_border_out,
    int* __restrict__ d_num_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_masks) return;

    const float* det = d_objects + i * 6;
    float bx = det[2], by = det[3], bw = det[4], bh = det[5];

    if (bw < 0.001f || bh < 0.001f) return;

    int mask_pixels = mask_h * mask_w;
    int min_r = mask_h, max_r = -1, min_c = mask_w, max_c = -1;
    for (int p = 0; p < mask_pixels; p++) {
        float v = 1.0f / (1.0f + expf(-d_masks[(size_t)i * mask_pixels + p]));
        if (v > 0.5f) {
            int r = p / mask_w, c = p % mask_w;
            if (r < min_r) min_r = r; if (r > max_r) max_r = r;
            if (c < min_c) min_c = c; if (c > max_c) max_c = c;
        }
    }
    if (min_r > max_r) return;

    float sx = (float)roi_x + bx * (float)source_w;
    float sy = (float)roi_y + by * (float)source_h;
    float sw = bw * (float)source_w;
    float sh = bh * (float)source_h;

    float rx0 = (float)min_c / (float)mask_w, ry0 = (float)min_r / (float)mask_h;
    float rx1 = (float)(max_c + 1) / (float)mask_w, ry1 = (float)(max_r + 1) / (float)mask_h;

    int x0 = (int)(sx + rx0 * sw), y0 = (int)(sy + ry0 * sh);
    int x1 = (int)(sx + rx1 * sw), y1 = (int)(sy + ry1 * sh);
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;

    int idx = atomicAdd(d_num_out, 1);
    int* v = d_vertex_out + idx * 8;
    v[0] = x0; v[1] = y0; v[2] = x1; v[3] = y0;
    v[4] = x1; v[5] = y1; v[6] = x0; v[7] = y1;

    unsigned int cid = (unsigned int)det[0];
    unsigned int color = palette[cid % 16];
    d_fill_out[idx]   = (color & 0xFFFFFF00) | 0x1A;
    d_border_out[idx] = color;
}

extern "C" int magma_to_primitives(
    const MagmaToPrimitivesParams* params,
    MagmaPrimitiveList* out)
{
    if (!params || !out) return -1;

    const void* data = params->cpu_data ? params->cpu_data : params->d_data;
    int data_size = params->cpu_data ? params->cpu_data_size : params->data_size;
    if (!data || data_size < 6 * (int)sizeof(float))
        return 0;

    int num = data_size / (6 * (int)sizeof(float));
    if (num > 500) num = 500;

    int has_gpu_masks = (params->d_masks_gpu != NULL && params->d_masks_bytes > 0 &&
                         params->d_objects_gpu != NULL);

    float sx = params->roi_x;
    float sy = params->roi_y;
    float sw = params->roi_w > 0 ? (float)params->roi_w : (float)params->source_width;
    float sh = params->roi_h > 0 ? (float)params->roi_h : (float)params->source_height;

    if (has_gpu_masks && num > 0) {
        int mask_pixels = params->mask_h * params->mask_w;
        if (mask_pixels > 0 && ensure_mask_decode_buffers(num) == 0) {
            int mask_h = params->mask_h;
            int mask_w = params->mask_w;

            (void)hipMemsetAsync(g_mask_buf.d_count, 0, sizeof(int), (hipStream_t)params->stream);

            int block = 128;
            int grid = (num + block - 1) / block;
            decode_masks_kernel<<<grid, block, 0, (hipStream_t)params->stream>>>(
                (const float*)params->d_masks_gpu,
                (const float*)params->d_objects_gpu,
                num, mask_h, mask_w,
                (int)params->source_width, (int)params->source_height,
                params->roi_x, params->roi_y,
                params->roi_w, params->roi_h,
                g_mask_buf.d_vertex,
                g_mask_buf.d_fill,
                g_mask_buf.d_border,
                g_mask_buf.d_count);

            (void)hipStreamSynchronize((hipStream_t)params->stream);

            int num_generated = 0;
            (void)hipMemcpyDtoH(&num_generated, g_mask_buf.d_count, sizeof(int));
            if (num_generated > 0) {
                if (num_generated > num) num_generated = num;

                int* host_verts = (int*)malloc((size_t)num_generated * 8 * sizeof(int));
                unsigned int* host_fill = (unsigned int*)malloc((size_t)num_generated * sizeof(unsigned int));
                unsigned int* host_border = (unsigned int*)malloc((size_t)num_generated * sizeof(unsigned int));
                if (host_verts && host_fill && host_border) {
                    (void)hipMemcpyDtoH(host_verts,   g_mask_buf.d_vertex, (size_t)num_generated * 8 * sizeof(int));
                    (void)hipMemcpyDtoH(host_fill,    g_mask_buf.d_fill,   (size_t)num_generated * sizeof(unsigned int));
                    (void)hipMemcpyDtoH(host_border,  g_mask_buf.d_border, (size_t)num_generated * sizeof(unsigned int));

                    for (int gi = 0; gi < num_generated; gi++) {
                        magma_primitive_list_add_polygon(out,
                            host_verts + gi * 8, 4,
                            host_fill[gi], host_border[gi]);
                    }
                }
                free(host_verts);
                free(host_fill);
                free(host_border);

                for (int i = 0; i < num; i++) {
                    const float* o = (const float*)data + i * 6;
                    unsigned int cid = (unsigned int)o[0];
                    float score = o[1];
                    float x = o[2], y = o[3], w = o[4], h = o[5];
                    if (w < 0.001f || h < 0.001f) continue;

                    unsigned int color = palette[cid % 16];
                    int rx = (int)(sx + x * sw);
                    int ry = (int)(sy + y * sh);
                    int rw = (int)(w * sw);
                    int rh = (int)(h * sh);
                    if (rw < 2 || rh < 2) continue;

                    char label[MAGMA_LABEL_MAX];
                    const char* name = yolo_lookup_label(cid, params->labels_path);
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

    for (int i = 0; i < num; i++) {
        const float* o = (const float*)data + i * 6;
        unsigned int cid = (unsigned int)o[0];
        float score = o[1];
        float x = o[2];
        float y = o[3];
        float w = o[4];
        float h = o[5];

        if (w < 0.001f || h < 0.001f) continue;

        unsigned int color = palette[cid % 16];
        int rx = (int)(sx + x * sw);
        int ry = (int)(sy + y * sh);
        int rw = (int)(w * sw);
        int rh = (int)(h * sh);
        if (rw < 2 || rh < 2) continue;

        magma_primitive_list_add_rect(out, rx, ry, rw, rh, color, 0);

        char label[MAGMA_LABEL_MAX];
        const char* name = yolo_lookup_label(cid, params->labels_path);
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
