#include "mgmosd.hpp"
#include "magma-hip-stream.hpp"
#include "magma-meta.h"
#include "magma-primitives.h"
#include "magma-infer-meta.h"

#include <cstring>
#include <cstdio>
#include <string>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

GST_DEBUG_CATEGORY_STATIC(magma_osd_debug);
#define GST_CAT_DEFAULT magma_osd_debug

/* ─── properties ──────────────────────────────────────────────────── */
enum {
    PROP_0,
    PROP_SHOW_LABELS,
    PROP_LINE_WIDTH,
    PROP_ROI_X, PROP_ROI_Y, PROP_ROI_W, PROP_ROI_H,
    PROP_MAX_PRIMITIVES,
    PROP_PALETTE_BY_TRACK,
    PROP_LABELS_FILE,
};

G_DEFINE_TYPE(GstMagmaOsd, gst_magma_osd, GST_TYPE_BASE_TRANSFORM)

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12;"
                    "video/x-raw,format=(string)NV12"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12;"
                    "video/x-raw,format=(string)NV12"));

/* ─── colour palette (RGBA, converted to YUV in kernels) ──────────── */
#define NUM_PALETTE 16

/* ─── kernel directory lookup ─────────────────────────────────────── */
static std::string find_kernel_dir(void) {
    const char* env = getenv("MAGMA_KERNEL_DIR");
    if (env) return env;
    return MAGMA_KERNEL_SRC_DIR;
}

/* ─── compile all primitive kernels ───────────────────────────────── */
static gboolean compile_primitive_kernels(GstMagmaOsd* self) {
    std::string dir = find_kernel_dir();
    std::string kpath = dir + "/osd_primitives_kernels.hip";
    std::string cpath = dir + "/common.hip";

    static const char* entry_points[] = {
        "draw_rects_kernel",
        "draw_polylines_kernel",
        "draw_polygons_kernel",
        "draw_points_kernel",
        "draw_text_kernel",
        "draw_arrows_kernel",
        nullptr, /* ticker — not yet implemented */
    };



    HipKernel k = compile_kernel(kpath.c_str(), entry_points[0], cpath.c_str());
    if (!k.func) {
        GST_ERROR_OBJECT(self, "failed to compile primitive kernels from %s", kpath.c_str());
        return FALSE;
    }
    self->kernel_module = k.module;
    self->kernel_funcs[0] = k.func;

    /* Extract remaining entry points from the same module */
    for (int i = 1; entry_points[i]; i++) {
        hipError_t e = hipModuleGetFunction(&self->kernel_funcs[i],
                                             self->kernel_module,
                                             entry_points[i]);
        if (e != hipSuccess) {
            GST_WARNING_OBJECT(self, "hipModuleGetFunction(%s) failed: %s",
                               entry_points[i], hipGetErrorString(e));
            self->kernel_funcs[i] = nullptr;
        } else {
            GST_INFO_OBJECT(self, "loaded kernel: %s", entry_points[i]);
        }
    }

    GST_INFO_OBJECT(self, "OSD primitive kernels compiled from %s", kpath.c_str());
    return TRUE;
}

/* ─── allocate device buffers ─────────────────────────────────────── */
static gboolean allocate_device_buffers(GstMagmaOsd* self) {
    int max_p = (int)self->max_primitives;
    if (max_p < 64) max_p = 64;

    hipError_t e;

    e = hipMalloc(&self->d_rects,     (size_t)max_p * sizeof(MagmaRectGpu));
    if (e != hipSuccess) goto fail;
    e = hipMalloc(&self->d_polylines, (size_t)max_p * sizeof(MagmaPolylineGpu));
    if (e != hipSuccess) goto fail;
    e = hipMalloc(&self->d_polygons,  (size_t)max_p * sizeof(MagmaPolygonGpu));
    if (e != hipSuccess) goto fail;
    e = hipMalloc(&self->d_points,    (size_t)max_p * sizeof(MagmaPointGpu));
    if (e != hipSuccess) goto fail;
    e = hipMalloc(&self->d_texts,     (size_t)max_p * sizeof(MagmaTextGpu));
    if (e != hipSuccess) goto fail;
    e = hipMalloc(&self->d_arrows,    (size_t)max_p * sizeof(MagmaArrowGpu));
    if (e != hipSuccess) goto fail;

    /* Vertex arena: 64K ints initially (grows on demand) */
    self->d_vertex_arena_bytes = 65536 * (int)sizeof(int);
    e = hipMalloc(&self->d_vertex_arena, (size_t)self->d_vertex_arena_bytes);
    if (e != hipSuccess) goto fail;

    return TRUE;

fail:
    GST_ERROR_OBJECT(self, "hipMalloc failed: %s", hipGetErrorString(e));
    return FALSE;
}

/* ─── start (READY → PAUSED) ──────────────────────────────────────── */
static gboolean gst_magma_osd_start(GstBaseTransform* trans) {
    GstMagmaOsd* self = GST_MAGMA_OSD(trans);

    self->hip_stream = magma_get_shared_hip_stream();
    if (!self->hip_stream) {
        GST_ERROR_OBJECT(self, "magma_get_shared_hip_stream failed");
        return FALSE;
    }

    if (!compile_primitive_kernels(self))
        return FALSE;

    if (!allocate_device_buffers(self)) {
        (void)hipModuleUnload(self->kernel_module);
        self->kernel_module = nullptr;
        memset(self->kernel_funcs, 0, sizeof(self->kernel_funcs));
        return FALSE;
    }

    magma_primitive_list_init(&self->primitives);
    self->kernel_ready = TRUE;
    return TRUE;
}

/* ─── stop (PAUSED → READY) ───────────────────────────────────────── */
static gboolean gst_magma_osd_stop(GstBaseTransform* trans) {
    GstMagmaOsd* self = GST_MAGMA_OSD(trans);

    auto safe_free = [](hipDeviceptr_t& p) {
        if (p) { (void)hipFree(p); p = nullptr; }
    };

    safe_free(self->d_rects);
    safe_free(self->d_polylines);
    safe_free(self->d_polygons);
    safe_free(self->d_points);
    safe_free(self->d_texts);
    safe_free(self->d_arrows);
    safe_free(self->d_vertex_arena);
    self->d_vertex_arena_bytes = 0;

    if (self->d_input_upload) {
        (void)hipFree(self->d_input_upload);
        self->d_input_upload = 0;
    }

    if (self->external_memory) {
        (void)hipDestroyExternalMemory(self->external_memory);
        self->external_memory = nullptr;
        self->d_image = 0;
    }

    if (self->kernel_module) {
        (void)hipModuleUnload(self->kernel_module);
        self->kernel_module = nullptr;
        memset(self->kernel_funcs, 0, sizeof(self->kernel_funcs));
        self->kernel_ready = FALSE;
    }

    magma_primitive_list_destroy(&self->primitives);
    return TRUE;
}

/* ─── import DMABuf → HIP ─────────────────────────────────────────── */
static hipExternalMemory_t import_dmabuf(int fd, gsize size, hipDeviceptr_t* d_ptr) {
    int hip_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (hip_fd < 0) return nullptr;

    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = hip_fd;
    desc.size = size;
    hipExternalMemory_t ext;
    hipError_t e = hipImportExternalMemory(&ext, &desc);
    close(hip_fd);
    if (e != hipSuccess) return nullptr;

    hipExternalMemoryBufferDesc bdesc{};
    bdesc.offset = 0;
    bdesc.size = size;
    e = hipExternalMemoryGetMappedBuffer(d_ptr, ext, &bdesc);
    if (e != hipSuccess) { (void)hipDestroyExternalMemory(ext); return nullptr; }
    return ext;
}

/* ─── get GPU frame pointer from buffer (reusable path) ───────────── */
static hipDeviceptr_t get_frame_pointer(GstMagmaOsd* self, GstBuffer* buf,
                                        gsize frame_bytes,
                                        int* out_y_stride,
                                        int* out_uv_stride,
                                        size_t* out_y_off,
                                        size_t* out_uv_off) {
    int y_stride = self->in_width;
    int uv_stride = y_stride;
    size_t y_off = 0;
    size_t uv_off = (size_t)y_stride * self->in_height;

    GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
    if (vmeta && vmeta->n_planes >= 2) {
        y_stride = vmeta->stride[0];
        uv_stride = vmeta->stride[1];
        y_off = vmeta->offset[0];
        uv_off = vmeta->offset[1];
    }

    *out_y_stride = y_stride;
    *out_uv_stride = uv_stride;
    *out_y_off = y_off;
    *out_uv_off = uv_off;

    hipDeviceptr_t d_frame = 0;

    MagmaHipMeta* hmeta = magma_buffer_get_hip_meta(buf);
    if (hmeta) {
        d_frame = hmeta->d_ptr;
    } else {
        GstMemory* mem = gst_buffer_peek_memory(buf, 0);
        if (mem && gst_is_dmabuf_memory(mem)) {
            gsize bytes = gst_memory_get_sizes(mem, NULL, NULL);
            if (bytes == 0) bytes = frame_bytes;

            gint raw_fd = gst_dmabuf_memory_get_fd(mem);
            hipExternalMemory_t new_ext = import_dmabuf(raw_fd, bytes, &d_frame);
            if (new_ext && d_frame) {
                if (self->external_memory)
                    (void)hipDestroyExternalMemory(self->external_memory);
                self->external_memory = new_ext;
            }
        }
        if (!d_frame) {
            GstMapInfo in_map;
            if (gst_buffer_map(buf, &in_map, GST_MAP_READ)) {
                if (!self->d_input_upload)
                    (void)hipMalloc(&self->d_input_upload, frame_bytes);
                if (self->d_input_upload) {
                    (void)hipMemcpy(self->d_input_upload, in_map.data, frame_bytes, hipMemcpyHostToDevice);
                    d_frame = self->d_input_upload;
                }
                gst_buffer_unmap(buf, &in_map);
            }
        }
    }

    return d_frame;
}

/* ─── upload primitives to GPU and launch kernels ─────────────────── */
static void dispatch_primitives(GstMagmaOsd* self,
                                unsigned char* d_y, unsigned char* d_uv,
                                int y_stride, int uv_stride) {
    MagmaPrimitiveList* pl = &self->primitives;
    if (pl->total_primitives == 0) return;

    hipStream_t s = self->hip_stream;
    int block = 64;

    /* Upload vertex arena first (shared across polylines & polygons) */
    if (pl->vertex_arena_used > 0) {
        size_t vb = (size_t)pl->vertex_arena_used * sizeof(int);
        if ((int)vb > self->d_vertex_arena_bytes) {
            /* Grow device buffer — should be rare with 64K int default */
            (void)hipFree(self->d_vertex_arena);
            self->d_vertex_arena_bytes = (int)(vb * 2);
            (void)hipMalloc(&self->d_vertex_arena, (size_t)self->d_vertex_arena_bytes);
        }
        hipMemcpyHtoDAsync(self->d_vertex_arena, pl->vertex_arena, vb, s);
    }

    /* ── rects ── */
    if (pl->num_rects > 0 && self->kernel_funcs[MAGMA_PRIMITIVE_RECT]) {
        size_t bytes = (size_t)pl->num_rects * sizeof(MagmaRectGpu);
        hipMemcpyHtoDAsync(self->d_rects, pl->rects, bytes, s);
        int n = pl->num_rects;
        int grid = (n + block - 1) / block;
        void* args[] = { &d_y, &d_uv, &y_stride, &uv_stride,
                         &self->in_width, &self->in_height,
                         &self->d_rects, &n, &self->line_width };
        hipModuleLaunchKernel(self->kernel_funcs[MAGMA_PRIMITIVE_RECT],
                              grid, 1, 1, block, 1, 1, 0, s, args, nullptr);
    }

    /* ── polylines ── */
    if (pl->num_polylines > 0 && self->kernel_funcs[MAGMA_PRIMITIVE_POLYLINE]) {
        size_t bytes = (size_t)pl->num_polylines * sizeof(MagmaPolylineGpu);
        hipMemcpyHtoDAsync(self->d_polylines, pl->polylines, bytes, s);
        int n = pl->num_polylines;
        int grid = (n + block - 1) / block;
        int arena_bytes = self->d_vertex_arena_bytes;
        void* args[] = { &d_y, &d_uv, &y_stride, &uv_stride,
                         &self->in_width, &self->in_height,
                         &self->d_polylines, &n,
                         &self->d_vertex_arena, &arena_bytes };
        hipModuleLaunchKernel(self->kernel_funcs[MAGMA_PRIMITIVE_POLYLINE],
                              grid, 1, 1, block, 1, 1, 0, s, args, nullptr);
    }

    /* ── polygons ── */
    if (pl->num_polygons > 0 && self->kernel_funcs[MAGMA_PRIMITIVE_POLYGON]) {
        size_t bytes = (size_t)pl->num_polygons * sizeof(MagmaPolygonGpu);
        hipMemcpyHtoDAsync(self->d_polygons, pl->polygons, bytes, s);
        int n = pl->num_polygons;
        int grid = (n + block - 1) / block;
        int arena_bytes = self->d_vertex_arena_bytes;
        void* args[] = { &d_y, &d_uv, &y_stride, &uv_stride,
                         &self->in_width, &self->in_height,
                         &self->d_polygons, &n,
                         &self->d_vertex_arena, &arena_bytes };
        hipModuleLaunchKernel(self->kernel_funcs[MAGMA_PRIMITIVE_POLYGON],
                              grid, 1, 1, block, 1, 1, 0, s, args, nullptr);
    }

    /* ── points ── */
    if (pl->num_points > 0 && self->kernel_funcs[MAGMA_PRIMITIVE_POINT]) {
        size_t bytes = (size_t)pl->num_points * sizeof(MagmaPointGpu);
        hipMemcpyHtoDAsync(self->d_points, pl->points, bytes, s);
        int n = pl->num_points;
        int grid = (n + block - 1) / block;
        void* args[] = { &d_y, &d_uv, &y_stride, &uv_stride,
                         &self->in_width, &self->in_height,
                         &self->d_points, &n };
        hipModuleLaunchKernel(self->kernel_funcs[MAGMA_PRIMITIVE_POINT],
                              grid, 1, 1, block, 1, 1, 0, s, args, nullptr);
    }

    /* ── text ── */
    if (pl->num_texts > 0 && self->kernel_funcs[MAGMA_PRIMITIVE_TEXT]) {
        size_t bytes = (size_t)pl->num_texts * sizeof(MagmaTextGpu);
        hipMemcpyHtoDAsync(self->d_texts, pl->texts, bytes, s);
        int n = pl->num_texts;
        int grid = (n + block - 1) / block;
        void* args[] = { &d_y, &d_uv, &y_stride, &uv_stride,
                         &self->in_width, &self->in_height,
                         &self->d_texts, &n };
        hipModuleLaunchKernel(self->kernel_funcs[MAGMA_PRIMITIVE_TEXT],
                              grid, 1, 1, block, 1, 1, 0, s, args, nullptr);
    }

    /* ── arrows ── */
    if (pl->num_arrows > 0 && self->kernel_funcs[MAGMA_PRIMITIVE_ARROW]) {
        size_t bytes = (size_t)pl->num_arrows * sizeof(MagmaArrowGpu);
        hipMemcpyHtoDAsync(self->d_arrows, pl->arrows, bytes, s);
        int n = pl->num_arrows;
        int grid = (n + block - 1) / block;
        void* args[] = { &d_y, &d_uv, &y_stride, &uv_stride,
                         &self->in_width, &self->in_height,
                         &self->d_arrows, &n };
        hipModuleLaunchKernel(self->kernel_funcs[MAGMA_PRIMITIVE_ARROW],
                              grid, 1, 1, block, 1, 1, 0, s, args, nullptr);
    }
}

/* ─── convert legacy MagmaInferenceMeta → primitives (backwards compat) ── */
static int legacy_detection_to_primitives(GstMagmaOsd* self,
                                          MagmaInferenceMeta* m,
                                          MagmaPrimitiveList* pl) {
    if (!m || m->num_objects == 0 || !m->objects_gpu)
        return 0;

    GstMapInfo obj_info;
    if (!gst_memory_map(m->objects_gpu, &obj_info, GST_MAP_READ))
        return 0;

    MagmaInferObjectGPU* objects = (MagmaInferObjectGPU*)obj_info.data;
    int num = (int)m->num_objects;
    if (num > (int)self->max_primitives)
        num = (int)self->max_primitives;

    /* Coordinate remapping */
    int off_x, off_y, map_w, map_h;
    if (m->model_width > 0 && m->model_height > 0) {
        off_x = m->roi_x;
        off_y = m->roi_y;
        map_w = m->roi_w > 0 ? m->roi_w : self->in_width;
        map_h = m->roi_h > 0 ? m->roi_h : self->in_height;
    } else {
        off_x = self->roi_x;
        off_y = self->roi_y;
        map_w = self->roi_w > 0 ? (int)self->roi_w : self->in_width;
        map_h = self->roi_h > 0 ? (int)self->roi_h : self->in_height;
    }

    static const uint32_t palette_rgba[] = {
        0xFF0000FF, 0xFF00FF00, 0xFFFF0000, 0xFFFFFF00,
        0xFFFF00FF, 0xFF00FFFF, 0xFFFFA500, 0xFF800080,
        0xFF87CEEB, 0xFFFFC0CB, 0xFF00FF80, 0xFF7FFF00,
        0xFFFF1493, 0xFF00BFFF, 0xFF98FB98, 0xFFDDA0DD,
    };

    for (int i = 0; i < num; i++) {
        MagmaInferObjectGPU* obj = &objects[i];
        int sx = off_x + (int)(obj->x * map_w);
        int sy = off_y + (int)(obj->y * map_h);
        int sw = (int)(obj->width  * map_w);
        int sh = (int)(obj->height * map_h);
        if (sw < 2 || sh < 2) continue;

        guint cid = self->palette_by_track ? 0 : obj->class_id;
        uint32_t color = palette_rgba[cid % NUM_PALETTE];

        if (magma_primitive_list_add_rect(pl, sx, sy, sw, sh, color, 0) != 0)
            break;

        if (self->show_labels) {
            char label[MAGMA_LABEL_MAX];
            /* Use labels file if available, otherwise numeric class_id */
            const char* name = magma_lookup_class_name(obj->class_id, self->labels_file);
            int n;
            if (name) {
                n = snprintf(label, sizeof(label), "%s %.0f%%",
                             name, obj->confidence * 100.0f);
            } else {
                n = snprintf(label, sizeof(label), "%u %.0f%%",
                             obj->class_id, obj->confidence * 100.0f);
            }
            if (n > 0)
                magma_primitive_list_add_text(pl, sx, sy - 12, color, label, n);
        }
    }

    gst_memory_unmap(m->objects_gpu, &obj_info);
    return 0;
}

/* ─── Extract GPU data from a GstMemory with DMABuf or CPU fallback ─── */
static int map_gpu_memory(GstMemory* mem, GstMapInfo* info,
                          const void** d_data, int* data_size,
                          const void** cpu_data, int* cpu_data_size) {
    if (!mem) return -1;

    gsize sz = gst_memory_get_sizes(mem, NULL, NULL);
    *data_size = (int)sz;
    *d_data = nullptr;
    *cpu_data = nullptr;
    *cpu_data_size = 0;

    /* Try DMABuf path for GPU access */
    if (gst_is_dmabuf_memory(mem)) {
        /* The FD-based import is done by the caller — we just note it's GPU-accessible */
        *d_data = (const void*)(uintptr_t)1; /* non-null marker */
    }

    /* Map for CPU access (always available — converter uses whichever it prefers) */
    if (gst_memory_map(mem, info, GST_MAP_READ)) {
        *cpu_data = info->data;
        *cpu_data_size = (int)info->size;
    }

    return 0;
}

/* ─── transform_ip: main render entry point ───────────────────────── */
static GstFlowReturn gst_magma_osd_transform_ip(GstBaseTransform* trans,
                                                 GstBuffer* buf) {
    GstMagmaOsd* self = GST_MAGMA_OSD(trans);
    if (!self->kernel_ready)
        return GST_FLOW_OK;

    gsize frame_bytes = (gsize)self->in_width * self->in_height * 3 / 2;

    /* Reset primitive list for this frame */
    magma_primitive_list_reset(&self->primitives);

    /* ─── Path 1: Semantic meta (model-agnostic) ─────────────────── */
    gpointer state = NULL;
    GstMeta* cur;
    while ((cur = gst_buffer_iterate_meta(buf, &state)) != NULL) {
        if (cur->info->api != MAGMA_SEMANTIC_META_API_TYPE)
            continue;

        MagmaSemanticMeta* smeta = (MagmaSemanticMeta*)cur;
        const char* tid_str = g_quark_to_string(smeta->type_id);
        if (!tid_str) continue;

        MagmaToPrimitivesFunc func = magma_lookup_to_primitives(tid_str);
        if (!func) continue;

        GstMapInfo map_info;
        const void* d_data = nullptr;
        int data_size = 0;
        const void* cpu_data = nullptr;
        int cpu_data_size = 0;

        map_gpu_memory(smeta->data_gpu, &map_info,
                       &d_data, &data_size,
                       &cpu_data, &cpu_data_size);

        /* Map mask data if present */
        GstMapInfo mask_map_info;
        const float* mask_cpu = NULL;
        int mask_bytes = 0;
        if (smeta->masks_gpu && smeta->mask_count > 0) {
            if (gst_memory_map(smeta->masks_gpu, &mask_map_info, GST_MAP_READ)) {
                mask_cpu = (const float*)mask_map_info.data;
                mask_bytes = (int)mask_map_info.size;
            }
        }

        MagmaToPrimitivesParams params{};
        params.type_id = tid_str;
        params.d_data = d_data;
        params.data_size = data_size;
        params.cpu_data = cpu_data;
        params.cpu_data_size = cpu_data_size;
        params.roi_x = smeta->roi_x;
        params.roi_y = smeta->roi_y;
        params.roi_w = smeta->roi_w;
        params.roi_h = smeta->roi_h;
        params.source_width = self->in_width;
        params.source_height = self->in_height;
        params.model_width = smeta->model_width;
        params.model_height = smeta->model_height;
        params.stream = (void*)self->hip_stream;
        params.labels_path = self->labels_file;
        params.mask_data = mask_cpu;
        params.mask_bytes = mask_bytes;
        params.mask_count = smeta->mask_count;
        params.mask_h = smeta->mask_height;
        params.mask_w = smeta->mask_width;
        params.d_masks_gpu = smeta->d_masks_gpu;
        params.d_objects_gpu = smeta->d_objects_gpu;
        params.d_masks_bytes = smeta->masks_gpu_bytes;

        func(&params, &self->primitives);

        if (mask_cpu)
            gst_memory_unmap(smeta->masks_gpu, &mask_map_info);
        if (cpu_data)
            gst_memory_unmap(smeta->data_gpu, &map_info);
    }

    /* ─── Path 2: Legacy MagmaInferenceMeta (backwards compat) ──── */
    if (self->primitives.total_primitives == 0) {
        MagmaInferenceMeta* m = magma_buffer_get_inference_meta(buf);
        if (m && m->num_objects > 0 && m->objects_gpu)
            legacy_detection_to_primitives(self, m, &self->primitives);
    }

    if (self->primitives.total_primitives == 0)
        return GST_FLOW_OK;

    /* ─── Get GPU frame pointer ──────────────────────────────────── */
    int y_stride, uv_stride;
    size_t y_off, uv_off;
    hipDeviceptr_t d_frame = get_frame_pointer(self, buf, frame_bytes,
                                                &y_stride, &uv_stride,
                                                &y_off, &uv_off);
    if (!d_frame) {
        GST_WARNING_OBJECT(self, "failed to get GPU pointer to frame");
        return GST_FLOW_OK;
    }

    unsigned char* d_y = (unsigned char*)d_frame + y_off;
    unsigned char* d_uv = (unsigned char*)d_frame + uv_off;

    /* ─── Dispatch per-primitive-type kernels ────────────────────── */
    dispatch_primitives(self, d_y, d_uv, y_stride, uv_stride);

    return GST_FLOW_OK;
}

/* ─── caps negotiation ────────────────────────────────────────────── */
static GstCaps* gst_magma_osd_transform_caps(GstBaseTransform* trans,
                                              GstPadDirection direction,
                                              GstCaps* caps, GstCaps* filter) {
    GstCaps* result = gst_caps_ref(caps);
    if (filter) {
        GstCaps* tmp = gst_caps_intersect_full(result, filter,
                                                 GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(result);
        result = tmp;
    }
    return result;
}

static gboolean gst_magma_osd_transform_size(GstBaseTransform* trans,
                                              GstPadDirection direction,
                                              GstCaps* caps, gsize size,
                                              GstCaps* othercaps,
                                              gsize* othersize) {
    *othersize = size;
    return TRUE;
}

static gboolean gst_magma_osd_set_caps(GstBaseTransform* trans,
                                        GstCaps* incaps,
                                        GstCaps* outcaps) {
    GstMagmaOsd* self = GST_MAGMA_OSD(trans);
    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, incaps)) {
        GST_ERROR_OBJECT(self, "failed to parse incaps");
        return FALSE;
    }
    self->in_width = GST_VIDEO_INFO_WIDTH(&info);
    self->in_height = GST_VIDEO_INFO_HEIGHT(&info);
    return TRUE;
}

/* ─── properties ──────────────────────────────────────────────────── */
static void gst_magma_osd_set_property(GObject* object, guint prop_id,
                                        const GValue* value,
                                        GParamSpec* pspec) {
    GstMagmaOsd* self = GST_MAGMA_OSD(object);
    switch (prop_id) {
    case PROP_SHOW_LABELS:      self->show_labels = g_value_get_boolean(value); break;
    case PROP_LINE_WIDTH:       self->line_width = g_value_get_uint(value); break;
    case PROP_ROI_X:            self->roi_x = g_value_get_uint(value); break;
    case PROP_ROI_Y:            self->roi_y = g_value_get_uint(value); break;
    case PROP_ROI_W:            self->roi_w = g_value_get_uint(value); break;
    case PROP_ROI_H:            self->roi_h = g_value_get_uint(value); break;
    case PROP_MAX_PRIMITIVES:   self->max_primitives = g_value_get_uint(value); break;
    case PROP_PALETTE_BY_TRACK: self->palette_by_track = g_value_get_boolean(value); break;
    case PROP_LABELS_FILE:
        g_free(self->labels_file);
        self->labels_file = g_value_dup_string(value);
        break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
    }
}

static void gst_magma_osd_get_property(GObject* object, guint prop_id,
                                        GValue* value, GParamSpec* pspec) {
    GstMagmaOsd* self = GST_MAGMA_OSD(object);
    switch (prop_id) {
    case PROP_SHOW_LABELS:      g_value_set_boolean(value, self->show_labels); break;
    case PROP_LINE_WIDTH:       g_value_set_uint(value, self->line_width); break;
    case PROP_ROI_X:            g_value_set_uint(value, self->roi_x); break;
    case PROP_ROI_Y:            g_value_set_uint(value, self->roi_y); break;
    case PROP_ROI_W:            g_value_set_uint(value, self->roi_w); break;
    case PROP_ROI_H:            g_value_set_uint(value, self->roi_h); break;
    case PROP_MAX_PRIMITIVES:   g_value_set_uint(value, self->max_primitives); break;
    case PROP_PALETTE_BY_TRACK: g_value_set_boolean(value, self->palette_by_track); break;
    case PROP_LABELS_FILE:      g_value_set_string(value, self->labels_file); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
    }
}

static void gst_magma_osd_finalize(GObject* object) {
    GstMagmaOsd* self = GST_MAGMA_OSD(object);

    auto safe_free = [](hipDeviceptr_t& p) {
        if (p) { (void)hipFree(p); p = nullptr; }
    };
    safe_free(self->d_rects);
    safe_free(self->d_polylines);
    safe_free(self->d_polygons);
    safe_free(self->d_points);
    safe_free(self->d_texts);
    safe_free(self->d_arrows);
    safe_free(self->d_vertex_arena);

    if (self->d_input_upload) { (void)hipFree(self->d_input_upload); self->d_input_upload = 0; }
    if (self->external_memory) { (void)hipDestroyExternalMemory(self->external_memory); self->external_memory = nullptr; }
    if (self->kernel_module) { (void)hipModuleUnload(self->kernel_module); self->kernel_module = nullptr; }
    magma_primitive_list_destroy(&self->primitives);
    g_free(self->labels_file);
    G_OBJECT_CLASS(gst_magma_osd_parent_class)->finalize(object);
}

static void gst_magma_osd_init(GstMagmaOsd* self) {
    self->in_width = self->in_height = 0;
    self->line_width = 2;
    self->show_labels = TRUE;
    self->max_primitives = 500;
    self->palette_by_track = FALSE;
    self->labels_file = NULL;
    self->roi_x = self->roi_y = self->roi_w = self->roi_h = 0;
    self->hip_stream = nullptr;
    self->kernel_module = nullptr;
    memset(self->kernel_funcs, 0, sizeof(self->kernel_funcs));
    self->kernel_ready = FALSE;
    self->external_memory = nullptr;
    self->d_image = 0;
    self->d_rects = self->d_polylines = self->d_polygons = nullptr;
    self->d_points = self->d_texts = self->d_arrows = nullptr;
    self->d_vertex_arena = nullptr;
    self->d_vertex_arena_bytes = 0;
    self->d_input_upload = 0;
    memset(&self->primitives, 0, sizeof(self->primitives));
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
    gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(self), TRUE);
}

/* ─── class init ──────────────────────────────────────────────────── */
static void gst_magma_osd_class_init(GstMagmaOsdClass* klass) {
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass* trans = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = gst_magma_osd_set_property;
    gobject_class->get_property = gst_magma_osd_get_property;
    gobject_class->finalize = gst_magma_osd_finalize;

    g_object_class_install_property(gobject_class, PROP_SHOW_LABELS,
        g_param_spec_boolean("show-labels","Show labels",
            "Show class label text on bounding boxes",
            TRUE, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_LINE_WIDTH,
        g_param_spec_uint("line-width","Line width",
            "Width of bounding box outlines in pixels",
            1, 10, 2, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_ROI_X,
        g_param_spec_uint("roi-x","ROI X",
            "ROI X offset in source pixels", 0, G_MAXUINT32, 0, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_ROI_Y,
        g_param_spec_uint("roi-y","ROI Y",
            "ROI Y offset in source pixels", 0, G_MAXUINT32, 0, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_ROI_W,
        g_param_spec_uint("roi-w","ROI Width",
            "ROI width (0=use source width)", 0, G_MAXUINT32, 0, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_ROI_H,
        g_param_spec_uint("roi-h","ROI Height",
            "ROI height (0=use source height)", 0, G_MAXUINT32, 0, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_MAX_PRIMITIVES,
        g_param_spec_uint("max-primitives","Max primitives",
            "Maximum number of render primitives per frame",
            16, 10000, 500, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_PALETTE_BY_TRACK,
        g_param_spec_boolean("palette-by-track","Palette by track",
            "Use track_id instead of class_id for colour selection",
            FALSE, G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_LABELS_FILE,
        g_param_spec_string("labels-file","Labels file",
            "Path to COCO-format labels file (one name per line, line=class_id)",
            NULL, G_PARAM_READWRITE));

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    gst_element_class_set_static_metadata(element_class,
        "Magma OSD","Filter/Effect/Video",
        "Model-agnostic overlay renderer for NV12 video (GPU)","Magma");

    trans->set_caps = gst_magma_osd_set_caps;
    trans->transform_caps = gst_magma_osd_transform_caps;
    trans->transform_size = gst_magma_osd_transform_size;
    trans->transform_ip = gst_magma_osd_transform_ip;
    trans->start = gst_magma_osd_start;
    trans->stop = gst_magma_osd_stop;

    magma_inference_meta_get_info();
    magma_semantic_meta_get_info();
    GST_DEBUG_CATEGORY_INIT(magma_osd_debug,"magma_osd",0,"Magma OSD Plugin");
}

static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin,"mgmosd",GST_RANK_NONE,
                                 GST_TYPE_MAGMA_OSD);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,GST_VERSION_MINOR,mgmosd,
    "Magma OSD Plugin (model-agnostic)",plugin_init,"0.1.0","LGPL","magma",
    "https://imeguras.eu.org")
