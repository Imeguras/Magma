#include "mgmosd.hpp"
#include "magma-hip-stream.hpp"
#include "magma-meta.h"

#include <cstring>
#include <cstdio>
#include <string>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

GST_DEBUG_CATEGORY_STATIC(magma_osd_debug);
#define GST_CAT_DEFAULT magma_osd_debug

/* ---------- properties ---------- */
enum {
    PROP_0,
    PROP_LINE_WIDTH,
    PROP_ROI_X, PROP_ROI_Y, PROP_ROI_W, PROP_ROI_H,
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

/* ---------- color palette (RGB -> precomputed YUV) ---------- */
struct YuvColor { guint8 y, u, v; };

static const YuvColor palette_yuv[] = {
    { 76, 84, 255},  /* red    */
    {149, 54, 34 },  /* green  */
    { 29, 255,107},  /* blue   */
    {225, 43, 21 },  /* yellow */
    {106, 84, 212},  /* magenta */
    {178, 42, 148},  /* cyan   */
    {162, 67, 50 },  /* orange */
    { 61, 168,208},  /* purple */
    { 70, 188,155},  /* sky blue */
    {111, 123,233},  /* rose   */
    {185, 18, 67 },  /* lime   */
    {147, 68, 82 },  /* spring */
    {177, 97, 127},  /* pink   */
    {106, 150,184},  /* lavender */
    {179, 75, 81 },  /* mint   */
    {170, 109,197},  /* light magenta */
};
#define NUM_PALETTE (sizeof(palette_yuv)/sizeof(palette_yuv[0]))

/* ---------- GPU kernel parameter struct (must match osd_kernels.hip) ---------- */
struct BoxParam {
    int x, y, w, h;
    unsigned char y_val, u_val, v_val;
    unsigned char _pad[1];
};

/* ---------- kernel directory lookup ---------- */
static std::string find_kernel_dir(void) {
    const char* env = getenv("MAGMA_KERNEL_DIR");
    if (env) return env;
    return MAGMA_KERNEL_SRC_DIR;
}

/* ---------- import DMABuf -> HIP ---------- */
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

/* ---------- start (READY -> PAUSED) ---------- */
static gboolean gst_magma_osd_start(GstBaseTransform* trans) {
    GstMagmaOsd* self = GST_MAGMA_OSD(trans);

    self->hip_stream = magma_get_shared_hip_stream();
    if (!self->hip_stream) {
        GST_ERROR_OBJECT(self, "magma_get_shared_hip_stream failed");
        return FALSE;
    }

    std::string dir = find_kernel_dir();
    std::string kpath = dir + "/osd_kernels.hip";
    std::string cpath = dir + "/common.hip";
    HipKernel k = compile_kernel(kpath.c_str(), "draw_boxes_kernel", cpath.c_str());
    if (!k.func) {
        GST_ERROR_OBJECT(self, "failed to compile draw_boxes_kernel from %s", kpath.c_str());
        return FALSE;
    }
    self->kernel_module = k.module;
    self->kernel_func = k.func;
    self->kernel_ready = TRUE;
    GST_INFO_OBJECT(self, "OSD kernel compiled from %s", kpath.c_str());

    hipError_t e = hipMalloc(&self->d_boxes, 100 * sizeof(BoxParam));
    if (e != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipMalloc(boxes) failed: %s", hipGetErrorString(e));
        (void)hipModuleUnload(self->kernel_module);
        self->kernel_module = nullptr;
        self->kernel_func = nullptr;
        self->kernel_ready = FALSE;
        return FALSE;
    }

    return TRUE;
}

/* ---------- stop (PAUSED -> READY) ---------- */
static gboolean gst_magma_osd_stop(GstBaseTransform* trans) {
    GstMagmaOsd* self = GST_MAGMA_OSD(trans);

    if (self->d_boxes) {
        (void)hipFree(self->d_boxes);
        self->d_boxes = nullptr;
    }

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
        self->kernel_func = nullptr;
        self->kernel_ready = FALSE;
    }

    return TRUE;
}

/* ---------- transform_ip ---------- */
/**
 * @brief Main transform entry point — draw detection overlays.
 *
 * Reads MagmaInferenceMeta from the input buffer, uploads
 * detection objects to the GPU, and runs the OSD HIP kernel
 * to draw bounding boxes and labels directly onto the NV12 frame.
 *
 * @param trans The base transform element
 * @param buf   Input/output NV12 buffer with MagmaInferenceMeta
 * @return GST_FLOW_OK on success
 */
static GstFlowReturn gst_magma_osd_transform_ip(GstBaseTransform* trans,
                                                 GstBuffer* buf) {
    GstMagmaOsd* self = GST_MAGMA_OSD(trans);

    if (!self->kernel_ready)
        return GST_FLOW_OK;

    MagmaInferenceMeta* m = magma_buffer_get_inference_meta(buf);
    if (!m || m->num_objects == 0 || !m->objects_gpu)
        return GST_FLOW_OK;

    GstMapInfo obj_info;
    if (!gst_memory_map(m->objects_gpu, &obj_info, GST_MAP_READ))
        return GST_FLOW_OK;

    MagmaInferObjectGPU* objects = (MagmaInferObjectGPU*)obj_info.data;
    int num = (int)m->num_objects;

    /* Coordinate mapping: prefer metadata from preprocessing chain */
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

    BoxParam params[100];
    int nparams = 0;
    for (int i = 0; i < num && nparams < 100; i++) {
        MagmaInferObjectGPU* obj = &objects[i];
        int sx = off_x + (int)(obj->x * map_w);
        int sy = off_y + (int)(obj->y * map_h);
        int sw = (int)(obj->width  * map_w);
        int sh = (int)(obj->height * map_h);
        if (sw < 2 || sh < 2) continue;

        guint cid = obj->class_id % NUM_PALETTE;
        params[nparams].x = sx;
        params[nparams].y = sy;
        params[nparams].w = sw;
        params[nparams].h = sh;
        params[nparams].y_val = palette_yuv[cid].y;
        params[nparams].u_val = palette_yuv[cid].u;
        params[nparams].v_val = palette_yuv[cid].v;
        params[nparams]._pad[0] = 0;
        nparams++;
    }

    gst_memory_unmap(m->objects_gpu, &obj_info);

    if (nparams == 0) return GST_FLOW_OK;

    gsize frame_bytes = (gsize)self->in_width * self->in_height * 3 / 2;
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

    // --- get GPU pointer to the frame ---
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
            // System memory fallback: upload to GPU
            GstMapInfo in_map;
            if (gst_buffer_map(buf, &in_map, GST_MAP_READ)) {
                if (!self->d_input_upload)
                    (void)hipMalloc(&self->d_input_upload, frame_bytes);
                if (self->d_input_upload) {
                    hipMemcpy(self->d_input_upload, in_map.data, frame_bytes, hipMemcpyHostToDevice);
                    d_frame = self->d_input_upload;
                }
                gst_buffer_unmap(buf, &in_map);
            }
        }
    }

    if (!d_frame) {
        GST_WARNING_OBJECT(self, "failed to get GPU pointer to frame");
        return GST_FLOW_OK;
    }

    uint8_t* d_y = (uint8_t*)d_frame + y_off;
    uint8_t* d_uv = (uint8_t*)d_frame + uv_off;

    size_t box_bytes = (size_t)nparams * sizeof(BoxParam);
    hipError_t e = hipMemcpyHtoDAsync(self->d_boxes, params, box_bytes, self->hip_stream);
    if (e != hipSuccess) { GST_WARNING_OBJECT(self, "hipMemcpyHtoD(boxes) failed"); return GST_FLOW_OK; }

    int block = 64;
    int grid = (nparams + block - 1) / block;
    void* args[] = { &d_y, &d_uv, &y_stride, &uv_stride,
                     &self->in_width, &self->in_height,
                     &self->d_boxes, &nparams, &self->line_width };

    e = hipModuleLaunchKernel(self->kernel_func, grid, 1, 1, block, 1, 1,
                               0, self->hip_stream, args, nullptr);
    if (e != hipSuccess) {
        GST_WARNING_OBJECT(self, "draw_boxes_kernel launch failed: %s", hipGetErrorString(e));
    }

    return GST_FLOW_OK;
}

/* ---------- caps ---------- */
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

/* ---------- properties ---------- */
static void gst_magma_osd_set_property(GObject* object, guint prop_id,
                                        const GValue* value,
                                        GParamSpec* pspec) {
    GstMagmaOsd* self = GST_MAGMA_OSD(object);
    switch (prop_id) {
    case PROP_LINE_WIDTH: self->line_width = g_value_get_uint(value); break;
    case PROP_ROI_X:      self->roi_x = g_value_get_uint(value); break;
    case PROP_ROI_Y:      self->roi_y = g_value_get_uint(value); break;
    case PROP_ROI_W:      self->roi_w = g_value_get_uint(value); break;
    case PROP_ROI_H:      self->roi_h = g_value_get_uint(value); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
    }
}

static void gst_magma_osd_get_property(GObject* object, guint prop_id,
                                        GValue* value, GParamSpec* pspec) {
    GstMagmaOsd* self = GST_MAGMA_OSD(object);
    switch (prop_id) {
    case PROP_LINE_WIDTH: g_value_set_uint(value, self->line_width); break;
    case PROP_ROI_X:      g_value_set_uint(value, self->roi_x); break;
    case PROP_ROI_Y:      g_value_set_uint(value, self->roi_y); break;
    case PROP_ROI_W:      g_value_set_uint(value, self->roi_w); break;
    case PROP_ROI_H:      g_value_set_uint(value, self->roi_h); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
    }
}

static void gst_magma_osd_finalize(GObject* object) {
    GstMagmaOsd* self = GST_MAGMA_OSD(object);
    if (self->d_boxes) { (void)hipFree(self->d_boxes); self->d_boxes = nullptr; }
    if (self->d_input_upload) { (void)hipFree(self->d_input_upload); self->d_input_upload = 0; }
    if (self->external_memory) { (void)hipDestroyExternalMemory(self->external_memory); self->external_memory = nullptr; }
    if (self->kernel_module) { (void)hipModuleUnload(self->kernel_module); self->kernel_module = nullptr; }
    G_OBJECT_CLASS(gst_magma_osd_parent_class)->finalize(object);
}

static void gst_magma_osd_init(GstMagmaOsd* self) {
    self->in_width = self->in_height = 0;
    self->line_width = 2;
    self->roi_x = self->roi_y = self->roi_w = self->roi_h = 0;
    self->hip_stream = nullptr;
    self->kernel_module = nullptr;
    self->kernel_func = nullptr;
    self->kernel_ready = FALSE;
    self->external_memory = nullptr;
    self->d_image = 0;
    self->d_boxes = nullptr;
    self->d_input_upload = 0;
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
    gst_base_transform_set_qos_enabled(GST_BASE_TRANSFORM(self), TRUE);
}

/* ---------- class init ---------- */
static void gst_magma_osd_class_init(GstMagmaOsdClass* klass) {
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass* trans = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = gst_magma_osd_set_property;
    gobject_class->get_property = gst_magma_osd_get_property;
    gobject_class->finalize = gst_magma_osd_finalize;

    g_object_class_install_property(gobject_class, PROP_LINE_WIDTH,
        g_param_spec_uint("line-width","Line width",
            "Width of bounding box outlines in pixels",
            1,10,2,G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_ROI_X,
        g_param_spec_uint("roi-x","ROI X",
            "ROI X offset in source pixels",0,G_MAXUINT32,0,G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_ROI_Y,
        g_param_spec_uint("roi-y","ROI Y",
            "ROI Y offset in source pixels",0,G_MAXUINT32,0,G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_ROI_W,
        g_param_spec_uint("roi-w","ROI Width",
            "ROI width (0=use source width)",0,G_MAXUINT32,0,G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_ROI_H,
        g_param_spec_uint("roi-h","ROI Height",
            "ROI height (0=use source height)",0,G_MAXUINT32,0,G_PARAM_READWRITE));

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    gst_element_class_set_static_metadata(element_class,
        "Magma OSD","Filter/Effect/Video",
        "Draws detection bounding boxes on NV12 video (GPU)","Magma");

    trans->set_caps = gst_magma_osd_set_caps;
    trans->transform_caps = gst_magma_osd_transform_caps;
    trans->transform_size = gst_magma_osd_transform_size;
    trans->transform_ip = gst_magma_osd_transform_ip;
    trans->start = gst_magma_osd_start;
    trans->stop = gst_magma_osd_stop;

    magma_inference_meta_get_info();
    GST_DEBUG_CATEGORY_INIT(magma_osd_debug,"magma_osd",0,"Magma OSD Plugin");
}

static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin,"mgmosd",GST_RANK_NONE,
                                GST_TYPE_MAGMA_OSD);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,GST_VERSION_MINOR,mgmosd,
    "Magma OSD Plugin",plugin_init,"0.1.0","LGPL","magma",
    "https://imeguras.eu.org")
