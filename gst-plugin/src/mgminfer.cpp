#include "mgminfer.hpp"
#include "magma-meta.h"
#include "kernel_utils.hpp"

#include <string>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <gbm.h>

/** --- GOBJECT / GSTREAMER STUFF --- */
GST_DEBUG_CATEGORY_STATIC(magma_infer_debug);
#define GST_CAT_DEFAULT magma_infer_debug

enum {
    PROP_0,
    PROP_MODEL_PATH,
    PROP_INFERENCE_INTERVAL,
};

G_DEFINE_TYPE(GstMagmaInfer, gst_magma_infer, GST_TYPE_BASE_TRANSFORM)

/** --- KERNEL PATH RESOLUTION --- */
static const char* find_kernel_dir(void) {
    const char* env = g_getenv("MAGMA_KERNEL_DIR");
    if (env)
        return env;
    return MAGMA_KERNEL_SRC_DIR;
}

/** --- DRM / GBM HELPERS --- */
static int open_drm_node(void) {
    for (int i = 0; i < 64; i++) {
        char path[64];
        g_snprintf(path, sizeof(path), "/dev/dri/renderD%d", 128 + i);
        int fd = open(path, O_RDWR);
        if (fd < 0)
            continue;
        drmVersionPtr ver = drmGetVersion(fd);
        if (ver) {
            drmFreeVersion(ver);
            return fd;
        }
        close(fd);
    }
    return -1;
}

static gboolean ensure_gbm_device(GstMagmaInfer* self) {
    if (self->gbm_ready)
        return TRUE;
    self->drm_fd = open_drm_node();
    if (self->drm_fd < 0) {
        GST_ERROR_OBJECT(self, "Failed to open DRM device");
        return FALSE;
    }
    self->gbm = gbm_create_device(self->drm_fd);
    if (!self->gbm) {
        GST_ERROR_OBJECT(self, "gbm_create_device failed");
        close(self->drm_fd);
        self->drm_fd = -1;
        return FALSE;
    }
    self->gbm_ready = TRUE;
    GST_INFO_OBJECT(self, "GBM device opened (fd=%d)", self->drm_fd);
    return TRUE;
}

/** --- OUTPUT OBJECTS DMABUF --- */
static gboolean ensure_objects_output(GstMagmaInfer* self) {
    if (self->d_objects)
        return TRUE;

    if (!ensure_gbm_device(self))
        return FALSE;

    gsize bytes = self->max_objects * sizeof(MagmaInferObjectGPU);

    struct gbm_bo* bo = gbm_bo_create(self->gbm, bytes, 1, GBM_FORMAT_R8, GBM_BO_USE_RENDERING);
    if (!bo) {
        GST_ERROR_OBJECT(self, "gbm_bo_create(%zu bytes) failed", bytes);
        return FALSE;
    }

    int dmabuf_fd = gbm_bo_get_fd(bo);
    gbm_bo_destroy(bo);

    if (dmabuf_fd < 0) {
        GST_ERROR_OBJECT(self, "gbm_bo_get_fd failed");
        return FALSE;
    }

    int hip_fd = fcntl(dmabuf_fd, F_DUPFD_CLOEXEC, 0);

    GstAllocator* dma_alloc = gst_dmabuf_allocator_new();
    self->objects_mem = gst_dmabuf_allocator_alloc(dma_alloc, dmabuf_fd, bytes);
    gst_object_unref(dma_alloc);
    if (!self->objects_mem) {
        GST_ERROR_OBJECT(self, "gst_dmabuf_allocator_alloc failed");
        close(hip_fd);
        return FALSE;
    }
    self->objects_dmabuf_fd = dmabuf_fd;

    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = hip_fd;
    desc.size = bytes;

    hipError_t err = hipImportExternalMemory(&self->objects_ext_mem, &desc);
    close(hip_fd);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipImportExternalMemory(objects) failed: %s", hipGetErrorString(err));
        gst_memory_unref(self->objects_mem);
        self->objects_mem = NULL;
        return FALSE;
    }

    hipExternalMemoryBufferDesc bdesc{};
    bdesc.offset = 0;
    bdesc.size = bytes;

    err = hipExternalMemoryGetMappedBuffer((hipDeviceptr_t*)&self->d_objects, self->objects_ext_mem, &bdesc);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipExternalMemoryGetMappedBuffer(objects) failed: %s", hipGetErrorString(err));
        (void)hipDestroyExternalMemory(self->objects_ext_mem);
        self->objects_ext_mem = nullptr;
        gst_memory_unref(self->objects_mem);
        self->objects_mem = NULL;
        return FALSE;
    }

    GST_INFO_OBJECT(self, "Objects DMABuf allocated (%zu bytes, max %u objects)", bytes, self->max_objects);
    return TRUE;
}

/** --- PAD TEMPLATES --- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

/** --- PROPERTIES --- */
static void gst_magma_infer_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
    GstMagmaInfer* self = GST_MAGMA_INFER(object);

    switch (prop_id) {
    case PROP_MODEL_PATH:
        g_free(self->model_path);
        self->model_path = g_value_dup_string(value);
        GST_INFO_OBJECT(self, "model path set to %s", self->model_path);
        break;
    case PROP_INFERENCE_INTERVAL:
        self->inference_interval = g_value_get_uint(value);
        GST_INFO_OBJECT(self, "inference interval set to %u", self->inference_interval);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_magma_infer_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
    GstMagmaInfer* self = GST_MAGMA_INFER(object);

    switch (prop_id) {
    case PROP_MODEL_PATH:
        g_value_set_string(value, self->model_path);
        break;
    case PROP_INFERENCE_INTERVAL:
        g_value_set_uint(value, self->inference_interval);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/** --- FINALIZE --- */
static void gst_magma_infer_finalize(GObject* object) {
    GstMagmaInfer* self = GST_MAGMA_INFER(object);

    if (self->kernel_module) {
        (void)hipModuleUnload(self->kernel_module);
        self->kernel_module = nullptr;
    }

    if (self->objects_ext_mem) {
        (void)hipDestroyExternalMemory(self->objects_ext_mem);
        self->objects_ext_mem = nullptr;
    }
    if (self->objects_mem) {
        gst_memory_unref(self->objects_mem);
        self->objects_mem = NULL;
    }
    self->d_objects = nullptr;

    if (self->hip_stream) {
        (void)hipStreamDestroy(self->hip_stream);
        self->hip_stream = nullptr;
    }

    if (self->gbm) {
        gbm_device_destroy(self->gbm);
        self->gbm = nullptr;
    }
    if (self->drm_fd >= 0) {
        close(self->drm_fd);
        self->drm_fd = -1;
    }

    g_free(self->model_path);
    self->model_path = NULL;

    G_OBJECT_CLASS(gst_magma_infer_parent_class)->finalize(object);
}

/** --- INIT --- */
static void gst_magma_infer_init(GstMagmaInfer* self) {
    self->model_path = NULL;
    self->inference_interval = 1;
    self->frame_counter = 0;
    self->in_width = 0;
    self->in_height = 0;

    self->drm_fd = -1;
    self->gbm = nullptr;
    self->gbm_ready = FALSE;
    self->objects_dmabuf_fd = -1;
    self->objects_ext_mem = nullptr;
    self->d_objects = nullptr;
    self->objects_mem = NULL;
    self->max_objects = 100;

    self->hip_stream = nullptr;
    self->kernel_module = nullptr;
    self->kernel_dummy = nullptr;
    self->kernel_ready = FALSE;

    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

/** --- CAPS NEGOTIATION --- */
static gboolean gst_magma_infer_set_caps(GstBaseTransform* trans, GstCaps* incaps, GstCaps* outcaps) {
    GstMagmaInfer* self = GST_MAGMA_INFER(trans);
    GstVideoInfo info;

    if (!gst_video_info_from_caps(&info, incaps)) {
        GST_ERROR_OBJECT(self, "failed to parse incaps");
        return FALSE;
    }

    self->in_width = GST_VIDEO_INFO_WIDTH(&info);
    self->in_height = GST_VIDEO_INFO_HEIGHT(&info);

    if (!ensure_objects_output(self)) {
        GST_ERROR_OBJECT(self, "failed to allocate objects output DMABuf");
        return FALSE;
    }

    GST_INFO_OBJECT(self, "configured %dx%d %s", self->in_width, self->in_height, GST_VIDEO_INFO_NAME(&info));
    return TRUE;
}

/** --- TRANSFORM --- */
static GstFlowReturn gst_magma_infer_transform_ip(GstBaseTransform* trans, GstBuffer* buf) {
    GstMagmaInfer* self = GST_MAGMA_INFER(trans);

    self->frame_counter++;

    if ((self->frame_counter - 1) % self->inference_interval != 0)
        return GST_FLOW_OK;

    MagmaTensorMeta* tmeta = magma_buffer_get_tensor_meta(buf);
    if (!tmeta) {
        GST_WARNING_OBJECT(self, "no MagmaTensorMeta on buffer — skipping inference");
        return GST_FLOW_OK;
    }

    if (!self->hip_stream) {
        hipError_t err = hipStreamCreate(&self->hip_stream);
        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipStreamCreate failed: %s", hipGetErrorString(err));
            return GST_FLOW_ERROR;
        }
    }

    /* compile dummy kernel once */
    if (!self->kernel_ready) {
        std::string kernel_path = std::string(find_kernel_dir()) + "/dummy_infer.hip";

        HipKernel kern = compile_kernel(kernel_path.c_str(), "dummy_infer");
        if (!kern.func) {
            GST_ERROR_OBJECT(self, "Failed to compile dummy_infer kernel");
            return GST_FLOW_ERROR;
        }
        self->kernel_module = kern.module;
        self->kernel_dummy = kern.func;
        self->kernel_ready = TRUE;
        GST_INFO_OBJECT(self, "dummy_infer kernel compiled from %s", kernel_path.c_str());
    }

    /* launch dummy kernel (writes one detection at d_objects) */
    void* d_ptr = (void*)self->d_objects;
    void* args[] = {&d_ptr};

    hipError_t err = hipModuleLaunchKernel(self->kernel_dummy, 1, 1, 1, 1, 1, 1, 0, self->hip_stream, args, nullptr);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipModuleLaunchKernel(dummy_infer) failed: %s", hipGetErrorString(err));
        return GST_FLOW_ERROR;
    }

    err = hipStreamSynchronize(self->hip_stream);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipStreamSynchronize failed: %s", hipGetErrorString(err));
        return GST_FLOW_ERROR;
    }

    /* attach inference meta */
    MagmaInferenceMeta* m = magma_buffer_add_inference_meta(buf, self->in_width, self->in_height);
    if (!m) {
        GST_ERROR_OBJECT(self, "failed to attach inference meta");
        return GST_FLOW_ERROR;
    }
    m->num_objects = 1;
    m->objects_gpu = gst_memory_ref(self->objects_mem);

    GST_LOG_OBJECT(self, "frame %u — dummy detection attached", self->frame_counter);
    return GST_FLOW_OK;
}

/** --- CLASS INIT --- */
static void gst_magma_infer_class_init(GstMagmaInferClass* klass) {
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass* trans = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = gst_magma_infer_set_property;
    gobject_class->get_property = gst_magma_infer_get_property;
    gobject_class->finalize = gst_magma_infer_finalize;

    g_object_class_install_property(gobject_class, PROP_MODEL_PATH, g_param_spec_string("model-path", "Model path", "Path to the inference model file", NULL, G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_INFERENCE_INTERVAL, g_param_spec_uint("inference-interval", "Inference interval", "Run inference every N frames (1 = every frame)", 1, G_MAXUINT32, 1, G_PARAM_READWRITE));

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    gst_element_class_set_static_metadata(element_class, "Magma Inference", "Meta/Inference/Video", "Attaches GPU inference results as buffer metadata (dummy kernel, Phase 1)", "Magma");

    trans->set_caps = gst_magma_infer_set_caps;
    trans->transform_ip = gst_magma_infer_transform_ip;

    magma_inference_meta_get_info();
    magma_tensor_meta_get_info();

    GST_DEBUG_CATEGORY_INIT(magma_infer_debug, "magma_infer", 0, "Magma Inference Plugin");
}

/** --- PLUGIN REGISTRATION --- */
static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "mgminfer", GST_RANK_NONE, GST_TYPE_MAGMA_INFER);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mgminfer, "Magma Inference Plugin", plugin_init, "0.1.0", "LGPL", "magma", "https://imeguras.eu.org")
