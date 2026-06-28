#include "mgmpreproc.hpp"
#include "kernel_utils.hpp"
#include "magma-meta.h"

#include <string>
#include <vector>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <gbm.h>

/** --- GOBJECT / GSTREAMER STUFF --- */
GST_DEBUG_CATEGORY_STATIC(magma_preproc_debug);
#define GST_CAT_DEFAULT magma_preproc_debug

enum {
    PROP_0,
    PROP_NET_WIDTH,
    PROP_NET_HEIGHT,
    PROP_SCALE_FACTOR,
};

G_DEFINE_TYPE(GstMagmaPreproc, gst_magma_preproc, GST_TYPE_BASE_TRANSFORM)

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
        if (fd < 0) continue;
        drmVersionPtr ver = drmGetVersion(fd);
        if (ver) { drmFreeVersion(ver); return fd; }
        close(fd);
    }
    return -1;
}

static gboolean ensure_gbm_device(GstMagmaPreproc* self) {
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

/** --- TENSOR OUTPUT (DMABuf-backed) --- */
static gboolean ensure_tensor(GstMagmaPreproc* self) {
    if (self->d_tensor_output)
        return TRUE;

    if (!ensure_gbm_device(self))
        return FALSE;

    gsize n = (gsize)self->net_width * self->net_height * 3;
    gsize bytes = n * sizeof(float);

    struct gbm_bo *bo = gbm_bo_create(self->gbm, bytes, 1, GBM_FORMAT_R8,
                                       GBM_BO_USE_RENDERING);
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

    // Dup for HIP import and GstMemory
    int hip_fd = fcntl(dmabuf_fd, F_DUPFD_CLOEXEC, 0);

    // Wrap in GstMemory (takes ownership of dmabuf_fd)
    GstAllocator *dma_alloc = gst_dmabuf_allocator_new();
    self->tensor_mem = gst_dmabuf_allocator_alloc(dma_alloc, dmabuf_fd, bytes);
    gst_object_unref(dma_alloc);
    if (!self->tensor_mem) {
        GST_ERROR_OBJECT(self, "gst_dmabuf_allocator_alloc failed");
        close(hip_fd);
        return FALSE;
    }
    self->tensor_dmabuf_fd = dmabuf_fd;

    // Import into HIP
    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = hip_fd;
    desc.size = bytes;

    hipError_t err = hipImportExternalMemory(&self->tensor_ext_mem, &desc);
    close(hip_fd);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipImportExternalMemory(tensor) failed: %s", hipGetErrorString(err));
        gst_memory_unref(self->tensor_mem);
        self->tensor_mem = NULL;
        return FALSE;
    }

    hipExternalMemoryBufferDesc bdesc{};
    bdesc.offset = 0;
    bdesc.size = bytes;

    err = hipExternalMemoryGetMappedBuffer((hipDeviceptr_t*)&self->d_tensor_output,
                                            self->tensor_ext_mem, &bdesc);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipExternalMemoryGetMappedBuffer(tensor) failed: %s", hipGetErrorString(err));
        (void)hipDestroyExternalMemory(self->tensor_ext_mem);
        self->tensor_ext_mem = nullptr;
        gst_memory_unref(self->tensor_mem);
        self->tensor_mem = NULL;
        return FALSE;
    }

    self->tensor_alloc_size = bytes;
    GST_INFO_OBJECT(self, "Tensor DMABuf %dx%dx3 (%zu bytes)", self->net_width, self->net_height, bytes);
    return TRUE;
}

/** --- PROPERTIES --- */
static void gst_magma_preproc_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
    GstMagmaPreproc* self = GST_MAGMA_PREPROC(object);
    switch (prop_id) {
    case PROP_NET_WIDTH:
        self->net_width = g_value_get_int(value);
        break;
    case PROP_NET_HEIGHT:
        self->net_height = g_value_get_int(value);
        break;
    case PROP_SCALE_FACTOR:
        self->scale_factor = g_value_get_float(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_magma_preproc_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
    GstMagmaPreproc* self = GST_MAGMA_PREPROC(object);
    switch (prop_id) {
    case PROP_NET_WIDTH:
        g_value_set_int(value, self->net_width);
        break;
    case PROP_NET_HEIGHT:
        g_value_set_int(value, self->net_height);
        break;
    case PROP_SCALE_FACTOR:
        g_value_set_float(value, self->scale_factor);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/** --- FINALIZE --- */
static void gst_magma_preproc_finalize(GObject* object) {
    GstMagmaPreproc* self = GST_MAGMA_PREPROC(object);

    if (self->kernel_module) {
        (void)hipModuleUnload(self->kernel_module);
        self->kernel_module = nullptr;
    }

    if (self->tensor_ext_mem) {
        (void)hipDestroyExternalMemory(self->tensor_ext_mem);
        self->tensor_ext_mem = nullptr;
    }
    if (self->tensor_mem) {
        gst_memory_unref(self->tensor_mem);
        self->tensor_mem = NULL;
    }
    self->d_tensor_output = nullptr;

    if (self->external_memory) {
        (void)hipDestroyExternalMemory(self->external_memory);
        self->external_memory = nullptr;
    }
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

    G_OBJECT_CLASS(gst_magma_preproc_parent_class)->finalize(object);
}

/** --- PAD TEMPLATES --- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

/** --- INIT --- */
static void gst_magma_preproc_init(GstMagmaPreproc* self) {
    self->net_width = 224;
    self->net_height = 224;
    self->scale_factor = 1.0f / 255.0f;

    self->hip_stream = nullptr;
    self->external_memory = nullptr;
    self->d_image = 0;
    self->imported = FALSE;

    self->kernel_module = nullptr;
    self->kernel_nv12_to_rgb = nullptr;
    self->kernel_ready = FALSE;

    self->drm_fd = -1;
    self->gbm = nullptr;
    self->gbm_ready = FALSE;
    self->tensor_dmabuf_fd = -1;
    self->tensor_ext_mem = nullptr;
    self->d_tensor_output = nullptr;
    self->tensor_mem = NULL;
    self->tensor_alloc_size = 0;

    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

/** --- CAPS NEGOTIATION --- */
static GstCaps* gst_magma_preproc_transform_caps(GstBaseTransform* trans, GstPadDirection direction, GstCaps* caps, GstCaps* filter) {
    GstCaps* result = gst_caps_ref(caps);
    if (filter) {
        GstCaps* tmp = gst_caps_intersect_full(result, filter, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(result);
        result = tmp;
    }
    return result;
}

static gboolean gst_magma_preproc_set_caps(GstBaseTransform* trans, GstCaps* incaps, GstCaps* outcaps) {
    GstMagmaPreproc* self = GST_MAGMA_PREPROC(trans);
    GstStructure* s = gst_caps_get_structure(incaps, 0);

    gst_structure_get_int(s, "width", &self->in_width);
    gst_structure_get_int(s, "height", &self->in_height);

    const gchar* fmt = gst_structure_get_string(s, "format");
    self->in_format = gst_video_format_from_string(fmt);

    if (self->in_format != GST_VIDEO_FORMAT_NV12) {
        GST_ERROR_OBJECT(self, "Only NV12 supported currently");
        return FALSE;
    }

    if (!ensure_tensor(self)) {
        GST_ERROR_OBJECT(self, "Failed to allocate tensor buffer");
        return FALSE;
    }

    GST_INFO_OBJECT(self, "Input %dx%d %s", self->in_width, self->in_height, fmt);
    return TRUE;
}

static gboolean gst_magma_preproc_transform_ip_size(GstBaseTransform* trans, GstPadDirection direction, GstCaps* caps, gsize size, GstCaps* othercaps, gsize* othersize) {
    *othersize = size;
    return TRUE;
}

/** --- TRANSFORM (per-frame): NV12 → tensor --- */
static GstFlowReturn gst_magma_preproc_transform_ip(GstBaseTransform* trans, GstBuffer* buf) {
    GstMagmaPreproc* self = GST_MAGMA_PREPROC(trans);

    // Ensure the tensor meta type is registered exactly once
    magma_tensor_meta_get_info();

    GstMemory* mem = gst_buffer_peek_memory(buf, 0);
    if (!gst_is_dmabuf_memory(mem)) {
        GST_ERROR_OBJECT(self, "mgmpreproc requires DMABuf memory");
        return GST_FLOW_ERROR;
    }

    gint fd = gst_dmabuf_memory_get_fd(mem);

    // Create HIP stream once
    if (!self->hip_stream) {
        hipError_t err = hipStreamCreate(&self->hip_stream);
        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipStreamCreate failed: %s", hipGetErrorString(err));
            return GST_FLOW_ERROR;
        }
    }

    // Import DMABuf into HIP once
    if (!self->imported) {
        hipExternalMemoryHandleDesc desc{};
        desc.type = hipExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = fd;
        desc.size = (gsize)self->in_width * self->in_height * 3 / 2;

        hipError_t err = hipImportExternalMemory(&self->external_memory, &desc);
        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipImportExternalMemory failed: %s", hipGetErrorString(err));
            return GST_FLOW_ERROR;
        }

        hipExternalMemoryBufferDesc bdesc{};
        bdesc.offset = 0;
        bdesc.size = desc.size;

        err = hipExternalMemoryGetMappedBuffer(&self->d_image, self->external_memory, &bdesc);
        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipExternalMemoryGetMappedBuffer failed: %s", hipGetErrorString(err));
            return GST_FLOW_ERROR;
        }

        self->imported = TRUE;
        GST_INFO_OBJECT(self, "HIP DMABUF import successful (ptr=%p, fd=%d)", (void*)self->d_image, fd);
    }

    // Compile kernel once on first frame
    if (!self->kernel_ready) {
        std::string kernel_path = std::string(find_kernel_dir()) + "/preproc_kernels.hip";

        HipKernel kern = compile_kernel(kernel_path.c_str(), "nv12_to_rgb_normalized");
        if (!kern.func) {
            GST_ERROR_OBJECT(self, "Failed to compile nv12_to_rgb_normalized kernel");
            return GST_FLOW_ERROR;
        }
        self->kernel_module = kern.module;
        self->kernel_nv12_to_rgb = kern.func;

        self->kernel_ready = TRUE;
        GST_INFO_OBJECT(self, "Kernel compiled and loaded from %s", kernel_path.c_str());
    }

    // Input frame params
    void* d_ptr = (void*)self->d_image;
    int w = self->in_width;
    int h = self->in_height;
    int stride = self->in_width;
    GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
    if (vmeta && vmeta->stride[0] > 0)
        stride = vmeta->stride[0];

    // --- Launch kernel: NV12 → RGB normalized tensor ---
    float* t_ptr = self->d_tensor_output;
    int nw = self->net_width;
    int nh = self->net_height;
    float sf = self->scale_factor;
    void* args[] = {&d_ptr, &w, &h, &stride, &t_ptr, &nw, &nh, &sf};

    int block_size = 16;
    int grid_x = (nw + block_size - 1) / block_size;
    int grid_y = (nh + block_size - 1) / block_size;

    hipError_t err = hipModuleLaunchKernel(self->kernel_nv12_to_rgb, grid_x, grid_y, 1,
                                           block_size, block_size, 1, 0,
                                           self->hip_stream, args, nullptr);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipModuleLaunchKernel(nv12_to_rgb) failed: %s", hipGetErrorString(err));
        return GST_FLOW_ERROR;
    }

    err = hipStreamSynchronize(self->hip_stream);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipStreamSynchronize failed: %s", hipGetErrorString(err));
        return GST_FLOW_ERROR;
    }

    // Attach tensor DMABuf as metadata on the buffer (zero-copy: refs the DMABuf)
    magma_buffer_add_tensor_meta(buf, self->tensor_mem, nw, nh, 3);

    return GST_FLOW_OK;
}

/** --- PLUGIN REGISTRATION --- */
static gboolean plugin_init(GstPlugin* plugin) {
    GST_DEBUG_CATEGORY_INIT(magma_preproc_debug, "magma_preproc", 0, "Magma GPU Preprocessor");
    return gst_element_register(plugin, "mgmpreproc", GST_RANK_NONE, GST_TYPE_MAGMA_PREPROC);
}

static void gst_magma_preproc_class_init(GstMagmaPreprocClass* klass) {
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GObjectClass* object_class = G_OBJECT_CLASS(klass);
    object_class->set_property = gst_magma_preproc_set_property;
    object_class->get_property = gst_magma_preproc_get_property;
    object_class->finalize = gst_magma_preproc_finalize;

    g_object_class_install_property(
        object_class, PROP_NET_WIDTH, g_param_spec_int("net-width", "Network Input Width", "Width of the input tensor expected by the neural network", 1, G_MAXINT, 224, G_PARAM_READWRITE));
    g_object_class_install_property(
        object_class, PROP_NET_HEIGHT, g_param_spec_int("net-height", "Network Input Height", "Height of the input tensor expected by the neural network", 1, G_MAXINT, 224, G_PARAM_READWRITE));
    g_object_class_install_property(
        object_class, PROP_SCALE_FACTOR, g_param_spec_float("scale-factor", "Scale Factor", "Factor by which to scale the input tensor", 0.0, G_MAXFLOAT, 1.0f, G_PARAM_READWRITE));

    gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass), &sink_template);
    gst_element_class_add_static_pad_template(GST_ELEMENT_CLASS(klass), &src_template);

    GstBaseTransformClass* trans_class = GST_BASE_TRANSFORM_CLASS(klass);
    trans_class->transform_size = gst_magma_preproc_transform_ip_size;
    trans_class->transform_caps = gst_magma_preproc_transform_caps;
    trans_class->set_caps = gst_magma_preproc_set_caps;
    trans_class->transform_ip = gst_magma_preproc_transform_ip;

    gst_element_class_set_static_metadata(element_class, "Magma Preprocessor", "Filter/Video", "ROCm/HIP tensor preprocessing (NV12 → float32 CHW)", "Magma");
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mgmpreproc, "Magma Preprocessing Plugin", plugin_init, "0.1.0", "LGPL", "magma", "https://imeguras.eu.org")
