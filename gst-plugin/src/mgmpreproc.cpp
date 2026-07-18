#include "mgmpreproc.hpp"
#include "kernel_utils.hpp"
#include "magma-meta.h"
#include "magma-hip-stream.hpp"

#include <string>
#include <vector>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

/** --- GOBJECT / GSTREAMER STUFF --- */
GST_DEBUG_CATEGORY_STATIC(magma_preproc_debug);
#define GST_CAT_DEFAULT magma_preproc_debug

enum {
    PROP_0,
    PROP_NET_WIDTH,
    PROP_NET_HEIGHT,
    PROP_SCALE_FACTOR,
    PROP_ENABLE_ROI,
    PROP_ROI_X,
    PROP_ROI_Y,
    PROP_ROI_W,
    PROP_ROI_H,
};

G_DEFINE_TYPE(GstMagmaPreproc, gst_magma_preproc, GST_TYPE_BASE_TRANSFORM)

/**
 * @brief Resolve the directory containing HIP kernel source files.
 *
 * Checks the MAGMA_KERNEL_DIR environment variable first; falls back
 * to the compile-time default MAGMA_KERNEL_SRC_DIR set by the build
 * system. The returned pointer is valid for the lifetime of the
 * process.
 *
 * @return Absolute path to the kernel directory
 */
static const char* find_kernel_dir(void) {
    const char* env = g_getenv("MAGMA_KERNEL_DIR");
    if (env)
        return env;
    return MAGMA_KERNEL_SRC_DIR;
}

/** --- DRM / GBM HELPERS --- */
/** --- TENSOR OUTPUT (DMABuf-backed) --- */
/**
 * @brief Allocate the tensor DMABuf for preprocessing output.
 *
 * Creates a DMABuf-backed GstMemory for the float32 RGB tensor,
 * imports it to HIP, and caches the device pointer for reuse.
 *
 * @param self Preprocessing element
 * @return TRUE on success
 */
static gboolean ensure_tensor(GstMagmaPreproc* self) {
    if (self->d_tensor_output)
        return TRUE;

    gsize n = (gsize)self->net_width * self->net_height * 3;
    gsize bytes = n * sizeof(float);

    /* Allocate tensor via hipMalloc, then export as DMABuf FD */
    hipError_t herr = hipMalloc(&self->d_tensor_output, bytes);
    if (herr != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipMalloc(tensor %zu bytes) failed: %s", bytes, hipGetErrorString(herr));
        return FALSE;
    }

    int dmabuf_fd = -1;
    herr = hipMemGetHandleForAddressRange(&dmabuf_fd, (hipDeviceptr_t)self->d_tensor_output, bytes, hipMemRangeHandleTypeDmaBufFd, 0);
    if (herr != hipSuccess || dmabuf_fd < 0) {
        GST_ERROR_OBJECT(self, "hipMemGetHandleForAddressRange failed: %s", hipGetErrorString(herr));
        (void)hipFree(self->d_tensor_output);
        self->d_tensor_output = nullptr;
        return FALSE;
    }

    /* Wrap in GstMemory (takes ownership of dmabuf_fd) */
    GstAllocator* dma_alloc = gst_dmabuf_allocator_new();
    self->tensor_mem = gst_dmabuf_allocator_alloc(dma_alloc, dmabuf_fd, bytes);
    gst_object_unref(dma_alloc);
    if (!self->tensor_mem) {
        GST_ERROR_OBJECT(self, "gst_dmabuf_allocator_alloc failed");
        close(dmabuf_fd);
        (void)hipFree(self->d_tensor_output);
        self->d_tensor_output = nullptr;
        return FALSE;
    }

    self->tensor_dmabuf_fd = dmabuf_fd;
    self->tensor_alloc_size = bytes;
    GST_INFO_OBJECT(self, "Tensor hipMalloc %dx%dx3 (%zu bytes) fd=%d ptr=%p", self->net_width, self->net_height, bytes, dmabuf_fd, (void*)self->d_tensor_output);
    return TRUE;
}

/**
 * @brief Set a GObject property on the preprocessing element.
 *
 * Supports net-width, net-height, scale-factor, enable-roi,
 * and roi-x/y/w/h. All properties take effect on the next
 * frame processed.
 */
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
    case PROP_ENABLE_ROI:
        self->enable_roi = g_value_get_boolean(value);
        break;
    case PROP_ROI_X:
        self->roi_x = g_value_get_int(value);
        break;
    case PROP_ROI_Y:
        self->roi_y = g_value_get_int(value);
        break;
    case PROP_ROI_W:
        self->roi_w = g_value_get_int(value);
        break;
    case PROP_ROI_H:
        self->roi_h = g_value_get_int(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/**
 * @brief Retrieve a GObject property value.
 */
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
    case PROP_ENABLE_ROI:
        g_value_set_boolean(value, self->enable_roi);
        break;
    case PROP_ROI_X:
        g_value_set_int(value, self->roi_x);
        break;
    case PROP_ROI_Y:
        g_value_set_int(value, self->roi_y);
        break;
    case PROP_ROI_W:
        g_value_set_int(value, self->roi_w);
        break;
    case PROP_ROI_H:
        g_value_set_int(value, self->roi_h);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/**
 * @brief Release all GPU resources held by the element.
 *
 * Unloads the JIT-compiled HIP kernel module, frees device memory
 * for the tensor output and input upload buffer, destroys any
 * imported external memory handle, and chains up to the parent
 * finalize.
 */
static void gst_magma_preproc_finalize(GObject* object) {
    GstMagmaPreproc* self = GST_MAGMA_PREPROC(object);

    if (self->kernel_module) {
        (void)hipModuleUnload(self->kernel_module);
        self->kernel_module = nullptr;
    }

    if (self->d_tensor_output) {
        (void)hipFree(self->d_tensor_output);
        self->d_tensor_output = nullptr;
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
    if (self->d_input_upload) {
        (void)hipFree(self->d_input_upload);
        self->d_input_upload = 0;
    }
    G_OBJECT_CLASS(gst_magma_preproc_parent_class)->finalize(object);
}

/** --- PAD TEMPLATES --- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));
/**
 * @brief Initialise a new GstMagmaPreproc instance.
 *
 * Sets default property values (224×224, scale=1/255), clears
 * all GPU pointers and kernel handles to zero, and enables
 * in-place transform mode so the element modifies the input
 * buffer rather than allocating new output buffers.
 */
static void gst_magma_preproc_init(GstMagmaPreproc* self) {
    self->net_width = 224;
    self->net_height = 224;
    self->scale_factor = 1.0f / 255.0f;

    self->enable_roi = FALSE;
    self->roi_x = 0;
    self->roi_y = 0;
    self->roi_w = 0;
    self->roi_h = 0;

    self->hip_stream = nullptr;
    self->external_memory = nullptr;
    self->d_image = 0;
    self->d_input_upload = 0;
    self->imported = FALSE;

    self->kernel_module = nullptr;
    self->kernel_nv12_to_rgb = nullptr;
    self->kernel_ready = FALSE;

    self->tensor_dmabuf_fd = -1;
    self->tensor_ext_mem = nullptr;
    self->d_tensor_output = nullptr;
    self->tensor_mem = NULL;
    self->tensor_alloc_size = 0;

    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

/**
 * @brief Passthrough caps negotiation — accept NV12 on both sides.
 *
 * Returns the intersection of the proposed caps and the optional
 * filter caps. The element does not re-negotiate format; the
 * transform is in-place.
 */
static GstCaps* gst_magma_preproc_transform_caps(GstBaseTransform* trans, GstPadDirection direction, GstCaps* caps, GstCaps* filter) {
    GstCaps* result = gst_caps_ref(caps);
    if (filter) {
        GstCaps* tmp = gst_caps_intersect_full(result, filter, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(result);
        result = tmp;
    }
    return result;
}

/**
 * @brief Configure the element for a new input/output caps pair.
 *
 * Extracts source frame dimensions and format from the input caps,
 * validates that the format is NV12, and allocates the tensor output
 * buffer via ensure_tensor(). Called once per stream start.
 *
 * @return TRUE on success
 */
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

/**
 * @brief Suggests output buffer size equals input size (in-place).
 */
static gboolean gst_magma_preproc_transform_ip_size(GstBaseTransform* trans, GstPadDirection direction, GstCaps* caps, gsize size, GstCaps* othercaps, gsize* othersize) {
    *othersize = size;
    return TRUE;
}

/** --- TRANSFORM (per-frame): NV12 → tensor --- */
/**
 * @brief Main transform entry point — convert NV12 frame to tensor.
 *
 * Receives an NV12 frame (DMABuf or HIP pointer), optionally crops
 * a region-of-interest, resizes to net_width/net_height with
 * letterboxing, normalizes pixel values, and attaches a
 * MagmaTensorMeta to the output buffer.
 *
 * @param trans The base transform element
 * @param buf   Input/output buffer (NV12 in, NV12+tensor meta out)
 * @return GST_FLOW_OK on success
 */
static GstFlowReturn gst_magma_preproc_transform_ip(GstBaseTransform* trans, GstBuffer* buf) {
    GstMagmaPreproc* self = GST_MAGMA_PREPROC(trans);

    magma_tensor_meta_get_info();

    if (!self->hip_stream) {
        self->hip_stream = magma_get_shared_hip_stream();
        if (!self->hip_stream) {
            GST_ERROR_OBJECT(self, "magma_get_shared_hip_stream failed");
            return GST_FLOW_ERROR;
        }
    }

    // --- Import input frame ---
    MagmaHipMeta* hip_meta = magma_buffer_get_hip_meta(buf);
    if (hip_meta) {
        self->d_image = hip_meta->d_ptr;
    } else {
        GstMemory* mem = gst_buffer_peek_memory(buf, 0);
        if (gst_is_dmabuf_memory(mem)) {
            if (self->external_memory) {
                (void)hipDestroyExternalMemory(self->external_memory);
                self->external_memory = nullptr;
                self->d_image = 0;
            }

            gint raw_fd = gst_dmabuf_memory_get_fd(mem);
            int fd = fcntl(raw_fd, F_DUPFD_CLOEXEC, 0);
            if (fd < 0) {
                GST_ERROR_OBJECT(self, "fcntl(DUPFD) failed");
                return GST_FLOW_ERROR;
            }
            hipExternalMemoryHandleDesc desc{};
            desc.type = hipExternalMemoryHandleTypeOpaqueFd;
            desc.handle.fd = fd;
            desc.size = (gsize)self->in_width * self->in_height * 3 / 2;

            hipError_t herr = hipImportExternalMemory(&self->external_memory, &desc);
            close(fd);
            if (herr != hipSuccess) {
                GST_ERROR_OBJECT(self, "hipImportExternalMemory failed: %s", hipGetErrorString(herr));
                return GST_FLOW_ERROR;
            }

            hipExternalMemoryBufferDesc bdesc{};
            bdesc.offset = 0;
            bdesc.size = desc.size;

            herr = hipExternalMemoryGetMappedBuffer(&self->d_image, self->external_memory, &bdesc);
            if (herr != hipSuccess) {
                (void)hipDestroyExternalMemory(self->external_memory);
                self->external_memory = nullptr;
                return GST_FLOW_ERROR;
            }
        } else {
            // System memory fallback: upload to GPU
            GstMapInfo in_map;
            if (!gst_buffer_map(buf, &in_map, GST_MAP_READ)) {
                GST_ERROR_OBJECT(self, "Failed to map input buffer");
                return GST_FLOW_ERROR;
            }

            gsize frame_bytes = (gsize)self->in_width * self->in_height * 3 / 2;
            if (!self->d_input_upload) {
                hipError_t e = hipMalloc(&self->d_input_upload, frame_bytes);
                if (e != hipSuccess) {
                    gst_buffer_unmap(buf, &in_map);
                    GST_ERROR_OBJECT(self, "hipMalloc(input) failed: %s", hipGetErrorString(e));
                    return GST_FLOW_ERROR;
                }
            }

            hipError_t herr = hipMemcpy(self->d_input_upload, in_map.data, frame_bytes, hipMemcpyHostToDevice);
            gst_buffer_unmap(buf, &in_map);
            if (herr != hipSuccess) {
                GST_ERROR_OBJECT(self, "hipMemcpy(H2D) failed: %s", hipGetErrorString(herr));
                return GST_FLOW_ERROR;
            }
            self->d_image = self->d_input_upload;
        }
    }

    // --- Compile kernel once ---
    if (!self->kernel_ready) {
        std::string kernel_dir(find_kernel_dir());
        std::string kernel_path = kernel_dir + "/preproc_kernels.hip";
        std::string common_path = kernel_dir + "/common.hip";

        HipKernel kern = compile_kernel(kernel_path.c_str(), "nv12_to_rgb_normalized", common_path.c_str());
        if (!kern.func) {
            GST_ERROR_OBJECT(self, "Failed to compile nv12_to_rgb_normalized kernel");
            return GST_FLOW_ERROR;
        }
        self->kernel_module = kern.module;
        self->kernel_nv12_to_rgb = kern.func;
        self->kernel_ready = TRUE;
        GST_INFO_OBJECT(self, "Kernel compiled and loaded from %s", kernel_path.c_str());
    }

    // --- ROI setup and host-side validation ---
    int cx = 0, cy = 0, cw = self->in_width, ch = self->in_height;
    if (self->enable_roi) {
        cx = self->roi_x;
        cy = self->roi_y;
        cw = self->roi_w;
        ch = self->roi_h;
    }

    gboolean roi_valid = (cx >= 0 && cy >= 0 && cw > 0 && ch > 0 && cx + cw <= self->in_width && cy + ch <= self->in_height);

    int block_size = 16;

    if (!roi_valid) {
        hipError_t herr = hipMemsetAsync(self->d_tensor_output, 0, self->tensor_alloc_size, self->hip_stream);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMemsetAsync failed: %s", hipGetErrorString(herr));
            return GST_FLOW_ERROR;
        }
    } else {
        void* d_ptr = (void*)self->d_image;
        int w = self->in_width, h = self->in_height;
        GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
        int stride = vmeta && vmeta->stride[0] > 0 ? vmeta->stride[0] : self->in_width;

        float* t_ptr = self->d_tensor_output;
        int nw = self->net_width, nh = self->net_height;
        float sf = self->scale_factor;
        int grid_x = (nw + block_size - 1) / block_size;
        int grid_y = (nh + block_size - 1) / block_size;

        void* args[] = {&d_ptr, &w, &h, &stride, &t_ptr, &nw, &nh, &cx, &cy, &cw, &ch, &sf};
        hipError_t err = hipModuleLaunchKernel(self->kernel_nv12_to_rgb, grid_x, grid_y, 1, block_size, block_size, 1, 0, self->hip_stream, args, nullptr);
        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipModuleLaunchKernel(nv12_to_rgb) failed: %s", hipGetErrorString(err));
            return GST_FLOW_ERROR;
        }
    }

    // Attach tensor DMABuf as metadata on the buffer
    MagmaTensorMeta* tmeta = magma_buffer_add_tensor_meta(buf, self->tensor_mem, self->net_width, self->net_height, 3);
    if (tmeta) {
        tmeta->roi_x = cx;
        tmeta->roi_y = cy;
        tmeta->roi_w = cw;
        tmeta->roi_h = ch;
    }

    return GST_FLOW_OK;
}

/**
 * @brief Plugin entry point — register the mgmpreproc element.
 *
 * Initialises the debug category and registers GstMagmaPreproc
 * with GStreamer under the element name "mgmpreproc".
 */
static gboolean plugin_init(GstPlugin* plugin) {
    GST_DEBUG_CATEGORY_INIT(magma_preproc_debug, "magma_preproc", 0, "Magma GPU Preprocessor");
    return gst_element_register(plugin, "mgmpreproc", GST_RANK_NONE, GST_TYPE_MAGMA_PREPROC);
}

/**
 * @brief Initialise the GstMagmaPreproc class.
 *
 * Installs properties (net-width, net-height, scale-factor, ROI),
 * pad templates (NV12 in/out), and wires up the GstBaseTransform
 * virtual methods (set_caps, transform_ip, transform_caps,
 * transform_size).
 */
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
    g_object_class_install_property(object_class, PROP_ENABLE_ROI, g_param_spec_boolean("enable-roi", "Enable ROI", "Crop to region of interest before resize", FALSE, G_PARAM_READWRITE));
    g_object_class_install_property(object_class, PROP_ROI_X, g_param_spec_int("roi-x", "ROI X", "Left coordinate of the crop rectangle in the source frame", 0, G_MAXINT, 0, G_PARAM_READWRITE));
    g_object_class_install_property(object_class, PROP_ROI_Y, g_param_spec_int("roi-y", "ROI Y", "Top coordinate of the crop rectangle in the source frame", 0, G_MAXINT, 0, G_PARAM_READWRITE));
    g_object_class_install_property(object_class, PROP_ROI_W, g_param_spec_int("roi-w", "ROI Width", "Width of the crop rectangle in the source frame", 0, G_MAXINT, 0, G_PARAM_READWRITE));
    g_object_class_install_property(object_class, PROP_ROI_H, g_param_spec_int("roi-h", "ROI Height", "Height of the crop rectangle in the source frame", 0, G_MAXINT, 0, G_PARAM_READWRITE));

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
