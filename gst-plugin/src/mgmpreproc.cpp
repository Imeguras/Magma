#include "mgmpreproc.hpp"

#include <string>
#include <vector>
#include <cstring>

/** --- GOBJECT / GSTREAMER STUFF --- */
GST_DEBUG_CATEGORY_STATIC(magma_preproc_debug);
#define GST_CAT_DEFAULT magma_preproc_debug

enum
{
    PROP_0,
    PROP_NET_WIDTH,
    PROP_NET_HEIGHT,
    PROP_SCALE_FACTOR,
};

G_DEFINE_TYPE(
    GstMagmaPreproc,
    gst_magma_preproc,
    GST_TYPE_BASE_TRANSFORM
)

/** --- HIPRTC KERNEL SOURCE --- */
static const char* kernel_source = R"(
extern "C" __global__ void invert_y(unsigned char* frame, int width, int height, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        frame[y * stride + x] = 255 - frame[y * stride + x];
    }
}
)";

/** --- COMPILE KERNEL VIA HIPRTC --- */
static gboolean compile_kernel(GstMagmaPreproc *self) {
    hiprtcProgram prog;
    hiprtcResult r;

    r = hiprtcCreateProgram(&prog, kernel_source, "invert_y.cu", 0, nullptr, nullptr);
    if (r != HIPRTC_SUCCESS) {
        GST_ERROR_OBJECT(self, "hiprtcCreateProgram failed: %d", r);
        return FALSE;
    }

    int device_idx;
    hipError_t err = hipGetDevice(&device_idx);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipGetDevice failed: %s", hipGetErrorString(err));
        hiprtcDestroyProgram(&prog);
        return FALSE;
    }

    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, device_idx);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipGetDeviceProperties failed: %s", hipGetErrorString(err));
        hiprtcDestroyProgram(&prog);
        return FALSE;
    }

    std::string arch_str(props.gcnArchName);
    std::string arch_opt = std::string("--gpu-architecture=") + arch_str;
    const char* opts[] = { arch_opt.c_str() };

    r = hiprtcCompileProgram(prog, 1, opts);
    if (r != HIPRTC_SUCCESS) {
        size_t log_size;
        hiprtcGetProgramLogSize(prog, &log_size);
        std::string log;
        if (log_size > 0) {
            log.resize(log_size);
            hiprtcGetProgramLog(prog, &log[0]);
        }
        GST_ERROR_OBJECT(self, "hiprtcCompileProgram failed (arch=%s): %s",
                         arch_str.c_str(), log.c_str());
        hiprtcDestroyProgram(&prog);
        return FALSE;
    }

    size_t code_size;
    r = hiprtcGetCodeSize(prog, &code_size);
    if (r != HIPRTC_SUCCESS) {
        GST_ERROR_OBJECT(self, "hiprtcGetCodeSize failed: %d", r);
        hiprtcDestroyProgram(&prog);
        return FALSE;
    }

    std::vector<char> code(code_size);
    r = hiprtcGetCode(prog, code.data());
    if (r != HIPRTC_SUCCESS) {
        GST_ERROR_OBJECT(self, "hiprtcGetCode failed: %d", r);
        hiprtcDestroyProgram(&prog);
        return FALSE;
    }

    hiprtcDestroyProgram(&prog);

    err = hipModuleLoadData(&self->kernel_module, code.data());
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipModuleLoadData failed: %s", hipGetErrorString(err));
        return FALSE;
    }

    err = hipModuleGetFunction(&self->kernel_func, self->kernel_module, "invert_y");
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipModuleGetFunction failed: %s", hipGetErrorString(err));
        (void)hipModuleUnload(self->kernel_module);
        self->kernel_module = nullptr;
        return FALSE;
    }

    self->kernel_ready = TRUE;
    GST_INFO_OBJECT(self, "Kernel compiled and loaded (arch=%s)", arch_str.c_str());
    return TRUE;
}

/** --- PROPERTIES --- */
static void gst_magma_preproc_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec){
    GstMagmaPreproc *self = GST_MAGMA_PREPROC(object);
    switch (prop_id){
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

static void gst_magma_preproc_get_property(GObject *object, guint prop_id, GValue *value,GParamSpec *pspec){
    GstMagmaPreproc *self = GST_MAGMA_PREPROC(object);
    switch (prop_id){
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
static void gst_magma_preproc_finalize(GObject *object) {
    GstMagmaPreproc *self = GST_MAGMA_PREPROC(object);

    if (self->kernel_module) {
        (void)hipModuleUnload(self->kernel_module);
        self->kernel_module = nullptr;
    }
    if (self->external_memory) {
        (void)hipDestroyExternalMemory(self->external_memory);
        self->external_memory = nullptr;
    }
    if (self->hip_stream) {
        (void)hipStreamDestroy(self->hip_stream);
        self->hip_stream = nullptr;
    }

    G_OBJECT_CLASS(gst_magma_preproc_parent_class)->finalize(object);
}

/** --- PAD TEMPLATES --- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12")
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12")
);

/** --- INIT --- */
static void gst_magma_preproc_init(GstMagmaPreproc *self)
{
    self->net_width = 224;
    self->net_height = 224;
    self->scale_factor = 1.0f / 255.0f;

    self->hip_stream = nullptr;
    self->d_tensor_output = nullptr;
    self->external_memory = nullptr;
    self->d_image = 0;
    self->imported = FALSE;

    self->kernel_module = nullptr;
    self->kernel_func = nullptr;
    self->kernel_ready = FALSE;

    gst_base_transform_set_in_place(
        GST_BASE_TRANSFORM(self),
        TRUE
    );
}

/** --- CAPS NEGOTIATION --- */
static GstCaps *gst_magma_preproc_transform_caps(
    GstBaseTransform *trans,
    GstPadDirection direction,
    GstCaps *caps,
    GstCaps *filter){

    GstCaps *result = gst_caps_ref(caps);

    if (filter) {
        GstCaps *tmp = gst_caps_intersect_full(
            result,
            filter,
            GST_CAPS_INTERSECT_FIRST
        );
        gst_caps_unref(result);
        result = tmp;
    }

    return result;
}

static gboolean gst_magma_preproc_set_caps(
    GstBaseTransform *trans,
    GstCaps *incaps,
    GstCaps *outcaps){
    GstMagmaPreproc *self = GST_MAGMA_PREPROC(trans);

    GstStructure *s = gst_caps_get_structure(incaps, 0);

    gst_structure_get_int(s, "width", &self->in_width);
    gst_structure_get_int(s, "height", &self->in_height);

    const gchar *fmt = gst_structure_get_string(s, "format");
    self->in_format = gst_video_format_from_string(fmt);

    if (self->in_format != GST_VIDEO_FORMAT_NV12) {
        GST_ERROR_OBJECT(self, "Only NV12 supported currently");
        return FALSE;
    }

    GST_INFO_OBJECT(self, "Input %dx%d %s", self->in_width, self->in_height, fmt);
    return TRUE;
}

static gboolean gst_magma_preproc_transform_ip_size(
    GstBaseTransform *trans,
    GstPadDirection direction,
    GstCaps *caps,
    gsize size,
    GstCaps *othercaps,
    gsize *othersize){
    // transform_ip keeps the same buffer — output size = input size
    *othersize = size;
    return TRUE;
}

/** --- TRANSFORM (per-frame): DMABuf only --- */
static GstFlowReturn gst_magma_preproc_transform_ip(GstBaseTransform *trans, GstBuffer *buf){
    GstMagmaPreproc *self = GST_MAGMA_PREPROC(trans);

    GstMemory *mem = gst_buffer_peek_memory(buf, 0);

    if (!gst_is_dmabuf_memory(mem)) {
        GST_ERROR_OBJECT(self, "mgmpreproc requires DMABuf memory — pipe through mgmvideoconvert first");
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

    // Import DMABuf into HIP once (reuses the same external memory)
    if (!self->imported) {
        hipExternalMemoryHandleDesc desc{};
        desc.type = hipExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = fd;
        desc.size = self->in_width * self->in_height * 3 / 2;

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
        GST_INFO_OBJECT(self, "HIP DMABUF import successful (ptr=%p, fd=%d)",
                        (void*)self->d_image, fd);
    }

    // Compile kernel once on first frame
    if (!self->kernel_ready) {
        if (!compile_kernel(self)) {
            GST_ERROR_OBJECT(self, "Failed to compile GPU kernel");
            return GST_FLOW_ERROR;
        }
    }

    // --- Launch kernel: invert Y plane ---
    void* d_ptr = (void*)self->d_image;
    int w = self->in_width;
    int h = self->in_height;

    // Use stride from video metadata if available, else assume tight packing
    int stride = self->in_width;
    GstVideoMeta *vmeta = gst_buffer_get_video_meta(buf);
    if (vmeta && vmeta->stride[0] > 0)
        stride = vmeta->stride[0];

    void* args[] = { &d_ptr, &w, &h, &stride };

    int block_size = 16;
    int grid_x = (w + block_size - 1) / block_size;
    int grid_y = (h + block_size - 1) / block_size;

    hipError_t err = hipModuleLaunchKernel(
        self->kernel_func,
        grid_x, grid_y, 1,
        block_size, block_size, 1,
        0,
        self->hip_stream,
        args, nullptr
    );

    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipModuleLaunchKernel failed: %s", hipGetErrorString(err));
        return GST_FLOW_ERROR;
    }

    err = hipStreamSynchronize(self->hip_stream);
    if (err != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipStreamSynchronize failed: %s", hipGetErrorString(err));
        return GST_FLOW_ERROR;
    }

    return GST_FLOW_OK;
}

/** --- PLUGIN REGISTRATION --- */
static gboolean plugin_init(GstPlugin *plugin){
    GST_DEBUG_CATEGORY_INIT(magma_preproc_debug, "magma_preproc", 0,
                            "Magma GPU Preprocessor");
    return gst_element_register(
        plugin,
        "mgmpreproc",
        GST_RANK_NONE,
        GST_TYPE_MAGMA_PREPROC
    );
}

static void gst_magma_preproc_class_init(GstMagmaPreprocClass *klass){
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GObjectClass *object_class = G_OBJECT_CLASS(klass);
    object_class->set_property = gst_magma_preproc_set_property;
    object_class->get_property = gst_magma_preproc_get_property;
    object_class->finalize = gst_magma_preproc_finalize;

    g_object_class_install_property(
        object_class,
        PROP_NET_WIDTH,
        g_param_spec_int(
            "net-width",
            "Network Input Width",
            "Width of the input tensor expected by the neural network",
            1, G_MAXINT, 224, G_PARAM_READWRITE
        )
    );
    g_object_class_install_property(
        object_class,
        PROP_NET_HEIGHT,
        g_param_spec_int(
            "net-height",
            "Network Input Height",
            "Height of the input tensor expected by the neural network",
            1, G_MAXINT, 224, G_PARAM_READWRITE
        )
    );
    g_object_class_install_property(
        object_class,
        PROP_SCALE_FACTOR,
        g_param_spec_float(
            "scale-factor",
            "Scale Factor",
            "Factor by which to scale the input tensor",
            0.0, G_MAXFLOAT, 1.0f, G_PARAM_READWRITE
        )
    );

    gst_element_class_add_static_pad_template(
        GST_ELEMENT_CLASS(klass),
        &sink_template
    );
    gst_element_class_add_static_pad_template(
        GST_ELEMENT_CLASS(klass),
        &src_template
    );

    GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS(klass);
    trans_class->transform_size = gst_magma_preproc_transform_ip_size;
    trans_class->transform_caps = gst_magma_preproc_transform_caps;
    trans_class->set_caps = gst_magma_preproc_set_caps;
    trans_class->transform_ip = gst_magma_preproc_transform_ip;

    gst_element_class_set_static_metadata(
        element_class,
        "Magma Preprocessor",
        "Filter/Video",
        "ROCm/HIP tensor preprocessing with GPU kernel demo",
        "Magma"
    );
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    mgmpreproc,
    "Magma Preprocessing Plugin",
    plugin_init,
    "0.1.0",
    "LGPL",
    "magma",
    "https://imeguras.eu.org"
)
