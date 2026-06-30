#include "mgminfer.hpp"
#include "magma-meta.h"
#include "kernel_utils.hpp"

#include <string>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <gbm.h>
#include <migraphx/migraphx.hpp>

/** --- GOBJECT / GSTREAMER STUFF --- */
GST_DEBUG_CATEGORY_STATIC(magma_infer_debug);
#define GST_CAT_DEFAULT magma_infer_debug

enum {
    PROP_0,
    PROP_MODEL_PATH,
    PROP_INFERENCE_INTERVAL,
};

G_DEFINE_TYPE(GstMagmaInfer, gst_magma_infer, GST_TYPE_BASE_TRANSFORM)

namespace {

struct MigraphXModel {
    migraphx::program prog;
    std::string input_name;
    migraphx::shape input_shape;
    std::vector<std::size_t> input_lengths;
    std::vector<std::pair<std::string, migraphx::shape>> all_params;
    hipDeviceptr_t d_output_scratch;
    std::size_t output_scratch_bytes;

    MigraphXModel() : d_output_scratch(nullptr), output_scratch_bytes(0) {}

    ~MigraphXModel() {
        if (d_output_scratch) {
            (void)hipFree(d_output_scratch);
            d_output_scratch = nullptr;
        }
    }
};

} // anonymous namespace

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

    if (self->migraphx_model) {
        delete static_cast<MigraphXModel*>(self->migraphx_model);
        self->migraphx_model = nullptr;
    }

    g_free(self->model_path);
    self->model_path = NULL;

    G_OBJECT_CLASS(gst_magma_infer_parent_class)->finalize(object);
}

static gboolean gst_magma_infer_start(GstBaseTransform* trans) {
    GstMagmaInfer* self = GST_MAGMA_INFER(trans);

    if (!self->model_path) {
        GST_WARNING_OBJECT(self, "no model-path set — MIGraphX model not loaded");
        return TRUE;
    }

    auto path = std::string(self->model_path);
    auto model = std::make_unique<MigraphXModel>();
    try {
        if (path.size() >= 4 && path.substr(path.size() - 4) == ".mxr") {
            model->prog = migraphx::load(path.c_str());
            GST_INFO_OBJECT(self, "loaded pre-compiled MIGraphX model from %s", path.c_str());

        } else if (path.size() >= 5 && path.substr(path.size() - 5) == ".onnx") {
            model->prog = migraphx::parse_onnx(path.c_str());
            GST_INFO_OBJECT(self, "parsed ONNX model from %s, compiling for GPU...", path.c_str());
            // TODO: yep this might need change if we add multiple GPU's, some people are born rich ig
            model->prog.compile(migraphx::target("gpu"));
            GST_INFO_OBJECT(self, "MIGraphX compiled for GPU");
        } else {

            GST_ERROR_OBJECT(self, "unsupported model file extension (must be .mxr or .onnx)");
            return FALSE;
        }
    } catch (const std::exception& e) {
        GST_ERROR_OBJECT(self, "MIGraphX model load failed: %s", e.what());
        return FALSE;
    }

    /* extract parameter info — find the real input (skip internal params like main:#...) */
    auto param_shapes = model->prog.get_parameter_shapes();
    auto names = param_shapes.names();
    if (names.empty()) {
        GST_ERROR_OBJECT(self, "MIGraphX model has no parameters");
        return FALSE;
    }

    model->input_name.clear();
    model->all_params.clear();
    model->d_output_scratch = nullptr;
    model->output_scratch_bytes = 0;

    for (auto& n : names) {
        auto s = param_shapes[n];
        model->all_params.emplace_back(n, s);
        /* real input param — not synthetic (synthetic has main:# prefix) */
        if (n[0] != 'm' || strncmp(n, "main:#", 6) != 0) {
            if (model->input_name.empty()) {
                model->input_name = n;
                model->input_shape = s;
                model->input_lengths = s.lengths();
            }
        }
    }

    if (model->input_name.empty()) {
        GST_ERROR_OBJECT(self, "could not identify input parameter");
        return FALSE;
    }

    /* pre-allocate GPU scratch for output params */
    std::size_t max_scratch = 0;
    for (auto& [n, s] : model->all_params) {
        if (n != model->input_name) {
            max_scratch = std::max(max_scratch, s.bytes());
        }
    }
    if (max_scratch > 0) {
        hipError_t err = hipMalloc(&model->d_output_scratch, max_scratch);
        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMalloc(output_scratch %zu) failed: %s",
                max_scratch, hipGetErrorString(err));
            return FALSE;
        }
        model->output_scratch_bytes = max_scratch;
        GST_INFO_OBJECT(self, "allocated %zu bytes GPU scratch for output params", max_scratch);
    }

    self->migraphx_model = model.release();
    self->model_loaded = TRUE;
    GST_INFO_OBJECT(self, "MIGraphX model ready — input '%s' (%zu dims), %zu total params",
        static_cast<MigraphXModel*>(self->migraphx_model)->input_name.c_str(),
        static_cast<MigraphXModel*>(self->migraphx_model)->input_lengths.size(),
        static_cast<MigraphXModel*>(self->migraphx_model)->all_params.size());

    return TRUE;
}

/** --- STOP (READY) --- */
static gboolean gst_magma_infer_stop(GstBaseTransform* trans) {
    GstMagmaInfer* self = GST_MAGMA_INFER(trans);

    if (self->migraphx_model) {
        delete static_cast<MigraphXModel*>(self->migraphx_model);
        self->migraphx_model = nullptr;
    }
    GST_INFO_OBJECT(self, "MIGraphX model unloaded");

    if (self->hip_stream) {
        (void)hipStreamDestroy(self->hip_stream);
        self->hip_stream = nullptr;
    }
    GST_INFO_OBJECT(self, "HIP stream destroyed");

    self->model_loaded = FALSE;

    return TRUE;
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

    self->migraphx_model = nullptr;
    self->model_loaded = FALSE;

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

/** --- import a DMABuf → HIP device pointer (must destroy handle after use) --- */
static hipExternalMemory_t import_dmabuf_to_hip(int dmabuf_fd, gsize size, hipDeviceptr_t* d_ptr) {
    int hip_fd = fcntl(dmabuf_fd, F_DUPFD_CLOEXEC, 0);
    if (hip_fd < 0) return nullptr;

    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = hip_fd;
    desc.size = size;
    hipExternalMemory_t ext_mem;
    hipError_t err = hipImportExternalMemory(&ext_mem, &desc);
    close(hip_fd);
    if (err != hipSuccess) return nullptr;

    hipExternalMemoryBufferDesc bdesc{};
    bdesc.offset = 0;
    bdesc.size = size;
    err = hipExternalMemoryGetMappedBuffer(d_ptr, ext_mem, &bdesc);
    if (err != hipSuccess) {
        (void)hipDestroyExternalMemory(ext_mem);
        return nullptr;
    }
    return ext_mem;
}

/** --- dummy kernel fallback (removed — model must succeed or fail) --- */

/** --- attach a single detection to the buffer --- */
static GstFlowReturn attach_inference_meta(GstMagmaInfer* self, GstBuffer* buf, gint num_objects) {
    MagmaInferenceMeta* m = magma_buffer_add_inference_meta(buf, self->in_width, self->in_height);
    if (!m) { GST_ERROR_OBJECT(self, "failed to attach inference meta"); return GST_FLOW_ERROR; }
    m->num_objects = num_objects;
    m->objects_gpu = gst_memory_ref(self->objects_mem);
    return GST_FLOW_OK;
}

/** --- TRANSFORM --- */
static GstFlowReturn gst_magma_infer_transform_ip(GstBaseTransform* trans, GstBuffer* buf) {
    GstMagmaInfer* self = GST_MAGMA_INFER(trans);

    self->frame_counter++;
    if ((self->frame_counter - 1) % self->inference_interval != 0)
        return GST_FLOW_OK;

    MagmaTensorMeta* tmeta = magma_buffer_get_tensor_meta(buf);
    if (!tmeta) {
        GST_WARNING_OBJECT(self, "no tensor meta — skipping inference");
        return GST_FLOW_OK;
    }

    /* ensure stream */
    if (!self->hip_stream) {
        hipError_t err = hipStreamCreate(&self->hip_stream);
        if (err != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipStreamCreate failed: %s", hipGetErrorString(err));
            return GST_FLOW_ERROR;
        }
    }

    /* MIGraphX path */
    if (self->model_loaded) {
        auto* model = static_cast<MigraphXModel*>(self->migraphx_model);

        if (model->input_lengths.size() == 4 &&
            (int)model->input_lengths[0] == 1 &&
            (int)model->input_lengths[1] == tmeta->channels &&
            (int)model->input_lengths[2] == tmeta->height &&
            (int)model->input_lengths[3] == tmeta->width) {

            /* import tensor DMABuf → HIP */
            int tensor_fd = gst_dmabuf_memory_get_fd(tmeta->tensor_mem);
            gsize tensor_bytes = gst_memory_get_sizes(tmeta->tensor_mem, NULL, NULL);
            hipDeviceptr_t d_tensor = 0;
            hipExternalMemory_t tensor_ext = import_dmabuf_to_hip(tensor_fd, tensor_bytes, &d_tensor);
            if (!tensor_ext || !d_tensor) {
                GST_ERROR_OBJECT(self, "failed to import tensor DMABuf to HIP");
                return GST_FLOW_ERROR;
            }

            /* build argument map — pass all params */
            try {
                migraphx::program_parameters eval_args;
                for (auto& [n, s] : model->all_params) {
                    migraphx::shape arg_shape(s.type(), s.lengths());
                    if (n == model->input_name) {
                        eval_args.add(n.c_str(), migraphx::argument(arg_shape, (void*)d_tensor));
                    } else {
                        eval_args.add(n.c_str(), migraphx::argument(arg_shape, (void*)model->d_output_scratch));
                    }
                }

                auto outputs = model->prog.eval(eval_args);
                (void)hipDestroyExternalMemory(tensor_ext);

                if (outputs.empty()) {
                    GST_ERROR_OBJECT(self, "MIGraphX returned no outputs");
                    return GST_FLOW_ERROR;
                }

                auto output_arg = outputs.front();
                auto output_shape = output_arg.get_shape();
                auto* d_output = static_cast<const char*>(output_arg.data());
                gsize output_bytes = output_shape.bytes();
                gsize copy_bytes = std::min(output_bytes, self->max_objects * sizeof(MagmaInferObjectGPU));

                hipError_t herr = hipMemcpyDtoD(self->d_objects, (hipDeviceptr_t)d_output, copy_bytes);
                if (herr != hipSuccess) {
                    GST_ERROR_OBJECT(self, "hipMemcpyDtoD failed: %s", hipGetErrorString(herr));
                    return GST_FLOW_ERROR;
                }

                /* read first float → confidence for object 0 */
                float confidence = 0.0f;
                (void)hipMemcpyDtoH(&confidence, self->d_objects, sizeof(float));
                MagmaInferObjectGPU objs[1] = {};
                objs[0].class_id = 1;
                objs[0].confidence = confidence;
                objs[0].x = 0.0f; objs[0].y = 0.0f; objs[0].width = 1.0f; objs[0].height = 1.0f;
                (void)hipMemcpyHtoD(self->d_objects, objs, sizeof(objs));

                GST_LOG_OBJECT(self, "frame %u — MIGraphX inference, confidence=%.4f",
                    self->frame_counter, confidence);
                return attach_inference_meta(self, buf, 1);
            } catch (const std::exception& e) {
                GST_ERROR_OBJECT(self, "MIGraphX eval failed: %s", e.what());
                (void)hipDestroyExternalMemory(tensor_ext);
                return GST_FLOW_ERROR;
            }
        } else {
            GST_ERROR_OBJECT(self, "tensor %dx%dx%d does not match model (expected %zu dims)",
                tmeta->channels, tmeta->height, tmeta->width, model->input_lengths.size());
            return GST_FLOW_ERROR;
        }
    }

    GST_ERROR_OBJECT(self, "no model loaded — cannot infer");
    return GST_FLOW_ERROR;
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
    trans->start = gst_magma_infer_start;
    trans->stop = gst_magma_infer_stop;

    magma_inference_meta_get_info();
    magma_tensor_meta_get_info();

    GST_DEBUG_CATEGORY_INIT(magma_infer_debug, "magma_infer", 0, "Magma Inference Plugin");
}

/** --- PLUGIN REGISTRATION --- */
static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "mgminfer", GST_RANK_NONE, GST_TYPE_MAGMA_INFER);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mgminfer, "Magma Inference Plugin", plugin_init, "0.1.0", "LGPL", "magma", "https://imeguras.eu.org")
