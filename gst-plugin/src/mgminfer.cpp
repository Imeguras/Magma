#include "mgminfer.hpp"
#include "magma-meta.h"
#include "kernel_utils.hpp"
#include "magma-hip-stream.hpp"

#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <gst/allocators/gstdmabuf.h>
#include <xf86drm.h>
#include <migraphx/migraphx.hpp>
#include <rocprofiler-sdk-roctx/roctx.h>

/** --- GOBJECT / GSTREAMER STUFF --- */
GST_DEBUG_CATEGORY_STATIC(magma_infer_debug);
#define GST_CAT_DEFAULT magma_infer_debug

/* forward declarations */
static gboolean gst_magma_infer_decide_allocation(GstBaseTransform* trans, GstQuery* query);

enum {
    PROP_0,
    PROP_ONNX_MODEL_PATH,
    PROP_MXR_MODEL_PATH,
    PROP_INFERENCE_INTERVAL,
    PROP_PARSER_PLUGIN,
    PROP_PARSER_FUNC,
    PROP_CONFIDENCE_THRESH,
    PROP_NMS_THRESH,
    PROP_MAX_DETECTIONS,
    PROP_CLASS_FILTER,
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
    int model_width;
    int model_height;

    MigraphXModel() : d_output_scratch(nullptr), output_scratch_bytes(0), model_width(0), model_height(0) {
    }

    ~MigraphXModel() {
        if (d_output_scratch) {
            (void)hipFree(d_output_scratch);
            d_output_scratch = nullptr;
        }
    }
};

} // anonymous namespace

/** --- OUTPUT OBJECTS GPU BUFFER --- */
/**
 * @brief Ensure GPU output buffers for detection objects are allocated.
 *
 * Allocates max_detections * sizeof(MagmaInferObjectGPU) on the GPU
 * plus a GPU counter. Reallocates if max_objects has changed.
 *
 * @param self Inference element
 * @return TRUE on success
 */
static gboolean ensure_objects_output(GstMagmaInfer* self) {
    if (self->d_objects)
        return TRUE;

    gsize bytes = self->max_objects * sizeof(MagmaInferObjectGPU);

    hipError_t herr = hipMalloc(&self->d_objects, bytes);
    if (herr != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipMalloc(objects %zu) failed: %s", bytes, hipGetErrorString(herr));
        return FALSE;
    }

    GST_INFO_OBJECT(self, "Objects hipMalloc (%zu bytes, max %u objects) ptr=%p", bytes, self->max_objects, (void*)self->d_objects);

    // GPU count buffer
    if (!self->d_num_det) {
        hipError_t e = hipMalloc(&self->d_num_det, sizeof(int));
        if (e != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMalloc(d_num_det) failed: %s", hipGetErrorString(e));
            return FALSE;
        }
    }

    return TRUE;
}

/** --- PAD TEMPLATES --- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

/** --- PROPERTIES --- */
static void gst_magma_infer_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
    GstMagmaInfer* self = GST_MAGMA_INFER(object);

    switch (prop_id) {
    case PROP_ONNX_MODEL_PATH:
        g_free(self->onnx_model_path);
        self->onnx_model_path = g_value_dup_string(value);
        GST_INFO_OBJECT(self, "ONNX model path set to %s", self->onnx_model_path);
        break;
    case PROP_MXR_MODEL_PATH:
        g_free(self->mxr_model_path);
        self->mxr_model_path = g_value_dup_string(value);
        GST_INFO_OBJECT(self, "MXR model path set to %s", self->mxr_model_path);
        break;
    case PROP_INFERENCE_INTERVAL:
        self->inference_interval = g_value_get_uint(value);
        GST_INFO_OBJECT(self, "inference interval set to %u", self->inference_interval);
        break;
    case PROP_PARSER_PLUGIN:
        g_free(self->parser_plugin_path);
        self->parser_plugin_path = g_value_dup_string(value);
        GST_INFO_OBJECT(self, "parser plugin set to %s", self->parser_plugin_path);
        break;
    case PROP_PARSER_FUNC:
        g_free(self->parser_func_name);
        self->parser_func_name = g_value_dup_string(value);
        break;
    case PROP_CONFIDENCE_THRESH:
        self->confidence_thresh = g_value_get_float(value);
        break;
    case PROP_NMS_THRESH:
        self->nms_thresh = g_value_get_float(value);
        break;
    case PROP_MAX_DETECTIONS:
        self->max_detections = g_value_get_uint(value);
        self->max_objects = self->max_detections;
        break;
    case PROP_CLASS_FILTER:
        self->class_filter = g_value_get_int(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_magma_infer_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
    GstMagmaInfer* self = GST_MAGMA_INFER(object);

    switch (prop_id) {
    case PROP_ONNX_MODEL_PATH:
        g_value_set_string(value, self->onnx_model_path);
        break;
    case PROP_MXR_MODEL_PATH:
        g_value_set_string(value, self->mxr_model_path);
        break;
    case PROP_INFERENCE_INTERVAL:
        g_value_set_uint(value, self->inference_interval);
        break;
    case PROP_PARSER_PLUGIN:
        g_value_set_string(value, self->parser_plugin_path);
        break;
    case PROP_PARSER_FUNC:
        g_value_set_string(value, self->parser_func_name);
        break;
    case PROP_CONFIDENCE_THRESH:
        g_value_set_float(value, self->confidence_thresh);
        break;
    case PROP_NMS_THRESH:
        g_value_set_float(value, self->nms_thresh);
        break;
    case PROP_MAX_DETECTIONS:
        g_value_set_uint(value, self->max_detections);
        break;
    case PROP_CLASS_FILTER:
        g_value_set_int(value, self->class_filter);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

/**
 * @brief Finalize the GstMagmaInfer object, releasing all allocated resources.
 *
 * @return void
 *
 */
static void gst_magma_infer_finalize(GObject* object) {
    GstMagmaInfer* self = GST_MAGMA_INFER(object);

    if (self->d_objects) {
        (void)hipFree(self->d_objects);
        self->d_objects = nullptr;
    }

    if (self->d_num_det) {
        hipError_t e = hipFree(self->d_num_det);
        if (e != hipSuccess)
            GST_WARNING_OBJECT(self, "hipFree(d_num_det) failed: %s", hipGetErrorString(e));
        self->d_num_det = nullptr;
    }

    if (self->cached_tensor_ext) {
        (void)hipDestroyExternalMemory(self->cached_tensor_ext);
        self->cached_tensor_ext = nullptr;
    }
    self->cached_tensor_dptr = 0;
    self->cached_tensor_mem = NULL;

    if (self->migraphx_model) {
        delete static_cast<MigraphXModel*>(self->migraphx_model);
        self->migraphx_model = nullptr;
    }

    if (self->parser_handle) {
        dlclose(self->parser_handle);
        self->parser_handle = nullptr;
    }
    self->parser_func = nullptr;
    if (self->mxr_model_path) {
        g_free(self->mxr_model_path);
        self->mxr_model_path = NULL;
    }
    if (self->onnx_model_path) {
        g_free(self->onnx_model_path);
        self->onnx_model_path = NULL;
    }
    g_free(self->parser_plugin_path);
    self->parser_plugin_path = NULL;
    g_free(self->parser_func_name);
    self->parser_func_name = NULL;

    G_OBJECT_CLASS(gst_magma_infer_parent_class)->finalize(object);
}
// this should be split up so its easier to read but im too lazy and with c like functions its always annoying
/**
 * @brief Start the GstMagmaInfer element as per gstreamer convention, loading the model and parser plugin if necessary.
 *        Starts by trying to load a pre-compiled MIGraphX model (.mxr). If that fails, it falls back to compiling from an ONNX model (.onnx) if both are provided.
 *
 * @return TRUE if successful, FALSE otherwise.
 *
 */
static gboolean gst_magma_infer_start(GstBaseTransform* trans) {
    GstMagmaInfer* self = GST_MAGMA_INFER(trans);
    GST_DEBUG_OBJECT(self,
                     "mgminfer start() called, mxr_model_path=%s onnx_model_path=%s parser=%s\n",
                     self->mxr_model_path ? self->mxr_model_path : "(null)",
                     self->onnx_model_path ? self->onnx_model_path : "(null)",
                     self->parser_plugin_path ? self->parser_plugin_path : "(null)");

    if (!self->mxr_model_path && !self->onnx_model_path) {
        GST_ERROR_OBJECT(self, "Neither mxr-model-path nor onnx-model-path was provided. At least one is required.");
        return FALSE;
    }

    std::string mxr_path = self->mxr_model_path ? std::string(self->mxr_model_path) : std::string();
    std::string onnx_path = self->onnx_model_path ? std::string(self->onnx_model_path) : std::string();

    // Determine the ultimate file path we intend to save the compiled model to
    std::string target_mxr_save_path = mxr_path;
    if (target_mxr_save_path.empty() && !onnx_path.empty()) {
        target_mxr_save_path = onnx_path.substr(0, onnx_path.size() - 5) + ".mxr";
    }

    auto model = std::make_unique<MigraphXModel>();
    bool model_loaded = false;

    if (!mxr_path.empty()) {
        try {
            GST_INFO_OBJECT(self, "Attempting to load pre-compiled MIGraphX model from %s", mxr_path.c_str());
            model->prog = migraphx::load(mxr_path.c_str());
            GST_INFO_OBJECT(self, "Successfully loaded pre-compiled MIGraphX model from %s", mxr_path.c_str());
            model_loaded = true;
        } catch (const std::exception& e) {
            GST_WARNING_OBJECT(self, "Failed to load pre-compiled model from %s (Error: %s).", mxr_path.c_str(), e.what());
            if (onnx_path.empty()) {
                GST_ERROR_OBJECT(self, "No .onnx fallback provided. Cannot recover.");
                return FALSE;
            }
            GST_INFO_OBJECT(self, "Falling back to compiling from ONNX...");
        }
    }

    // Womp womp... you get to compile it from ONNX.
    if (!model_loaded) {
        if (onnx_path.empty()) {
            GST_ERROR_OBJECT(self, "Could not load .mxr model and no fallback .onnx path was provided.");
            return FALSE;
        }

        try {
            GST_INFO_OBJECT(self, "Parsing ONNX model from %s...", onnx_path.c_str());
            auto prog_tmp = migraphx::parse_onnx(onnx_path.c_str());

            GST_INFO_OBJECT(self, "Compiling ONNX model for GPU target...");
            prog_tmp.compile(migraphx::target("gpu"));

            // Move it into our runtime container
            model->prog = std::move(prog_tmp);
            model_loaded = true;

            // Generate/Overwrite the target .mxr path so it's production-ready for next time
            try {
                GST_INFO_OBJECT(self, "Saving/Overwriting optimized MIGraphX model to %s", target_mxr_save_path.c_str());
                migraphx::save(model->prog, target_mxr_save_path.c_str());
                GST_INFO_OBJECT(self, "Saved compiled model successfully.");
            } catch (const std::exception& save_ex) {
                // If saving fails (e.g. read-only directory), don't crash the pipeline, we can still run in RAM!
                GST_WARNING_OBJECT(self, "Model compiled successfully but failed to serialize to disk: %s", save_ex.what());
            }

        } catch (const std::exception& e) {
            GST_ERROR_OBJECT(self, "MIGraphX ONNX parsing/compilation failed: %s", e.what());
            return FALSE;
        }
    }

    /* Second step load parser plugin if configured (after MIGraphX init, to avoid conflicts) */
    if (self->parser_plugin_path && !self->parser_handle) {
        GST_INFO_OBJECT(self, "loading parser plugin: %s", self->parser_plugin_path);
        self->parser_handle = dlopen(self->parser_plugin_path, RTLD_NOW | RTLD_LOCAL);
        if (!self->parser_handle) {
            GST_ERROR_OBJECT(self, "failed to load parser plugin '%s': %s", self->parser_plugin_path, dlerror());
            return FALSE;
        }
        GST_INFO_OBJECT(self, "dlopen succeeded, looking up symbol %s", self->parser_func_name);
        self->parser_func = (MagmaParseFunc)dlsym(self->parser_handle, self->parser_func_name);
        if (!self->parser_func) {
            GST_ERROR_OBJECT(self, "symbol '%s' not found in parser plugin: %s", self->parser_func_name, dlerror());
            dlclose(self->parser_handle);
            self->parser_handle = nullptr;
            return FALSE;
        }
        GST_INFO_OBJECT(self, "loaded parser plugin '%s' → %s (func=%p)", self->parser_plugin_path, self->parser_func_name, (void*)self->parser_func);
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
                if (model->input_lengths.size() >= 4) {
                    model->model_width = (int)model->input_lengths[3];
                    model->model_height = (int)model->input_lengths[2];
                } else if (model->input_lengths.size() == 3) {
                    model->model_width = (int)model->input_lengths[2];
                    model->model_height = (int)model->input_lengths[1];
                }
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
            GST_ERROR_OBJECT(self, "hipMalloc(output_scratch %zu) failed: %s", max_scratch, hipGetErrorString(err));
            return FALSE;
        }
        model->output_scratch_bytes = max_scratch;
        GST_INFO_OBJECT(self, "allocated %zu bytes GPU scratch for output params", max_scratch);
    }

    self->migraphx_model = model.release();
    self->model_loaded = TRUE;
    GST_INFO_OBJECT(self,
                    "MIGraphX model ready — input '%s' (%zu dims), %zu total params",
                    static_cast<MigraphXModel*>(self->migraphx_model)->input_name.c_str(),
                    static_cast<MigraphXModel*>(self->migraphx_model)->input_lengths.size(),
                    static_cast<MigraphXModel*>(self->migraphx_model)->all_params.size());

    return TRUE;
}

/** --- STOP (READY → NULL) --- */
static gboolean gst_magma_infer_stop(GstBaseTransform* trans) {
    GstMagmaInfer* self = GST_MAGMA_INFER(trans);

    if (self->migraphx_model) {
        delete static_cast<MigraphXModel*>(self->migraphx_model);
        self->migraphx_model = nullptr;
    }
    GST_INFO_OBJECT(self, "MIGraphX model unloaded");

    /* destroy cached tensor DMABuf import */
    if (self->cached_tensor_ext) {
        (void)hipDestroyExternalMemory(self->cached_tensor_ext);
        self->cached_tensor_ext = nullptr;
    }
    self->cached_tensor_dptr = 0;
    self->cached_tensor_mem = NULL;

    if (self->parser_handle) {
        dlclose(self->parser_handle);
        self->parser_handle = nullptr;
        self->parser_func = nullptr;
        GST_INFO_OBJECT(self, "parser plugin unloaded");
    }

    self->model_loaded = FALSE;

    return TRUE;
}

/** --- INIT --- */
static void gst_magma_infer_init(GstMagmaInfer* self) {
    self->onnx_model_path = NULL;
    self->mxr_model_path = NULL;
    self->inference_interval = 1;
    self->frame_counter = 0;
    self->in_width = 0;
    self->in_height = 0;

    self->d_objects = nullptr;
    self->max_objects = 100;

    self->d_num_det = nullptr;

    self->hip_stream = nullptr;

    self->migraphx_model = nullptr;
    self->model_loaded = FALSE;

    self->parser_plugin_path = NULL;
    self->parser_func_name = g_strdup("magma_parse");
    self->parser_handle = nullptr;
    self->parser_func = nullptr;

    self->confidence_thresh = 0.5f;
    self->nms_thresh = 0.45f;
    self->max_detections = 100;

    self->class_filter = -1; /* -1 = no filter */

    /* DMABuf import cache */
    self->cached_tensor_mem = NULL;
    self->cached_tensor_ext = nullptr;
    self->cached_tensor_dptr = 0;

    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);

    /*self->d_parser_input = 0;

    hipError_t pe = hipMalloc(&self->d_parser_input, output_bytes);
    if (pe != hipSuccess) {
        GST_ERROR_OBJECT(self, "hipMalloc(parser_input %zu) failed", output_bytes);
        // TODO this is such a shit pattern... like "yeah everythings fucked but even though we failed to allocate memory for the parser input buffer, lets just keep going and hope it works out"
    }*/
}

/** --- CAPS NEGOTIATION --- */
static gboolean gst_magma_infer_set_caps(GstBaseTransform* trans, GstCaps* incaps, GstCaps* outcaps) {
    GstMagmaInfer* self = GST_MAGMA_INFER(trans);
    GstVideoInfo info;
    fprintf(stderr, "MAGMA_DBG: mgminfer set_caps()\n");

    if (!gst_video_info_from_caps(&info, incaps)) {
        GST_ERROR_OBJECT(self, "failed to parse incaps");
        return FALSE;
    }

    self->in_width = GST_VIDEO_INFO_WIDTH(&info);
    self->in_height = GST_VIDEO_INFO_HEIGHT(&info);

    if (!ensure_objects_output(self)) {
        GST_ERROR_OBJECT(self, "failed to allocate objects output buffer");
        return FALSE;
    }

    GST_INFO_OBJECT(self, "configured %dx%d %s", self->in_width, self->in_height, GST_VIDEO_INFO_NAME(&info));
    return TRUE;
}

/** --- import a DMABuf → HIP device pointer (must destroy handle after use) --- */
static hipExternalMemory_t import_dmabuf_to_hip(int dmabuf_fd, gsize size, hipDeviceptr_t* d_ptr) {
    int hip_fd = fcntl(dmabuf_fd, F_DUPFD_CLOEXEC, 0);
    if (hip_fd < 0)
        return nullptr;

    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = hip_fd;
    desc.size = size;
    hipExternalMemory_t ext_mem;
    hipError_t err = hipImportExternalMemory(&ext_mem, &desc);
    close(hip_fd);
    if (err != hipSuccess)
        return nullptr;

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
/**
 * @brief Attach a MagmaInferenceMeta with detection results to the buffer.
 *
 * Creates the meta, transfers GPU object count to CPU, and copies
 * MagmaInferObjectGPU entries from the GPU output buffer into a
 * GPtrArray of CPU-side MagmaInferObject for downstream elements.
 *
 * @param self        Inference element
 * @param buf         Target buffer
 * @param num_objects Number of detected objects on GPU
 * @return GST_FLOW_OK on success
 */
static GstFlowReturn attach_inference_meta(GstMagmaInfer* self, GstBuffer* buf, gint num_objects) {
    MagmaInferenceMeta* m = magma_buffer_add_inference_meta(buf, self->in_width, self->in_height);
    if (!m) {
        GST_ERROR_OBJECT(self, "failed to attach inference meta");
        return GST_FLOW_ERROR;
    }

    /* copy preprocessing context from tensor meta (ROI + model dims) */
    MagmaTensorMeta* tmeta = magma_buffer_get_tensor_meta(buf);
    if (tmeta) {
        m->roi_x = tmeta->roi_x;
        m->roi_y = tmeta->roi_y;
        m->roi_w = tmeta->roi_w;
        m->roi_h = tmeta->roi_h;
        m->model_width = tmeta->width;
        m->model_height = tmeta->height;
    }

    m->num_objects = num_objects;

    /* copy parsed objects from GPU to a host-accessible GstMemory */
    if (num_objects > 0) {
        gsize bytes = (gsize)num_objects * sizeof(MagmaInferObjectGPU);
        MagmaInferObjectGPU* host = (MagmaInferObjectGPU*)g_malloc(bytes);
        hipError_t herr = hipMemcpyDtoH(host, self->d_objects, bytes);
        if (herr != hipSuccess) {
            GST_ERROR_OBJECT(self, "hipMemcpyDtoH(objects %zu) failed: %s", bytes, hipGetErrorString(herr));
            g_free(host);
            return GST_FLOW_ERROR;
        }

        /* apply class filter */
        if (self->class_filter >= 0) {
            gint wr = 0;
            for (gint rd = 0; rd < num_objects; rd++) {
                if ((gint)host[rd].class_id == self->class_filter)
                    host[wr++] = host[rd];
            }
            num_objects = wr;
        }

        gsize filtered_bytes = num_objects > 0 ? (gsize)num_objects * sizeof(MagmaInferObjectGPU) : 0;
        m->num_objects = num_objects;
        if (num_objects > 0) {
            m->objects_gpu = gst_memory_new_wrapped(GST_MEMORY_FLAG_READONLY, host, filtered_bytes, 0, filtered_bytes, host, g_free);
        } else {
            g_free(host);
            m->objects_gpu = NULL;
        }
    } else {
        m->objects_gpu = NULL;
    }
    return GST_FLOW_OK;
}

/** --- TRANSFORM --- */
/**
 * @brief Main transform entry point — run inference on the frame.
 *
 * Reads the MagmaTensorMeta (preprocessed tensor) from the input
 * buffer, runs the MIGraphX model, invokes the parser plugin to
 * decode raw output into detection objects, and attaches a
 * MagmaInferenceMeta with the results.
 *
 * @param trans The base transform element
 * @param buf   Input buffer (must have MagmaTensorMeta)
 * @return GST_FLOW_OK on success
 */
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

    /* ensure stream (shared with mgmpreproc — guarantees GPU ordering without CPU sync) */
    if (!self->hip_stream) {
        self->hip_stream = magma_get_shared_hip_stream();
        if (!self->hip_stream) {
            GST_ERROR_OBJECT(self, "magma_get_shared_hip_stream failed");
            return GST_FLOW_ERROR;
        }
    }

    /* MIGraphX path */
    if (self->model_loaded) {
        auto* model = static_cast<MigraphXModel*>(self->migraphx_model);

        if (model->input_lengths.size() == 4 && (int)model->input_lengths[0] == 1 && (int)model->input_lengths[1] == tmeta->channels && (int)model->input_lengths[2] == tmeta->height &&
            (int)model->input_lengths[3] == tmeta->width) {

            /* import tensor DMABuf → HIP (cached — tensor fd is stable across frames) */
            if (tmeta->tensor_mem != self->cached_tensor_mem) {
                if (self->cached_tensor_ext) {
                    (void)hipDestroyExternalMemory(self->cached_tensor_ext);
                    self->cached_tensor_ext = nullptr;
                }
                self->cached_tensor_dptr = 0;
                self->cached_tensor_mem = NULL;

                int tensor_fd = gst_dmabuf_memory_get_fd(tmeta->tensor_mem);
                if (tensor_fd < 0) {
                    GST_ERROR_OBJECT(self, "failed to get tensor DMABuf fd");
                    return GST_FLOW_ERROR;
                }
                gsize tensor_bytes = gst_memory_get_sizes(tmeta->tensor_mem, NULL, NULL);
                hipExternalMemory_t ext = import_dmabuf_to_hip(tensor_fd, tensor_bytes, &self->cached_tensor_dptr);
                close(tensor_fd);
                if (!ext || !self->cached_tensor_dptr) {
                    self->cached_tensor_dptr = 0;
                    GST_ERROR_OBJECT(self, "failed to import tensor DMABuf to HIP");
                    return GST_FLOW_ERROR;
                }
                self->cached_tensor_ext = ext;
                self->cached_tensor_mem = tmeta->tensor_mem;
            }
            hipDeviceptr_t d_tensor = self->cached_tensor_dptr;

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

                auto outputs = model->prog.run_async(eval_args, self->hip_stream);

                if (outputs.empty()) {

                    GST_ERROR_OBJECT(self, "MIGraphX returned no outputs");
                    return GST_FLOW_ERROR;
                }

                auto output_arg = outputs.front();
                auto output_shape = output_arg.get_shape();
                auto* d_output = static_cast<const char*>(output_arg.data());
                GST_LOG_OBJECT(self, "output ptr=%p", (void*)d_output);

                /* ensure MIGraphX eval is fully done before parser touches the output */
                //(void)hipStreamSynchronize(self->hip_stream);

                /* copy output to a fresh parser-owned buffer (MIGraphX internally managed) */
                // gsize output_bytes = output_shape.bytes();

                /* pe = hipMemcpyDtoD(self->d_parser_input, (hipDeviceptr_t)d_output, output_bytes);
                 if (pe != hipSuccess) {
                    GST_ERROR_OBJECT(self, "hipMemcpyDtoD(parser_input) failed");
                    (void)hipFree(self->d_parser_input);
                    return GST_FLOW_ERROR;
                }
                // Sync all GPU operations before parser touches the data
                //(void)hipStreamSynchronize(self->hip_stream);
                */
                /**
#ifdef __MGM_TRACE_HIP__
                roctxRangePush("mgminfer: magma_infer_transform_ip|DeviceSynchronize");
#endif
                hipError_t sync_err = hipDeviceSynchronize();
                GST_DEBUG_OBJECT(self, "MAGMA_DBG: hipDeviceSynchronize after eval = %s\n", hipGetErrorString(sync_err));
#ifdef __MGM_TRACE_HIP__
                roctxRangePop();
#endif
                */
                /* --- parser dispatch --- */
                if (self->parser_func) {
                    auto lengths = output_shape.lengths();
                    int ndim = (int)lengths.size();
                    std::vector<int64_t> host_lengths(lengths.begin(), lengths.end());

                    {
                        std::string dims;
                        for (auto l : lengths)
                            dims += std::to_string(l) + " ";
                        GST_INFO_OBJECT(self, "MIGraphX output shape: %s(%d dims, %zu bytes)", dims.c_str(), ndim, output_shape.bytes());
                    }

                    int net_w = self->in_width;
                    int net_h = self->in_height;
                    {
                        auto* m = static_cast<MigraphXModel*>(self->migraphx_model);
                        if (m) {
                            net_w = m->model_width;
                            net_h = m->model_height;
                        }
                    }

                    MagmaParseParams params{};

                    // params.d_raw_output = (const void*)d_parser_input;
                    params.d_raw_output = (const void*)d_output;
                    params.output_shape = host_lengths.data();
                    params.num_dims = ndim;
                    params.net_width = net_w;
                    params.net_height = net_h;
                    params.confidence_thresh = self->confidence_thresh;
                    params.nms_thresh = self->nms_thresh;
                    params.max_detections = (int)self->max_detections;
                    params.d_objects = self->d_objects;
                    params.d_num_detected = self->d_num_det;
                    params.stream = (void*)self->hip_stream;

                    // Test: just do a memset to verify GPU access before parser
                    GST_INFO_OBJECT(
                        self, "pre-parser: memset d_objects=%p size=%zu stream=%p", (void*)self->d_objects, (size_t)(self->max_objects * sizeof(MagmaInferObjectGPU)), (void*)self->hip_stream);
                    GST_INFO_OBJECT(self, "pre-parser: calling parser_func at %p", (void*)self->parser_func);

                    int pret = self->parser_func(&params);
                    //(void)hipFree(d_parser_input);

                    /* Ensure GPU writes to objects buffer are visible */
                    (void)hipStreamSynchronize(self->hip_stream);

                    /* Debug: read first object back to verify GPU→CPU data path */
                    {
                        MagmaInferObjectGPU dbg[4];
                        (void)hipMemcpyDtoH(dbg, self->d_objects, sizeof(dbg));
                        // send it to
                        GST_DEBUG_OBJECT(self, "Object[0] class_id=%d conf=%f x=%f y=%f w=%f h=%f", dbg[0].class_id, dbg[0].confidence, dbg[0].x, dbg[0].y, dbg[0].width, dbg[0].height);
                    }

                    if (pret != 0) {
                        GST_ERROR_OBJECT(self, "parser failed with code %d", pret);

                        return GST_FLOW_ERROR;
                    }

                    int num_detected = 0;
                    (void)hipMemcpyDtoH(&num_detected, self->d_num_det, sizeof(int));

                    GST_LOG_OBJECT(self, "frame %u — parser produced %d objects", self->frame_counter, num_detected);

                    return attach_inference_meta(self, buf, num_detected);
                } else {
                    /* --- fallback: legacy hardcoded path (no parser) --- */
                    gsize output_bytes = output_shape.bytes();
                    gsize copy_bytes = std::min(output_bytes, self->max_objects * sizeof(MagmaInferObjectGPU));

                    hipError_t herr = hipMemcpyDtoD(self->d_objects, (hipDeviceptr_t)d_output, copy_bytes);
                    if (herr != hipSuccess) {
                        GST_ERROR_OBJECT(self, "hipMemcpyDtoD failed: %s", hipGetErrorString(herr));

                        return GST_FLOW_ERROR;
                    }

                    float confidence = 0.0f;
                    (void)hipMemcpyDtoH(&confidence, self->d_objects, sizeof(float));
                    MagmaInferObjectGPU objs[1] = {};
                    objs[0].class_id = 1;
                    objs[0].confidence = confidence;
                    objs[0].x = 0.0f;
                    objs[0].y = 0.0f;
                    objs[0].width = 1.0f;
                    objs[0].height = 1.0f;
                    (void)hipMemcpyHtoD(self->d_objects, objs, sizeof(objs));

                    GST_LOG_OBJECT(self, "frame %u — MIGraphX inference, confidence=%.4f", self->frame_counter, confidence);
                    return attach_inference_meta(self, buf, 1);
                }
            } catch (const std::exception& e) {
                GST_ERROR_OBJECT(self, "MIGraphX eval failed: %s", e.what());

                return GST_FLOW_ERROR;
            }
        } else {
            GST_ERROR_OBJECT(self, "tensor %dx%dx%d does not match model (expected %zu dims)", tmeta->channels, tmeta->height, tmeta->width, model->input_lengths.size());
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

    g_object_class_install_property(gobject_class, PROP_ONNX_MODEL_PATH, g_param_spec_string("model-onnx-file", "The Onnx Model path", "Path to the onnx model file", NULL, G_PARAM_READWRITE));
    g_object_class_install_property(
        gobject_class,
        PROP_MXR_MODEL_PATH,
        g_param_spec_string("model-mxr-file", "The MXR Model path", "Path to the mxr model file(needs model-onnx-file, if there path is either invalid or outdated)", NULL, G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_INFERENCE_INTERVAL, g_param_spec_uint("inference-interval", "Inference interval", "Run inference every N frames (1 = every frame)", 1, G_MAXUINT32, 1, G_PARAM_READWRITE));

    g_object_class_install_property(gobject_class, PROP_PARSER_PLUGIN, g_param_spec_string("parser-plugin", "Parser plugin", "Path to parser .so (dlopen)", NULL, G_PARAM_READWRITE));

    g_object_class_install_property(gobject_class, PROP_PARSER_FUNC, g_param_spec_string("parser-function", "Parser function", "Symbol name in parser .so", "magma_parse", G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_CONFIDENCE_THRESH, g_param_spec_float("confidence-threshold", "Confidence threshold", "Minimum confidence to keep a detection", 0.0f, 1.0f, 0.5f, G_PARAM_READWRITE));

    g_object_class_install_property(gobject_class, PROP_NMS_THRESH, g_param_spec_float("nms-threshold", "NMS threshold", "IoU threshold for NMS suppression", 0.0f, 1.0f, 0.45f, G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_MAX_DETECTIONS, g_param_spec_uint("max-detections", "Max detections", "Maximum number of output objects per frame", 1, 10000, 100, G_PARAM_READWRITE));

    g_object_class_install_property(
        gobject_class, PROP_CLASS_FILTER, g_param_spec_int("class-filter", "Class filter", "Only keep detections of this class (-1 = all)", -1, 1000, -1, G_PARAM_READWRITE));

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    gst_element_class_set_static_metadata(element_class, "Magma Inference", "Meta/Inference/Video", "Attaches GPU inference results as buffer metadata (MIGraphX + parser plugins)", "Magma");

    trans->set_caps = gst_magma_infer_set_caps;
    trans->transform_ip = gst_magma_infer_transform_ip;
    trans->start = gst_magma_infer_start;
    trans->stop = gst_magma_infer_stop;
    trans->decide_allocation = gst_magma_infer_decide_allocation;

    magma_inference_meta_get_info();
    magma_tensor_meta_get_info();

    GST_DEBUG_CATEGORY_INIT(magma_infer_debug, "magma_infer", 0, "Magma Inference Plugin");
}

/** --- DECIDE_ALLOCATION: increase buffer pool min-buffers for GPU pipeline cushion --- */
static gboolean gst_magma_infer_decide_allocation(GstBaseTransform* trans, GstQuery* query) {
    GstBufferPool* pool = NULL;
    GstStructure* config;
    guint size, min_bufs, max_bufs;

    if (!GST_BASE_TRANSFORM_CLASS(gst_magma_infer_parent_class)->decide_allocation(trans, query))
        return FALSE;

    if (gst_query_get_n_allocation_pools(query) > 0) {
        gst_query_parse_nth_allocation_pool(query, 0, &pool, &size, &min_bufs, &max_bufs);

        if (pool) {
            config = gst_buffer_pool_get_config(pool);

            if (min_bufs < 12)
                min_bufs = 12;

            gst_buffer_pool_config_set_params(config, NULL, size, min_bufs, max_bufs);
            gst_buffer_pool_set_config(pool, config);
            gst_object_unref(pool);
        }
    }

    return TRUE;
}

/** --- PLUGIN REGISTRATION --- */
static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "mgminfer", GST_RANK_NONE, GST_TYPE_MAGMA_INFER);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mgminfer, "Magma Inference Plugin", plugin_init, "0.1.0", "LGPL", "magma", "https://imeguras.eu.org")
