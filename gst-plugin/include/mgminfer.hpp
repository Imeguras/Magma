#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>

G_BEGIN_DECLS

/* ---------- caps ---------- */

#define INFER_SINK_CAPS "video/x-raw(memory:DMABuf),format=(string)NV12"
#define INFER_SRC_CAPS "video/x-raw(memory:DMABuf),format=(string)NV12"

/* ---------- GPU-side result structures ---------- */

/**
 * MagmaInferObjectGPU:
 * Plain-old-data struct; designed to live in GPU memory (DMABuf) as a
 * packed array so that GPU kernels can write results directly.
 */
typedef struct {
    guint class_id;
    gfloat confidence;
    gfloat x, y, width, height; /* normalised bounding box (0..1) */
} MagmaInferObjectGPU;

/**
 * MagmaInferObject:
 * CPU-side convenience struct (e.g. for display).  Not stored on GPU.
 */
typedef struct _MagmaInferObject MagmaInferObject;
struct _MagmaInferObject {
    guint class_id;
    gchar* label; /* owned string, may be NULL */
    gfloat confidence;
    gfloat x, y, width, height;
};

/* ---------- inference metadata (attached to GstBuffer) ---------- */

typedef struct _MagmaInferenceMeta MagmaInferenceMeta;

/**
 * MagmaInferenceMeta:
 *
 * Attached to every output buffer of mgminfer.
 * The actual detection results live in a DMABuf on the GPU
 * (`objects_gpu` memory) ; the CPU-side meta only holds a reference
 * and a count.  Downstream can map the DMABuf for CPU read-back
 * when needed, or keep everything on GPU.
 */
struct _MagmaInferenceMeta {
    GstMeta meta;

    guint source_width;
    guint source_height;

    /* --- GPU-resident results --- */

    guint num_objects;      /* number of valid entries             */
    GstMemory* objects_gpu; /* DMABuf with MagmaInferObjectGPU[]   */

    /* --- Optional: raw output tensors (GPU memories) --- */
    GPtrArray* output_tensors; /* of GstMemory* (each a GPU DMABuf)   */
};

/* --- metadata API --- */

#define MAGMA_INFERENCE_META_API_TYPE (magma_inference_meta_api_get_type())
#define magma_buffer_get_inference_meta(b) ((MagmaInferenceMeta*)gst_buffer_get_meta((b), MAGMA_INFERENCE_META_API_TYPE))

GType magma_inference_meta_api_get_type(void);
const GstMetaInfo* magma_inference_meta_get_info(void);
MagmaInferenceMeta* magma_buffer_add_inference_meta(GstBuffer* buffer, guint source_width, guint source_height);

/* CPU-side convenience helpers */
MagmaInferObject* magma_infer_object_new(guint class_id, const gchar* label, gfloat confidence, gfloat x, gfloat y, gfloat w, gfloat h);
void magma_infer_object_free(MagmaInferObject* obj);

/* ---------- element type ---------- */

#define GST_TYPE_MAGMA_INFER (gst_magma_infer_get_type())

G_DECLARE_FINAL_TYPE(GstMagmaInfer, gst_magma_infer, GST, MAGMA_INFER, GstBaseTransform)

struct _GstMagmaInfer {
    GstBaseTransform parent;

    gchar* model_path;

    guint inference_interval;
    guint frame_counter;

    gint in_width;
    gint in_height;

    // GBM for output objects DMABuf
    int drm_fd;
    struct gbm_device* gbm;
    gboolean gbm_ready;

    // Output objects DMABuf (GPU-resident detections)
    int objects_dmabuf_fd;
    hipExternalMemory_t objects_ext_mem;
    MagmaInferObjectGPU* d_objects;
    GstMemory* objects_mem;
    guint max_objects;

    // HIP stream
    hipStream_t hip_stream;

    // Compiled dummy kernel
    hipModule_t kernel_module;
    hipFunction_t kernel_dummy;
    gboolean kernel_ready;
};

G_END_DECLS
