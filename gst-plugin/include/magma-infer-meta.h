#pragma once

/**
 * @file magma-infer-meta.h
 * @brief Inference metadata types — detection objects and per-buffer metadata.
 *
 * Defines GPU-resident and CPU-resident detection result structures,
 * plus the GstMeta type that carries them through the pipeline.
 */

#include <gst/gst.h>

G_BEGIN_DECLS

/**
 * @brief GPU-resident detection object (packed, 4-byte aligned).
 *
 * Stored in a DMABuf-backed GstMemory for zero-copy transfer.
 */
typedef struct {
    guint class_id;
    gfloat confidence;
    gfloat x, y, width, height;
} MagmaInferObjectGPU;

/**
 * @brief CPU-resident detection object with string label.
 */
typedef struct _MagmaInferObject MagmaInferObject;
struct _MagmaInferObject {
    guint class_id;
    gchar* label;
    gfloat confidence;
    gfloat x, y, width, height;
};

/**
 * @brief GStreamer metadata carrying per-frame inference results.
 *
 * Attached to output buffers of mgminfer. Contains GPU-resident
 * detection objects -- the actual tensor outputs (raw model output),
 * optional segmentation masks, and optional anomaly detection data.
 */
typedef struct _MagmaInferenceMeta MagmaInferenceMeta;

struct _MagmaInferenceMeta {
    GstMeta meta;

    /** Source frame dimensions (for coordinate mapping) */
    guint source_width;
    guint source_height;

    /** ROI within the source frame that was fed to the model */
    gint roi_x, roi_y, roi_w, roi_h;

    /** Model input dimensions */
    gint model_width, model_height;

    /** Number of detected objects */
    guint num_objects;

    /** GPU DMABuf containing MagmaInferObjectGPU array */
    GstMemory* objects_gpu;

    /** Raw model output tensors (GPU DMABuf array) */
    GPtrArray* output_tensors;

    /** Optional segmentation masks (GPU DMABuf — float[num_masks][mask_h][mask_w]) */
    guint       num_masks;
    GstMemory*  masks_gpu;
    guint       mask_width, mask_height;

    /** Optional anomaly detection output */
    gboolean    has_anomaly;
    float       anomaly_score;
    GstMemory*  anomaly_heatmap;
};

#define MAGMA_INFERENCE_META_API_TYPE (magma_inference_meta_api_get_type())
#define magma_buffer_get_inference_meta(b) ((MagmaInferenceMeta*)gst_buffer_get_meta((b), MAGMA_INFERENCE_META_API_TYPE))

/** @brief Query the GType for MagmaInferenceMeta */
GType magma_inference_meta_api_get_type(void);

/** @brief Get the GstMetaInfo for MagmaInferenceMeta */
const GstMetaInfo* magma_inference_meta_get_info(void);

/**
 * @brief Attach a MagmaInferenceMeta to a GstBuffer.
 * @param buffer        Target GstBuffer
 * @param source_width  Source frame width
 * @param source_height Source frame height
 * @return Pointer to the newly-attached meta, or NULL on failure
 */
MagmaInferenceMeta* magma_buffer_add_inference_meta(GstBuffer* buffer, guint source_width, guint source_height);

/**
 * @brief Create a new CPU-side MagmaInferObject.
 * @param class_id   Class identifier
 * @param label      Human-readable class label (copied internally)
 * @param confidence Detection confidence
 * @param x          Center X (normalized 0..1)
 * @param y          Center Y (normalized 0..1)
 * @param w          Width (normalized 0..1)
 * @param h          Height (normalized 0..1)
 * @return Newly allocated MagmaInferObject (must be freed with magma_infer_object_free)
 */
MagmaInferObject* magma_infer_object_new(guint class_id, const gchar* label, gfloat confidence,
                                          gfloat x, gfloat y, gfloat w, gfloat h);

/** @brief Free a MagmaInferObject allocated by magma_infer_object_new */
void magma_infer_object_free(MagmaInferObject* obj);

G_END_DECLS
