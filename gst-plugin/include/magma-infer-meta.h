#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

/* ---------- GPU-side result structures ---------- */

typedef struct {
    guint class_id;
    gfloat confidence;
    gfloat x, y, width, height;
} MagmaInferObjectGPU;

typedef struct _MagmaInferObject MagmaInferObject;
struct _MagmaInferObject {
    guint class_id;
    gchar* label;
    gfloat confidence;
    gfloat x, y, width, height;
};

/* ---------- inference metadata ---------- */

typedef struct _MagmaInferenceMeta MagmaInferenceMeta;

struct _MagmaInferenceMeta {
    GstMeta meta;

    guint source_width;
    guint source_height;

    guint num_objects;
    GstMemory* objects_gpu;

    GPtrArray* output_tensors;
};

#define MAGMA_INFERENCE_META_API_TYPE (magma_inference_meta_api_get_type())
#define magma_buffer_get_inference_meta(b) ((MagmaInferenceMeta*)gst_buffer_get_meta((b), MAGMA_INFERENCE_META_API_TYPE))

GType magma_inference_meta_api_get_type(void);
const GstMetaInfo* magma_inference_meta_get_info(void);
MagmaInferenceMeta* magma_buffer_add_inference_meta(GstBuffer* buffer, guint source_width, guint source_height);

MagmaInferObject* magma_infer_object_new(guint class_id, const gchar* label, gfloat confidence, gfloat x, gfloat y, gfloat w, gfloat h);
void magma_infer_object_free(MagmaInferObject* obj);

G_END_DECLS
