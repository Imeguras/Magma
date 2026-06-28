#include "mgminfer.hpp"

/* ---------- MagmaInferObject CPU convenience ---------- */

MagmaInferObject* magma_infer_object_new(guint class_id, const gchar* label, gfloat confidence, gfloat x, gfloat y, gfloat w, gfloat h) {
    MagmaInferObject* obj = g_slice_new0(MagmaInferObject);
    obj->class_id = class_id;
    obj->label = g_strdup(label);
    obj->confidence = confidence;
    obj->x = x;
    obj->y = y;
    obj->width = w;
    obj->height = h;
    return obj;
}

void magma_infer_object_free(MagmaInferObject* obj) {
    if (!obj)
        return;
    g_free(obj->label);
    g_slice_free(MagmaInferObject, obj);
}

/* ---------- MagmaInferenceMeta GstMeta ---------- */

static gsize _magma_inference_meta_quark = 0;

GType magma_inference_meta_api_get_type(void) {
    if (g_once_init_enter(&_magma_inference_meta_quark)) {
        GQuark q = g_quark_from_static_string("MagmaInferenceMetaAPI");
        g_once_init_leave(&_magma_inference_meta_quark, q);
    }
    return (GType)_magma_inference_meta_quark;
}

static gboolean magma_inference_meta_init(GstMeta* meta, gpointer params, GstBuffer* buffer) {
    MagmaInferenceMeta* m = (MagmaInferenceMeta*)meta;
    m->source_width = 0;
    m->source_height = 0;
    m->num_objects = 0;
    m->objects_gpu = NULL;
    m->output_tensors = NULL;
    return TRUE;
}

static void magma_inference_meta_free(GstMeta* meta, GstBuffer* buffer) {
    MagmaInferenceMeta* m = (MagmaInferenceMeta*)meta;
    if (m->objects_gpu) {
        gst_memory_unref(m->objects_gpu);
        m->objects_gpu = NULL;
    }
    if (m->output_tensors) {
        for (guint i = 0; i < m->output_tensors->len; i++) {
            GstMemory* mem = (GstMemory*)g_ptr_array_index(m->output_tensors, i);
            gst_memory_unref(mem);
        }
        g_ptr_array_unref(m->output_tensors);
        m->output_tensors = NULL;
    }
}

static gboolean magma_inference_meta_transform(GstBuffer* transbuf, GstMeta* meta, GstBuffer* buffer, GQuark type, gpointer data) {
    MagmaInferenceMeta* src = (MagmaInferenceMeta*)meta;
    if (type != 0)
        return FALSE;
    MagmaInferenceMeta* dest = magma_buffer_add_inference_meta(transbuf, src->source_width, src->source_height);
    if (!dest)
        return FALSE;
    dest->num_objects = src->num_objects;
    if (src->objects_gpu)
        dest->objects_gpu = gst_memory_ref(src->objects_gpu);
    if (src->output_tensors) {
        dest->output_tensors = g_ptr_array_new_with_free_func((GDestroyNotify)gst_memory_unref);
        for (guint i = 0; i < src->output_tensors->len; i++) {
            GstMemory* mem = (GstMemory*)g_ptr_array_index(src->output_tensors, i);
            g_ptr_array_add(dest->output_tensors, gst_memory_ref(mem));
        }
    }
    return TRUE;
}

static gsize _magma_inference_meta_info = 0;

const GstMetaInfo* magma_inference_meta_get_info(void) {
    if (g_once_init_enter(&_magma_inference_meta_info)) {
        const GstMetaInfo* info =
            gst_meta_register(MAGMA_INFERENCE_META_API_TYPE, "MagmaInferenceMeta", sizeof(MagmaInferenceMeta), magma_inference_meta_init, magma_inference_meta_free, magma_inference_meta_transform);
        g_once_init_leave(&_magma_inference_meta_info, (gsize)info);
    }
    return (const GstMetaInfo*)_magma_inference_meta_info;
}

MagmaInferenceMeta* magma_buffer_add_inference_meta(GstBuffer* buffer, guint source_width, guint source_height) {
    MagmaInferenceMeta* m = (MagmaInferenceMeta*)gst_buffer_add_meta(buffer, magma_inference_meta_get_info(), NULL);
    if (m) {
        m->source_width = source_width;
        m->source_height = source_height;
    }
    return m;
}
