#include "magma-meta.h"

static gsize _magma_tensor_meta_quark = 0;

GType magma_tensor_meta_api_get_type(void) {
    if (g_once_init_enter(&_magma_tensor_meta_quark)) {
        const gchar* tags[] = {NULL};
        GType t = gst_meta_api_type_register("MagmaTensorMetaAPI", tags);
        g_once_init_leave(&_magma_tensor_meta_quark, (gsize)t);
    }
    return (GType)_magma_tensor_meta_quark;
}

static gboolean magma_tensor_meta_init(GstMeta* meta, gpointer params, GstBuffer* buffer) {
    MagmaTensorMeta* m = (MagmaTensorMeta*)meta;
    m->tensor_mem = NULL;
    m->width = 0;
    m->height = 0;
    m->channels = 0;
    return TRUE;
}

static void magma_tensor_meta_free(GstMeta* meta, GstBuffer* buffer) {
    MagmaTensorMeta* m = (MagmaTensorMeta*)meta;
    if (m->tensor_mem) {
        gst_memory_unref(m->tensor_mem);
        m->tensor_mem = NULL;
    }
}

static gboolean magma_tensor_meta_transform(GstBuffer* transbuf, GstMeta* meta, GstBuffer* buffer, GQuark type, gpointer data) {
    if (type != 0)
        return FALSE;
    MagmaTensorMeta* src = (MagmaTensorMeta*)meta;
    MagmaTensorMeta* dest;
    dest = magma_buffer_add_tensor_meta(transbuf, src->tensor_mem, src->width, src->height, src->channels);
    return dest != NULL;
}

static gsize _magma_tensor_meta_info = 0;

const GstMetaInfo* magma_tensor_meta_get_info(void) {
    if (g_once_init_enter(&_magma_tensor_meta_info)) {
        const GstMetaInfo* info = gst_meta_register(MAGMA_TENSOR_META_API_TYPE, "MagmaTensorMeta", sizeof(MagmaTensorMeta), magma_tensor_meta_init, magma_tensor_meta_free, magma_tensor_meta_transform);
        if (!info)
            info = gst_meta_get_info("MagmaTensorMeta");
        g_once_init_leave(&_magma_tensor_meta_info, (gsize)info);
    }
    return (const GstMetaInfo*)_magma_tensor_meta_info;
}

MagmaTensorMeta* magma_buffer_add_tensor_meta(GstBuffer* buffer, GstMemory* tensor_mem, gint width, gint height, gint channels) {
    MagmaTensorMeta* m = (MagmaTensorMeta*)gst_buffer_add_meta(buffer, magma_tensor_meta_get_info(), NULL);
    if (m) {
        m->tensor_mem = tensor_mem ? gst_memory_ref(tensor_mem) : NULL;
        m->width = width;
        m->height = height;
        m->channels = channels;
    }
    return m;
}
