#include "mgmserialize.hpp"
#include "magma-infer-meta.h"

#include <cstring>
#include <string>

GST_DEBUG_CATEGORY_STATIC(magma_serialize_debug);
#define GST_CAT_DEFAULT magma_serialize_debug

enum { PROP_0, PROP_FORMAT };

G_DEFINE_TYPE(GstMagmaSerialize, gst_magma_serialize, GST_TYPE_BASE_TRANSFORM)

/* ---------- pad templates ---------- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("application/x-magma-msg"));

/* ---------- JSON serialization ---------- */
static gchar* serialize_to_json(MagmaInferenceMeta* m, int* out_len) {
    GString* s = g_string_new("");

    if (!m || m->num_objects == 0) {
        g_string_append(s, "{\"objects\":[]}");
        *out_len = (int)s->len;
        return g_string_free(s, FALSE);
    }

    /* read objects_gpu DMABuf back to CPU */
    gsize bytes = (gsize)m->num_objects * sizeof(MagmaInferObjectGPU);
    MagmaInferObjectGPU* host = (MagmaInferObjectGPU*)g_malloc(bytes);

    GstMapFlags flags = GST_MAP_READ;
    GstMapInfo info;
    if (gst_memory_map(m->objects_gpu, &info, flags)) {
        memcpy(host, info.data, bytes);
        gst_memory_unmap(m->objects_gpu, &info);
    } else {
        /* fallback: send empty result */
        g_free(host);
        g_string_append(s, "{\"objects\":[]}");
        *out_len = (int)s->len;
        return g_string_free(s, FALSE);
    }

    g_string_append_printf(s,
                           "{\"source_width\":%u,\"source_height\":%u,"
                           "\"timestamp_ns\":%" G_GUINT64_FORMAT ","
                           "\"objects\":[",
                           m->source_width,
                           m->source_height,
                           (guint64)g_get_real_time() * 1000);

    for (guint i = 0; i < m->num_objects; i++) {
        if (i > 0)
            g_string_append(s, ",");
        g_string_append_printf(s,
                               "{\"class_id\":%u,\"confidence\":%.6f,"
                               "\"bbox\":{\"x\":%.6f,\"y\":%.6f,\"w\":%.6f,\"h\":%.6f}}",
                               host[i].class_id,
                               host[i].confidence,
                               host[i].x,
                               host[i].y,
                               host[i].width,
                               host[i].height);
    }
    g_string_append(s, "]}");
    g_free(host);

    *out_len = (int)s->len;
    return g_string_free(s, FALSE);
}

#ifdef HAVE_PROTOBUF
#include "magma_msg.pb.h"

static gchar* serialize_to_protobuf(MagmaInferenceMeta* m, int* out_len) {
    magma::FrameResult result;
    result.set_source_width(m->source_width);
    result.set_source_height(m->source_height);
    result.set_timestamp_ns((guint64)g_get_real_time() * 1000);

    if (m && m->num_objects > 0) {
        gsize bytes = (gsize)m->num_objects * sizeof(MagmaInferObjectGPU);
        MagmaInferObjectGPU* host = (MagmaInferObjectGPU*)g_malloc(bytes);

        GstMapFlags flags = GST_MAP_READ;
        GstMapInfo info;
        if (gst_memory_map(m->objects_gpu, &info, flags)) {
            memcpy(host, info.data, bytes);
            gst_memory_unmap(m->objects_gpu, &info);
        } else {
            g_free(host);
            std::string data = result.SerializeAsString();
            *out_len = (int)data.size();
            return g_strndup(data.data(), data.size());
        }

        for (guint i = 0; i < m->num_objects; i++) {
            magma::Detection* d = result.add_objects();
            d->set_class_id(host[i].class_id);
            d->set_confidence(host[i].confidence);
            d->set_x(host[i].x);
            d->set_y(host[i].y);
            d->set_width(host[i].width);
            d->set_height(host[i].height);
        }
        g_free(host);
    }

    std::string data = result.SerializeAsString();
    *out_len = (int)data.size();
    return g_strndup(data.data(), data.size());
}
#endif

/* ---------- transform ---------- */
static GstFlowReturn gst_magma_serialize_transform(GstBaseTransform* trans, GstBuffer* inbuf, GstBuffer* outbuf) {
    GstMagmaSerialize* self = GST_MAGMA_SERIALIZE(trans);

    MagmaInferenceMeta* m = magma_buffer_get_inference_meta(inbuf);

    int len = 0;
    gchar* serialized = NULL;

    if (g_strcmp0(self->format, "protobuf") == 0) {
#ifdef HAVE_PROTOBUF
        serialized = serialize_to_protobuf(m, &len);
#else
        GST_ERROR_OBJECT(self, "protobuf not compiled in");
        return GST_FLOW_NOT_SUPPORTED;
#endif
    } else {
        serialized = serialize_to_json(m, &len);
    }

    GstMemory* mem = gst_memory_new_wrapped(GST_MEMORY_FLAG_READONLY, serialized, len, 0, len, serialized, g_free);
    gst_buffer_remove_all_memory(outbuf);
    gst_buffer_append_memory(outbuf, mem);

    return GST_FLOW_OK;
}

/* ---------- transform caps ---------- */
static GstCaps* gst_magma_serialize_transform_caps(GstBaseTransform* trans, GstPadDirection direction, GstCaps* caps, GstCaps* filter) {
    GstCaps* result;
    if (direction == GST_PAD_SINK) {
        result = gst_caps_new_empty_simple("application/x-magma-msg");
    } else {
        result = gst_caps_from_string("video/x-raw(memory:DMABuf),format=(string)NV12");
    }
    if (filter) {
        GstCaps* tmp = gst_caps_intersect_full(result, filter, GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref(result);
        result = tmp;
    }
    return result;
}

static gboolean gst_magma_serialize_set_caps(GstBaseTransform* trans, GstCaps* incaps, GstCaps* outcaps) {
    GstMagmaSerialize* self = GST_MAGMA_SERIALIZE(trans);

    if (g_strcmp0(self->format, "protobuf") == 0) {
#ifndef HAVE_PROTOBUF
        GST_ERROR_OBJECT(self, "protobuf format requires protobuf support at build time");
        return FALSE;
#endif
    }

    return TRUE;
}

static gboolean gst_magma_serialize_transform_size(GstBaseTransform* trans, GstPadDirection direction, GstCaps* caps, gsize size, GstCaps* othercaps, gsize* othersize) {
    /* output size is 0 — we allocate in transform() */
    *othersize = 0;
    return TRUE;
}

/* ---------- properties ---------- */
static void gst_magma_serialize_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
    GstMagmaSerialize* self = GST_MAGMA_SERIALIZE(object);
    switch (prop_id) {
    case PROP_FORMAT:
        g_free(self->format);
        self->format = g_value_dup_string(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_magma_serialize_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
    GstMagmaSerialize* self = GST_MAGMA_SERIALIZE(object);
    switch (prop_id) {
    case PROP_FORMAT:
        g_value_set_string(value, self->format);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_magma_serialize_finalize(GObject* object) {
    GstMagmaSerialize* self = GST_MAGMA_SERIALIZE(object);
    g_free(self->format);
    G_OBJECT_CLASS(gst_magma_serialize_parent_class)->finalize(object);
}

static void gst_magma_serialize_init(GstMagmaSerialize* self) {
    self->format = g_strdup("json");
}

/* ---------- class init ---------- */
static void gst_magma_serialize_class_init(GstMagmaSerializeClass* klass) {
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass* trans = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = gst_magma_serialize_set_property;
    gobject_class->get_property = gst_magma_serialize_get_property;
    gobject_class->finalize = gst_magma_serialize_finalize;

    g_object_class_install_property(gobject_class, PROP_FORMAT, g_param_spec_string("format", "Format", "Serialization format: json or protobuf", "json", G_PARAM_READWRITE));

    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    gst_element_class_set_static_metadata(element_class, "Magma Serialize", "Filter/Converter", "Serializes MagmaInferenceMeta to JSON or protobuf", "Magma");

    trans->transform_caps = gst_magma_serialize_transform_caps;
    trans->transform_size = gst_magma_serialize_transform_size;
    trans->set_caps = gst_magma_serialize_set_caps;
    trans->transform = gst_magma_serialize_transform;

    magma_inference_meta_get_info();

    GST_DEBUG_CATEGORY_INIT(magma_serialize_debug, "magma_serialize", 0, "Magma Serialize Plugin");
}

static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "mgmserialize", GST_RANK_NONE, GST_TYPE_MAGMA_SERIALIZE);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mgmserialize, "Magma Serialize Plugin", plugin_init, "0.1.0", "LGPL", "magma", "https://imeguras.eu.org")
