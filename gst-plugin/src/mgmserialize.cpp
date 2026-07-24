#include "mgmserialize.hpp"
#include "magma-infer-meta.h"

#include <cstring>
#include <string>
#include <vector>

GST_DEBUG_CATEGORY_STATIC(magma_serialize_debug);
#define GST_CAT_DEFAULT magma_serialize_debug

enum { PROP_0, PROP_FORMAT };

G_DEFINE_TYPE(GstMagmaSerialize, gst_magma_serialize, GST_TYPE_BASE_TRANSFORM)

/* ---------- pad templates ---------- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS("application/x-magma-msg"));

/* ──────────────────────────────────────────────
 *  Modular serializer interface
 * ────────────────────────────────────────────── */

typedef void  (*SerializeFunc)(MagmaInferenceMeta* meta, GString* out);

typedef struct {
    const char*  key;        /* JSON key, e.g. "detections" */
    gboolean    (*probe)(MagmaInferenceMeta* meta);   /* check if data exists */
    SerializeFunc to_json;                             /* append JSON fragment */
} SerializerEntry;

/* ---------- helpers ---------- */

static void append_float(GString* s, double val) {
    char buf[64];
    g_ascii_dtostr(buf, sizeof(buf), val);
    g_string_append(s, buf);
}

static gboolean map_gpu_memory(GstMemory* mem, gsize bytes, void* dst) {
    GstMapInfo info;
    if (gst_memory_map(mem, &info, GST_MAP_READ)) {
        memcpy(dst, info.data, bytes);
        gst_memory_unmap(mem, &info);
        return TRUE;
    }
    return FALSE;
}

#define GPU_MAP(mem, vec) map_gpu_memory((mem), (vec).size() * sizeof((vec)[0]), (vec).data())

/* ──────────────────────────────────────────────
 *  Detections serializer
 * ────────────────────────────────────────────── */

static gboolean probe_detections(MagmaInferenceMeta* m) {
    return m && m->num_objects > 0 && m->objects_gpu;
}

static void serialize_detections_json(MagmaInferenceMeta* m, GString* s) {
    gsize n = (gsize)m->num_objects;
    std::vector<MagmaInferObjectGPU> host(n);
    if (!GPU_MAP(m->objects_gpu, host)) {
        g_string_append(s, "\"detections\":[]");
        return;
    }
    g_string_append(s, "\"detections\":[");
    for (guint i = 0; i < m->num_objects; i++) {
        if (i > 0) g_string_append(s, ",");
        g_string_append_printf(s,
            "{\"class_id\":%u,\"confidence\":", host[i].class_id);
        append_float(s, host[i].confidence);
        g_string_append(s, ",\"bbox\":{\"x\":");
        append_float(s, host[i].x);
        g_string_append(s, ",\"y\":");
        append_float(s, host[i].y);
        g_string_append(s, ",\"w\":");
        append_float(s, host[i].width);
        g_string_append(s, ",\"h\":");
        append_float(s, host[i].height);
        g_string_append(s, "}}");
    }
    g_string_append(s, "]");
}

/* ──────────────────────────────────────────────
 *  Segmentation masks serializer
 * ────────────────────────────────────────────── */

static gboolean probe_masks(MagmaInferenceMeta* m) {
    return m && m->num_masks > 0 && m->masks_gpu && m->mask_width > 0 && m->mask_height > 0;
}

static void serialize_masks_json(MagmaInferenceMeta* m, GString* s) {
    gsize n = (gsize)m->num_masks * m->mask_width * m->mask_height;
    std::vector<float> host(n);
    if (!GPU_MAP(m->masks_gpu, host)) {
        g_string_append(s, "\"masks\":[]");
        return;
    }

    g_string_append_printf(s, "\"masks\":{\"num_masks\":%u,\"mask_width\":%u,\"mask_height\":%u,\"data\":[",
                           m->num_masks, m->mask_width, m->mask_height);

    gsize total = n;
    guint max_pixels = 64;
    if (total > max_pixels) total = max_pixels;
    for (gsize i = 0; i < total; i++) {
        if (i > 0) g_string_append(s, ",");
        g_string_append_printf(s, "%.6f", host[i]);
    }
    g_string_append(s, "]}");
}

/* ──────────────────────────────────────────────
 *  Anomaly detection serializer
 * ────────────────────────────────────────────── */

static gboolean probe_anomaly(MagmaInferenceMeta* m) {
    return m && m->has_anomaly;
}

static void serialize_anomaly_json(MagmaInferenceMeta* m, GString* s) {
    g_string_append(s, "\"anomaly\":{\"score\":");
    append_float(s, m->anomaly_score);

    if (m->anomaly_heatmap) {
        gsize hbytes = gst_memory_get_sizes(m->anomaly_heatmap, NULL, NULL);
        std::vector<float> heat(hbytes / sizeof(float));
        if (GPU_MAP(m->anomaly_heatmap, heat)) {
            g_string_append(s, ",\"heatmap\":[");
            gsize n = heat.size();
            if (n > 64) n = 64;
            for (gsize i = 0; i < n; i++) {
                if (i > 0) g_string_append(s, ",");
                g_string_append_printf(s, "%.6f", heat[i]);
            }
            g_string_append(s, "]");
        }
    }
    g_string_append(s, "}");
}

/* ──────────────────────────────────────────────
 *  Raw output tensors serializer
 * ────────────────────────────────────────────── */

static gboolean probe_tensors(MagmaInferenceMeta* m) {
    return m && m->output_tensors && m->output_tensors->len > 0;
}

static void serialize_tensors_json(MagmaInferenceMeta* m, GString* s) {
    g_string_append(s, "\"raw_outputs\":[");
    for (guint i = 0; i < m->output_tensors->len; i++) {
        if (i > 0) g_string_append(s, ",");
        GstMemory* mem = (GstMemory*)g_ptr_array_index(m->output_tensors, i);
        gsize bytes = gst_memory_get_sizes(mem, NULL, NULL);
        std::vector<float> host(bytes / sizeof(float));
        if (!GPU_MAP(mem, host)) {
            g_string_append(s, "[]");
            continue;
        }
        gsize nf = host.size();
        g_string_append_printf(s, "[%zu floats]:[", nf);
        if (nf > 16) nf = 16;
        for (gsize j = 0; j < nf; j++) {
            if (j > 0) g_string_append(s, ",");
            append_float(s, host[j]);
        }
        g_string_append(s, "]");
    }
    g_string_append(s, "]");
}

/* ──────────────────────────────────────────────
 *  Serializer registry
 *  Add new entries here for future model output types.
 * ────────────────────────────────────────────── */

static const SerializerEntry serializers[] = {
    {"detections", probe_detections, serialize_detections_json},
    {"masks",      probe_masks,      serialize_masks_json},
    {"anomaly",    probe_anomaly,    serialize_anomaly_json},
    {"raw_outputs",probe_tensors,    serialize_tensors_json},
    {NULL,         NULL,             NULL}
};

/* ──────────────────────────────────────────────
 *  Main serialize functions
 * ────────────────────────────────────────────── */

static gchar* serialize_to_json(MagmaInferenceMeta* m, int* out_len) {
    GString* s = g_string_new("");

    g_string_append(s, "{");
    if (m) {
        g_string_append_printf(s, "\"source_width\":%u,\"source_height\":%u,",
                               m->source_width, m->source_height);
    }
    g_string_append_printf(s, "\"timestamp_ns\":%" G_GUINT64_FORMAT,
                           (guint64)g_get_real_time() * 1000);

    gboolean any = FALSE;
    for (const SerializerEntry* e = serializers; e->key; e++) {
        if (e->probe(m)) {
            g_string_append_c(s, ',');
            e->to_json(m, s);
            any = TRUE;
        }
    }

    if (!any && m) {
        g_string_append(s, ",\"detections\":[]");
    }

    g_string_append(s, "}");
    *out_len = (int)s->len;
    return g_string_free(s, FALSE);
}

#ifdef HAVE_PROTOBUF
#include "magma_msg.pb.h"

static gchar* serialize_to_protobuf(MagmaInferenceMeta* m, int* out_len) {
    magma::FrameResult result;
    if (m) {
        result.set_source_width(m->source_width);
        result.set_source_height(m->source_height);
    }
    result.set_timestamp_ns((guint64)g_get_real_time() * 1000);

    if (probe_detections(m)) {
        std::vector<MagmaInferObjectGPU> host((gsize)m->num_objects);
        if (GPU_MAP(m->objects_gpu, host)) {
            for (guint i = 0; i < m->num_objects; i++) {
                magma::Detection* d = result.add_objects();
                d->set_class_id(host[i].class_id);
                d->set_confidence(host[i].confidence);
                d->set_x(host[i].x);
                d->set_y(host[i].y);
                d->set_width(host[i].width);
                d->set_height(host[i].height);
            }
        }
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

    gst_element_class_set_static_metadata(element_class, "Magma Serialize", "Filter/Converter", "Serializes MagmaInferenceMeta (detections, masks, anomaly, raw tensors) to JSON or protobuf", "Magma");

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
