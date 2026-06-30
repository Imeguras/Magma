#include "mgmkpublish.hpp"
#include <cstring>
#include <string>
#include <librdkafka/rdkafka.h>

GST_DEBUG_CATEGORY_STATIC(magma_publish_debug);
#define GST_CAT_DEFAULT magma_publish_debug

enum {
    PROP_0,
    PROP_BROKER,
    PROP_TOPIC,
    PROP_CLIENT_ID,
    PROP_COMPRESSION,
    PROP_EXTRA_FLAGS,
};

G_DEFINE_TYPE(GstMagmaPublish, gst_magma_publish, GST_TYPE_BASE_SINK)

/* ---------- pad templates ---------- */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("application/x-magma-msg"));

/* ---------- render ---------- */
static GstFlowReturn gst_magma_publish_render(GstBaseSink* bsink, GstBuffer* buf) {
    GstMagmaPublish* self = GST_MAGMA_PUBLISH(bsink);

    if (!self->rk_handle) {
        GST_WARNING_OBJECT(self, "Kafka producer not initialized");
        return GST_FLOW_OK;
    }

    GstMapInfo info;
    if (!gst_buffer_map(buf, &info, GST_MAP_READ)) {
        GST_ERROR_OBJECT(self, "Failed to map buffer");
        return GST_FLOW_ERROR;
    }

    rd_kafka_t* rk = (rd_kafka_t*)self->rk_handle;
    rd_kafka_topic_t* rkt = rd_kafka_topic_new(rk, self->topic, NULL);
    if (!rkt) {
        GST_ERROR_OBJECT(self, "Failed to create topic handle: %s", rd_kafka_err2str(rd_kafka_last_error()));
        gst_buffer_unmap(buf, &info);
        return GST_FLOW_ERROR;
    }

    rd_kafka_resp_err_t err = (rd_kafka_resp_err_t)rd_kafka_produce(rkt, RD_KAFKA_PARTITION_UA, RD_KAFKA_MSG_F_COPY, info.data, info.size, NULL, 0, NULL);

    if (err != RD_KAFKA_RESP_ERR_NO_ERROR) {
        GST_WARNING_OBJECT(self, "Kafka produce failed: %s", rd_kafka_err2str(err));
    }

    rd_kafka_topic_destroy(rkt);
    gst_buffer_unmap(buf, &info);

    rd_kafka_poll(rk, 0);

    return GST_FLOW_OK;
}

/* ---------- start / stop ---------- */
static gboolean gst_magma_publish_start(GstBaseSink* bsink) {
    GstMagmaPublish* self = GST_MAGMA_PUBLISH(bsink);

    if (self->rk_handle)
        return TRUE;

    char errstr[512];
    rd_kafka_conf_t* conf = rd_kafka_conf_new();

    if (self->client_id)
        rd_kafka_conf_set(conf, "client.id", self->client_id, errstr, sizeof(errstr));

    if (self->broker)
        rd_kafka_conf_set(conf, "bootstrap.servers", self->broker, errstr, sizeof(errstr));

    if (self->compression && g_strcmp0(self->compression, "none") != 0)
        rd_kafka_conf_set(conf, "compression.codec", self->compression, errstr, sizeof(errstr));

    if (self->extra_flags && strlen(self->extra_flags) > 0) {
        /* extra-flags: comma-separated key=value pairs */
        gchar** pairs = g_strsplit(self->extra_flags, ",", 0);
        for (gchar** p = pairs; p && *p; p++) {
            gchar** kv = g_strsplit(*p, "=", 2);
            if (kv[0] && kv[1]) {
                rd_kafka_conf_set(conf, kv[0], kv[1], errstr, sizeof(errstr));
            }
            g_strfreev(kv);
        }
        g_strfreev(pairs);
    }

    rd_kafka_t* rk = rd_kafka_new(RD_KAFKA_PRODUCER, conf, errstr, sizeof(errstr));
    if (!rk) {
        GST_ERROR_OBJECT(self, "Failed to create Kafka producer: %s", errstr);
        return FALSE;
    }

    self->rk_handle = rk;
    GST_INFO_OBJECT(self, "Kafka producer started: broker=%s topic=%s", self->broker, self->topic);
    return TRUE;
}

static gboolean gst_magma_publish_stop(GstBaseSink* bsink) {
    GstMagmaPublish* self = GST_MAGMA_PUBLISH(bsink);

    if (self->rk_handle) {
        rd_kafka_t* rk = (rd_kafka_t*)self->rk_handle;
        rd_kafka_flush(rk, 5000);
        rd_kafka_destroy(rk);
        self->rk_handle = NULL;
    }
    return TRUE;
}

/* ---------- event: caps ---------- */
static gboolean gst_magma_publish_event(GstBaseSink* bsink, GstEvent* event) {
    return GST_BASE_SINK_CLASS(gst_magma_publish_parent_class)->event(bsink, event);
}

/* ---------- properties ---------- */
static void gst_magma_publish_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
    GstMagmaPublish* self = GST_MAGMA_PUBLISH(object);
    switch (prop_id) {
    case PROP_BROKER:
        g_free(self->broker);
        self->broker = g_value_dup_string(value);
        break;
    case PROP_TOPIC:
        g_free(self->topic);
        self->topic = g_value_dup_string(value);
        break;
    case PROP_CLIENT_ID:
        g_free(self->client_id);
        self->client_id = g_value_dup_string(value);
        break;
    case PROP_COMPRESSION:
        g_free(self->compression);
        self->compression = g_value_dup_string(value);
        break;
    case PROP_EXTRA_FLAGS:
        g_free(self->extra_flags);
        self->extra_flags = g_value_dup_string(value);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_magma_publish_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
    GstMagmaPublish* self = GST_MAGMA_PUBLISH(object);
    switch (prop_id) {
    case PROP_BROKER:
        g_value_set_string(value, self->broker);
        break;
    case PROP_TOPIC:
        g_value_set_string(value, self->topic);
        break;
    case PROP_CLIENT_ID:
        g_value_set_string(value, self->client_id);
        break;
    case PROP_COMPRESSION:
        g_value_set_string(value, self->compression);
        break;
    case PROP_EXTRA_FLAGS:
        g_value_set_string(value, self->extra_flags);
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
        break;
    }
}

static void gst_magma_publish_finalize(GObject* object) {
    GstMagmaPublish* self = GST_MAGMA_PUBLISH(object);
    g_free(self->broker);
    g_free(self->topic);
    g_free(self->client_id);
    g_free(self->compression);
    g_free(self->extra_flags);
    G_OBJECT_CLASS(gst_magma_publish_parent_class)->finalize(object);
}

static void gst_magma_publish_init(GstMagmaPublish* self) {
    self->broker = g_strdup("localhost:9092");
    self->topic = g_strdup("magma");
    self->client_id = g_strdup("magma-publisher");
    self->compression = g_strdup("none");
    self->extra_flags = g_strdup("");
    self->rk_handle = NULL;
}

static void gst_magma_publish_class_init(GstMagmaPublishClass* klass) {
    GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
    GstBaseSinkClass* sink_class = GST_BASE_SINK_CLASS(klass);

    gobject_class->set_property = gst_magma_publish_set_property;
    gobject_class->get_property = gst_magma_publish_get_property;
    gobject_class->finalize = gst_magma_publish_finalize;

    sink_class->start = gst_magma_publish_start;
    sink_class->stop = gst_magma_publish_stop;
    sink_class->render = gst_magma_publish_render;
    sink_class->event = gst_magma_publish_event;

    g_object_class_install_property(gobject_class, PROP_BROKER, g_param_spec_string("broker", "Broker", "Kafka broker address (host:port)", "localhost:9092", G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_TOPIC, g_param_spec_string("topic", "Topic", "Kafka topic name", "magma", G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_CLIENT_ID, g_param_spec_string("client-id", "Client ID", "Kafka client ID", "magma-publisher", G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_COMPRESSION, g_param_spec_string("compression", "Compression", "Compression codec (none, gzip, snappy, lz4, zstd)", "none", G_PARAM_READWRITE));
    g_object_class_install_property(gobject_class, PROP_EXTRA_FLAGS, g_param_spec_string("extra-flags", "Extra Flags", "Comma-separated key=value Kafka config flags", "", G_PARAM_READWRITE));

    gst_element_class_add_static_pad_template(element_class, &sink_template);

    gst_element_class_set_static_metadata(element_class, "Magma Publish", "Sink", "Publishes serialized inference results to Kafka", "Magma");

    GST_DEBUG_CATEGORY_INIT(magma_publish_debug, "magma_publish", 0, "Magma Publish Plugin");
}

static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "mgmkpublish", GST_RANK_NONE, GST_TYPE_MAGMA_PUBLISH);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mgmkpublish, "Magma Publish Plugin", plugin_init, "0.1.0", "LGPL", "magma", "https://imeguras.eu.org")
