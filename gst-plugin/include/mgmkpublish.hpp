#pragma once

/**
 * @file mgmkpublish.hpp
 * @brief Kafka sink element — publishes inference results to a Kafka topic.
 */

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_PUBLISH (gst_magma_publish_get_type())
G_DECLARE_FINAL_TYPE(GstMagmaPublish, gst_magma_publish, GST, MAGMA_PUBLISH, GstBaseSink)

/**
 * @brief Magma Kafka publish sink.
 *
 * Reads MagmaInferenceMeta from the buffer and publishes serialized
 * results (JSON or Protobuf) to a Kafka topic via librdkafka.
 *
 * @property broker     Kafka broker address
 * @property topic      Kafka topic name
 * @property client-id  Kafka client ID
 * @property compression Compression codec
 * @property extra-flags Additional librdkafka config flags
 */
struct _GstMagmaPublish {
    GstBaseSink parent;

    gchar* broker;
    gchar* topic;
    gchar* client_id;
    gchar* compression;
    gchar* extra_flags;

    void* rk_handle;
};

G_END_DECLS
