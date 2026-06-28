#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_PUBLISH (gst_magma_publish_get_type())
G_DECLARE_FINAL_TYPE(GstMagmaPublish, gst_magma_publish, GST, MAGMA_PUBLISH, GstBaseSink)

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
