#pragma once

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_SERIALIZE (gst_magma_serialize_get_type())
G_DECLARE_FINAL_TYPE(GstMagmaSerialize, gst_magma_serialize, GST, MAGMA_SERIALIZE, GstBaseTransform)

struct _GstMagmaSerialize {
    GstBaseTransform parent;
    gchar* format;
};

G_END_DECLS
