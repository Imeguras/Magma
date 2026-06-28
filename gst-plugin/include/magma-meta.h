#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

typedef struct _MagmaTensorMeta MagmaTensorMeta;

struct _MagmaTensorMeta {
    GstMeta meta;

    GstMemory *tensor_mem;
    gint width;
    gint height;
    gint channels;
};

GType magma_tensor_meta_api_get_type(void);
const GstMetaInfo *magma_tensor_meta_get_info(void);

MagmaTensorMeta *magma_buffer_add_tensor_meta(GstBuffer *buffer, GstMemory *tensor_mem,
                                               gint width, gint height, gint channels);

#define MAGMA_TENSOR_META_API_TYPE (magma_tensor_meta_api_get_type())
#define magma_buffer_get_tensor_meta(b) ((MagmaTensorMeta *)gst_buffer_get_meta((b), MAGMA_TENSOR_META_API_TYPE))

G_END_DECLS
