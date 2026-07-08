#pragma once

/**
 * @file mgmtensordump.hpp
 * @brief Tensor dump element — writes tensor data to disk with optional RGB preview.
 */

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>
#include "magma-meta.h"

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_TENSOR_DUMP (gst_magma_tensor_dump_get_type())

G_DECLARE_FINAL_TYPE(GstMagmaTensorDump, gst_magma_tensor_dump, GST, MAGMA_TENSOR_DUMP, GstBaseTransform)

/**
 * @brief Magma tensor dump element.
 *
 * Reads MagmaTensorMeta from input buffers, downloads the tensor
 * from GPU to CPU, and writes it to a file. Optionally writes a
 * downscaled RGB preview image alongside the raw data.
 *
 * @property net-width      Expected tensor width
 * @property net-height     Expected tensor height
 * @property dump-location  Output directory path
 * @property frame-skip     Dump every N frames (default 1)
 */
struct _GstMagmaTensorDump {
    GstBaseTransform parent;

    gint net_width;
    gint net_height;
    gchar* dump_location;
    gint frame_skip;

    gint in_width;
    gint in_height;

    hipStream_t hip_stream;
    hipExternalMemory_t tensor_ext_mem;
    hipDeviceptr_t d_tensor;
    gboolean tensor_imported;

    float* host_tensor;
    gsize tensor_bytes;

    gint panel_w;
    gint panel_h;

    FILE* dump_file;
    guint64 frame_num;
};

G_END_DECLS
