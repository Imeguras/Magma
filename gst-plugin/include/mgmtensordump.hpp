#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
#include <gst/allocators/gstdmabuf.h>
#include <hip/hip_runtime.h>
#include "magma-meta.h"

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_TENSOR_DUMP (gst_magma_tensor_dump_get_type())

G_DECLARE_FINAL_TYPE(GstMagmaTensorDump, gst_magma_tensor_dump, GST, MAGMA_TENSOR_DUMP, GstBaseTransform)

struct _GstMagmaTensorDump {
    GstBaseTransform parent;

    // Properties
    gint net_width;
    gint net_height;
    gchar* dump_location;
    gint frame_skip;

    // Input frame info (from caps)
    gint in_width;
    gint in_height;

    // HIP stream + imported tensor (same DMABuf reused each frame)
    hipStream_t hip_stream;
    hipExternalMemory_t tensor_ext_mem;
    hipDeviceptr_t d_tensor;
    gboolean tensor_imported;

    // Host-side tensor buffer
    float* host_tensor;
    gsize tensor_bytes;

    // Preview panel dimensions (downsampled to fit screen)
    gint panel_w;
    gint panel_h;

    // Dump file
    FILE* dump_file;
    guint64 frame_num;
};

G_END_DECLS
