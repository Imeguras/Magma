#pragma once

/**
 * @file mgmh264dec.hpp
 * @brief H.264 GPU decoder element using rocDecode.
 */

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideodecoder.h>
#include <hip/hip_runtime.h>
#include <vector>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_H264_DEC (gst_magma_h264_dec_get_type())
G_DECLARE_FINAL_TYPE(GstMagmaH264Dec, gst_magma_h264_dec, GST, MAGMA_H264_DEC, GstVideoDecoder)

#define MAX_PENDING_FRAMES 32

/**
 * @brief A decoded frame awaiting reordering and output.
 */
typedef struct {
    GstVideoCodecFrame* frame;
    int64_t pts_roc;  /**< PTS in rocDecode units (100ns) */
} PendingFrame;

/**
 * @brief Magma H.264 decoder element.
 *
 * Uses rocDecode for GPU-accelerated H.264 decoding. Produces NV12
 * video frames backed by DMABuf or MagmaHipMeta.
 */
struct _GstMagmaH264Dec {
    GstVideoDecoder parent;

    void* roc_decoder;

    PendingFrame pending[MAX_PENDING_FRAMES];
    gint pending_count;

    gint width, height;
    GstVideoFormat output_format;
    gboolean configured;

    uint8_t* codec_data;
    gsize codec_data_size;

    /* Frame buffer pool: avoids hipMalloc/hipFree per frame */
#define FRAME_POOL_SIZE 16
    GAsyncQueue* pool_free;

    /* Reusable bitstream conversion buffers to avoid per-frame malloc */
    std::vector<uint8_t> bitstream_buf;
    std::vector<uint8_t> avcc_buf;
};

G_END_DECLS
