#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideodecoder.h>
#include <hip/hip_runtime.h>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_H264_DEC (gst_magma_h264_dec_get_type())
G_DECLARE_FINAL_TYPE(GstMagmaH264Dec, gst_magma_h264_dec, GST, MAGMA_H264_DEC, GstVideoDecoder)

// PendingFrame is stored in a C-compatible array managed manually
#define MAX_PENDING_FRAMES 32
typedef struct {
    GstVideoCodecFrame* frame;
    int64_t pts_roc;  // PTS in rocDecode units (100ns)
} PendingFrame;

struct _GstMagmaH264Dec {
    GstVideoDecoder parent;

    // rocDecode decoder
    void* roc_decoder;       // RocVideoDecoder*

    // pending frames awaiting output from decoder (decode order FIFO)
    PendingFrame pending[MAX_PENDING_FRAMES];
    gint pending_count;

    // current frame info
    gint width, height;
    gboolean configured;

    // codec_data (SPS/PPS) in Annex B format — prepended to first bitstream
    uint8_t* codec_data;
    gsize codec_data_size;
};

G_END_DECLS
