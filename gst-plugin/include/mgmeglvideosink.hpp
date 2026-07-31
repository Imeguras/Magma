#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasesink.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/video/video-overlay-composition.h>
#include <epoxy/egl.h>
#include <xcb/xcb.h>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_EGL_VIDEO_SINK (gst_magma_egl_video_sink_get_type ())
G_DECLARE_FINAL_TYPE (GstMagmaEGLVideoSink, gst_magma_egl_video_sink,
                      GST, MAGMA_EGL_VIDEO_SINK, GstBaseSink)

struct _GstMagmaEGLVideoSink {
    GstBaseSink parent;

    gint in_width;
    gint in_height;
    gint in_stride;

    gboolean sync;
    gboolean show_hud;
    gint win_width;
    gint win_height;

    guintptr window_handle;
    gboolean handle_set;

    GAsyncQueue* frame_queue;
    volatile gboolean render_thread_running;
    GThread* render_thread;

    GMutex fps_mutex;
    guint64 fps_last_time;
    gdouble fps_instant;
    gdouble fps_avg;
    guint64 frames_rendered;
    gint frames_dropped;

    /* Stashed for deferred cleanup in finalize (after all pipeline
     * elements have released their GPU resources) */
    EGLDisplay egl_display;
    xcb_connection_t* xcb_conn;
    xcb_window_t      xcb_win;


};

G_END_DECLS
