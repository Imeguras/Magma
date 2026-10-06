#pragma once

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/base/gstbasesink.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/video/video-overlay-composition.h>
#include <epoxy/egl.h>
#include <xcb/xcb.h>

G_BEGIN_DECLS

#define GST_TYPE_MAGMA_EGL_VIDEO_SINK (gst_magma_egl_video_sink_get_type())
G_DECLARE_FINAL_TYPE(GstMagmaEGLVideoSink, gst_magma_egl_video_sink, GST, MAGMA_EGL_VIDEO_SINK, GstBaseSink)

struct _GstMagmaEGLVideoSink {
	GstBaseSink parent;

	gint in_width;
	gint in_height;
	gint in_stride;

	/* NOTE: named "vsync" as a property — GstBaseSink already owns "sync". */
	gboolean vsync;
	gboolean show_hud;
	gint win_width; /* current window size (render thread) */
	gint win_height;
	gboolean win_size_explicit; /* TRUE once window-width/height is set */
	gint pending_resize_w;      /* atomic: >0 asks render thread to resize */
	gint pending_resize_h;

	guintptr window_handle;
	gboolean handle_set;

	GAsyncQueue* frame_queue;
	gint render_thread_running; /* atomic — shared render/streaming thread */
	GThread* render_thread;

	GMutex fps_mutex;
	guint64 fps_last_time; /* last frame-present time (actual) */
	gdouble fps_instant;   /* actual FPS (presented, vsync-bound) */
	gdouble fps_avg;
	guint64 pot_last_time; /* last frame-arrival time (potential) */
	gdouble pot_instant;   /* potential FPS (pipeline delivery rate) */
	gdouble pot_avg;
	guint64 frames_rendered;
	gint frames_dropped;

	/* Stashed for deferred cleanup in finalize (after all pipeline
	 * elements have released their GPU resources) */
	EGLDisplay egl_display;
	xcb_connection_t* xcb_conn;
	xcb_window_t xcb_win;
	gboolean owns_window; /* FALSE when using an app-provided handle */
};

G_END_DECLS
