#include "mgmeglvideosink.hpp"

#include <epoxy/egl.h>
#include <epoxy/gl.h>

#include <xcb/xcb.h>
#include <xcb/xcb_aux.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <drm/drm_fourcc.h>

#ifndef EGL_PLATFORM_XCB_EXT
#define EGL_PLATFORM_XCB_EXT 0x31DC
#endif

GST_DEBUG_CATEGORY_STATIC(mgmeglvideosink_debug);
#define GST_CAT_DEFAULT mgmeglvideosink_debug

enum { PROP_0, PROP_VSYNC, PROP_SHOW_HUD, PROP_WIN_W, PROP_WIN_H };

// ─── Forward declarations ─────────────────────────────────────────────
static void gst_magma_egl_video_sink_video_overlay_init(gpointer g_iface, gpointer iface_data);

// ─── Type registration ───────────────────────────────────────────────
G_DEFINE_TYPE_WITH_CODE(GstMagmaEGLVideoSink, gst_magma_egl_video_sink, GST_TYPE_BASE_SINK, G_IMPLEMENT_INTERFACE(GST_TYPE_VIDEO_OVERLAY, gst_magma_egl_video_sink_video_overlay_init))

// ─── Queue wake-up sentinel ───────────────────────────────────────────
// GAsyncQueue refuses NULL payloads, so stop() pushes this address instead.
static gpointer queue_wakeup_sentinel(void) {
	static int marker;
	return &marker;
}

// ─── monotonic µs ─────────────────────────────────────────────────────
static inline guint64 now_us(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (guint64)ts.tv_sec * 1000000ULL + (guint64)ts.tv_nsec / 1000ULL;
}

// ─── GL shaders (GLES 3.0) ───────────────────────────────────────────
static const char* vertex_src = R"glsl(
#version 300 es
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_tex;
out vec2 v_tex;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_tex = a_tex;
}
)glsl";

static const char* frag_yuv_src = R"glsl(
#version 300 es
precision mediump float;
in vec2 v_tex;
out vec4 frag_color;
uniform sampler2D u_tex_y;
uniform sampler2D u_tex_uv;
void main() {
    float y = texture(u_tex_y, v_tex).r;
    vec2 uv = texture(u_tex_uv, v_tex).rg;
    float yv = y * 255.0 - 16.0;
    float u_val = uv.x * 255.0 - 128.0;
    float v_val = uv.y * 255.0 - 128.0;
    float r = (1.164 * yv + 1.596 * v_val) / 255.0;
    float g = (1.164 * yv - 0.391 * u_val - 0.813 * v_val) / 255.0;
    float b = (1.164 * yv + 2.018 * u_val) / 255.0;
    frag_color = vec4 (clamp (r, 0.0, 1.0),
                       clamp (g, 0.0, 1.0),
                       clamp (b, 0.0, 1.0), 1.0);
}
)glsl";

// ─── Render-thread state (owned exclusively by render thread) ──────────
struct RenderState {
	xcb_connection_t* conn = nullptr;
	xcb_screen_t* screen = nullptr;
	xcb_window_t win = 0;
	xcb_atom_t wm_delete = 0;
	xcb_atom_t wm_protocols = 0;
	bool eos_sent = false;

	EGLDisplay dpy = EGL_NO_DISPLAY;
	EGLContext ctx = EGL_NO_CONTEXT;
	EGLSurface surf = EGL_NO_SURFACE;
	EGLConfig cfg = nullptr;

	GLuint program = 0;
	GLuint vao = 0, vbo = 0;
	GLuint tex_y = 0, tex_uv = 0;
	GLint u_tex_y_loc = -1;
	GLint u_tex_uv_loc = -1;

	GLuint hud_program = 0;
	GLuint hud_tex = 0;
	GLuint hud_vao = 0, hud_vbo = 0;
	char hud_last_text[160] = {};
	int hud_cache_w = 0, hud_cache_h = 0;

	int last_w = 0, last_h = 0;
	bool has_valid = false;

	char display_name[64] = {};
};

static GLuint create_program(const char* vs_src, const char* fs_src);

// ─── 5x7 bitmap font (full ASCII 32-126, reused from osd_kernels.hip) ──
static const unsigned char font5x7[95][7] = {
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}, // 32 space
    {0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04}, // 33 !
    {0x0a, 0x0a, 0x0a, 0x00, 0x00, 0x00, 0x00}, // 34 "
    {0x0a, 0x0a, 0x1f, 0x0a, 0x1f, 0x0a, 0x0a}, // 35 #
    {0x0e, 0x15, 0x05, 0x0e, 0x14, 0x15, 0x0e}, // 36 $
    {0x00, 0x12, 0x09, 0x04, 0x12, 0x09, 0x00}, // 37 %
    {0x0c, 0x12, 0x12, 0x0c, 0x12, 0x12, 0x0c}, // 38 &
    {0x04, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00}, // 39 '
    {0x08, 0x04, 0x04, 0x04, 0x04, 0x04, 0x08}, // 40 (
    {0x02, 0x04, 0x04, 0x04, 0x04, 0x04, 0x02}, // 41 )
    {0x00, 0x04, 0x15, 0x0e, 0x15, 0x04, 0x00}, // 42 *
    {0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00}, // 43 +
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x08}, // 44 ,
    {0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00}, // 45 -
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04}, // 46 .
    {0x10, 0x08, 0x08, 0x04, 0x02, 0x02, 0x01}, // 47 /
    {0x0e, 0x11, 0x19, 0x15, 0x13, 0x11, 0x0e}, // 48 0
    {0x04, 0x06, 0x04, 0x04, 0x04, 0x04, 0x0e}, // 49 1
    {0x0e, 0x11, 0x10, 0x08, 0x04, 0x02, 0x1f}, // 50 2
    {0x1f, 0x08, 0x04, 0x08, 0x10, 0x11, 0x0e}, // 51 3
    {0x08, 0x0c, 0x0a, 0x09, 0x1f, 0x08, 0x08}, // 52 4
    {0x1f, 0x01, 0x0f, 0x10, 0x10, 0x11, 0x0e}, // 53 5
    {0x0c, 0x02, 0x01, 0x0f, 0x11, 0x11, 0x0e}, // 54 6
    {0x1f, 0x10, 0x08, 0x04, 0x02, 0x02, 0x02}, // 55 7
    {0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e}, // 56 8
    {0x0e, 0x11, 0x11, 0x1e, 0x10, 0x08, 0x06}, // 57 9
    {0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00}, // 58 :
    {0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x08}, // 59 ;
    {0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08}, // 60 <
    {0x00, 0x00, 0x1f, 0x00, 0x1f, 0x00, 0x00}, // 61 =
    {0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02}, // 62 >
    {0x0e, 0x11, 0x10, 0x08, 0x04, 0x00, 0x04}, // 63 ?
    {0x0e, 0x11, 0x10, 0x16, 0x15, 0x15, 0x0e}, // 64 @
    {0x04, 0x0a, 0x11, 0x11, 0x1f, 0x11, 0x11}, // 65 A
    {0x0f, 0x11, 0x11, 0x0f, 0x11, 0x11, 0x0f}, // 66 B
    {0x0e, 0x11, 0x01, 0x01, 0x01, 0x11, 0x0e}, // 67 C
    {0x0f, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0f}, // 68 D
    {0x1f, 0x01, 0x01, 0x0f, 0x01, 0x01, 0x1f}, // 69 E
    {0x1f, 0x01, 0x01, 0x0f, 0x01, 0x01, 0x01}, // 70 F
    {0x0e, 0x11, 0x01, 0x1d, 0x11, 0x11, 0x0e}, // 71 G
    {0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11}, // 72 H
    {0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e}, // 73 I
    {0x1c, 0x08, 0x08, 0x08, 0x08, 0x09, 0x06}, // 74 J
    {0x11, 0x09, 0x05, 0x03, 0x05, 0x09, 0x11}, // 75 K
    {0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x1f}, // 76 L
    {0x11, 0x1b, 0x15, 0x11, 0x11, 0x11, 0x11}, // 77 M
    {0x11, 0x13, 0x15, 0x19, 0x11, 0x11, 0x11}, // 78 N
    {0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e}, // 79 O
    {0x0f, 0x11, 0x11, 0x0f, 0x01, 0x01, 0x01}, // 80 P
    {0x0e, 0x11, 0x11, 0x11, 0x15, 0x09, 0x16}, // 81 Q
    {0x0f, 0x11, 0x11, 0x0f, 0x05, 0x09, 0x11}, // 82 R
    {0x0e, 0x11, 0x01, 0x0e, 0x10, 0x11, 0x0e}, // 83 S
    {0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}, // 84 T
    {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e}, // 85 U
    {0x11, 0x11, 0x11, 0x11, 0x0a, 0x0a, 0x04}, // 86 V
    {0x11, 0x11, 0x11, 0x15, 0x15, 0x1b, 0x11}, // 87 W
    {0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11}, // 88 X
    {0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04}, // 89 Y
    {0x1f, 0x10, 0x08, 0x04, 0x02, 0x01, 0x1f}, // 90 Z
    {0x0e, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0e}, // 91 [
    {0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10}, // 92 backslash
    {0x0e, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0e}, // 93 ]
    {0x04, 0x0a, 0x11, 0x00, 0x00, 0x00, 0x00}, // 94 ^
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f}, // 95 _
    {0x02, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00}, // 96 `
    {0x00, 0x00, 0x0e, 0x10, 0x1e, 0x11, 0x1e}, // 97 a
    {0x01, 0x01, 0x0f, 0x11, 0x11, 0x11, 0x0f}, // 98 b
    {0x00, 0x00, 0x0e, 0x11, 0x01, 0x11, 0x0e}, // 99 c
    {0x10, 0x10, 0x1e, 0x11, 0x11, 0x11, 0x1e}, // 100 d
    {0x00, 0x00, 0x0e, 0x11, 0x1f, 0x01, 0x0e}, // 101 e
    {0x0c, 0x12, 0x02, 0x07, 0x02, 0x02, 0x02}, // 102 f
    {0x00, 0x00, 0x1e, 0x11, 0x1e, 0x10, 0x0e}, // 103 g
    {0x01, 0x01, 0x0f, 0x11, 0x11, 0x11, 0x11}, // 104 h
    {0x04, 0x00, 0x06, 0x04, 0x04, 0x04, 0x0e}, // 105 i
    {0x08, 0x00, 0x0c, 0x08, 0x08, 0x09, 0x06}, // 106 j
    {0x01, 0x01, 0x09, 0x05, 0x03, 0x05, 0x09}, // 107 k
    {0x06, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e}, // 108 l
    {0x00, 0x00, 0x0b, 0x15, 0x15, 0x11, 0x11}, // 109 m
    {0x00, 0x00, 0x0f, 0x11, 0x11, 0x11, 0x11}, // 110 n
    {0x00, 0x00, 0x0e, 0x11, 0x11, 0x11, 0x0e}, // 111 o
    {0x00, 0x00, 0x0f, 0x11, 0x0f, 0x01, 0x01}, // 112 p
    {0x00, 0x00, 0x1e, 0x11, 0x1e, 0x10, 0x10}, // 113 q
    {0x00, 0x00, 0x0f, 0x11, 0x01, 0x01, 0x01}, // 114 r
    {0x00, 0x00, 0x0e, 0x01, 0x0e, 0x10, 0x0e}, // 115 s
    {0x02, 0x02, 0x0f, 0x02, 0x02, 0x12, 0x0c}, // 116 t
    {0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0d}, // 117 u
    {0x00, 0x00, 0x11, 0x11, 0x0a, 0x0a, 0x04}, // 118 v
    {0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0a}, // 119 w
    {0x00, 0x00, 0x11, 0x0a, 0x04, 0x0a, 0x11}, // 120 x
    {0x00, 0x00, 0x11, 0x11, 0x1e, 0x10, 0x0e}, // 121 y
    {0x00, 0x00, 0x1f, 0x08, 0x04, 0x02, 0x1f}, // 122 z
    {0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08}, // 123 {
    {0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}, // 124 |
    {0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02}, // 125 }
    {0x00, 0x00, 0x00, 0x00, 0x12, 0x09, 0x00}, // 126 ~
};

// Char width = 5 glyph + 1 pad, height = 7 glyph + 1 pad
#define HUD_CHAR_W 6
#define HUD_CHAR_H (7 + 1)

static int hud_text_width(const char* s) {
	return (int)strlen(s) * HUD_CHAR_W + 4;
}
static int hud_text_height(void) {
	return HUD_CHAR_H + 4;
}

static void hud_build_text(uint8_t* rgba, int buf_w, int buf_h, const char* text) {
	memset(rgba, 0, (size_t)buf_w * buf_h * 4);
	int x0 = 2, y0 = 2;
	for (const char* p = text; *p; p++) {
		if (*p == '\n') {
			x0 = 2;
			y0 += HUD_CHAR_H + 1;
			continue;
		}
		int ci = (unsigned char)*p - 32;
		if (ci < 0 || ci > 94)
			continue;
		for (int row = 0; row < 7; row++) {
			int py = y0 + row;
			if (py < 0 || py >= buf_h)
				continue;
			unsigned char bits = font5x7[ci][row];
			for (int col = 0; col < 5; col++) {
				int px = x0 + col;
				if (px < 0 || px >= buf_w)
					continue;
				/* font5x7 is LSB-left (bit 0 = leftmost column), matching
				 * kernels/osd_kernels.hip. Using 0x10>>col mirrors every glyph. */
				if (bits & (1 << col)) {
					uint8_t* p_out = rgba + (py * buf_w + px) * 4;
					p_out[0] = p_out[1] = p_out[2] = 0xFF;
					p_out[3] = 0xFF;
				}
			}
		}
		x0 += HUD_CHAR_W;
	}
}

static const char* hud_frag_src = R"glsl(
#version 300 es
precision mediump float;
in vec2 v_tex;
out vec4 frag_color;
uniform sampler2D u_tex;
void main() {
    vec4 c = texture (u_tex, v_tex);
    if (c.a < 0.01) discard;
    frag_color = c;
}
)glsl";

static bool init_hud_resources(RenderState* rs) {
	rs->hud_program = create_program(vertex_src, hud_frag_src);
	return rs->hud_program != 0;
}

static void hud_render(RenderState* rs, const char* text, int win_w, int win_h) {
	if (!rs->hud_program || !text || !*text)
		return;

	int tw = hud_text_width(text);
	int th = hud_text_height();

	// Only regenerate texture when text changes
	bool text_changed = (strcmp(text, rs->hud_last_text) != 0);
	if (text_changed || rs->hud_tex == 0 || rs->hud_cache_w != tw || rs->hud_cache_h != th) {
		uint8_t* rgba = (uint8_t*)calloc((size_t)tw * th, 4);
		if (!rgba)
			return;
		hud_build_text(rgba, tw, th, text);
		g_strlcpy(rs->hud_last_text, text, sizeof(rs->hud_last_text));

		if (rs->hud_tex)
			glDeleteTextures(1, &rs->hud_tex);
		glGenTextures(1, &rs->hud_tex);
		glBindTexture(GL_TEXTURE_2D, rs->hud_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		free(rgba);
		rs->hud_cache_w = tw;
		rs->hud_cache_h = th;
	}

	float sx1 = (float)tw / (float)win_w * 2.0f - 1.0f;
	float sy1 = (float)th / (float)win_h * 2.0f - 1.0f;
	float y_top = -sy1; /* HUD anchored at top-left */

	float verts[] = {
	    -1, y_top, 0, 1, sx1, y_top, 1, 1, sx1, 1, 1, 0, -1, y_top, 0, 1, sx1, 1, 1, 0, -1, 1, 0, 0,
	};

	// One-time VAO/VBO creation
	if (rs->hud_vao == 0) {
		glGenVertexArrays(1, &rs->hud_vao);
		glGenBuffers(1, &rs->hud_vbo);
	}
	glBindVertexArray(rs->hud_vao);
	glBindBuffer(GL_ARRAY_BUFFER, rs->hud_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STREAM_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, (void*)(2 * 4));
	glEnableVertexAttribArray(1);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glUseProgram(rs->hud_program);
	glUniform1i(glGetUniformLocation(rs->hud_program, "u_tex"), 0);
	glBindVertexArray(rs->hud_vao);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rs->hud_tex);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
	glUseProgram(0);

	glDisable(GL_BLEND);
}

// ─── GL helpers ────────────────────────────────────────────────────────
static GLuint compile_shader(GLenum type, const char* src) {
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, nullptr);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[1024] = {};
		glGetShaderInfoLog(s, sizeof(log), nullptr, log);
		GST_ERROR("shader compile: %s", log);
		return 0;
	}
	return s;
}

static GLuint create_program(const char* vs_src, const char* fs_src) {
	GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
	GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
	if (!vs || !fs)
		return 0;
	GLuint prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);
	GLint ok = 0;
	glGetProgramiv(prog, GL_LINK_STATUS, &ok);
	if (!ok) {
		char log[1024] = {};
		glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
		GST_ERROR("program link: %s", log);
		return 0;
	}
	glDeleteShader(vs);
	glDeleteShader(fs);
	return prog;
}

// ─── Render-thread functions ──────────────────────────────────────────

// Create GL resources
static bool init_gl_resources(RenderState* rs) {
	rs->program = create_program(vertex_src, frag_yuv_src);
	if (!rs->program)
		return false;

	glUseProgram(rs->program);
	rs->u_tex_y_loc = glGetUniformLocation(rs->program, "u_tex_y");
	rs->u_tex_uv_loc = glGetUniformLocation(rs->program, "u_tex_uv");
	glUniform1i(rs->u_tex_y_loc, 0);
	glUniform1i(rs->u_tex_uv_loc, 1);
	glUseProgram(0);

	float verts[] = {-1, -1, 0, 1, 1, -1, 1, 1, 1, 1, 1, 0, -1, -1, 0, 1, 1, 1, 1, 0, -1, 1, 0, 0};
	glGenVertexArrays(1, &rs->vao);
	glGenBuffers(1, &rs->vbo);
	glBindVertexArray(rs->vao);
	glBindBuffer(GL_ARRAY_BUFFER, rs->vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, (void*)(2 * 4));
	glEnableVertexAttribArray(1);
	glBindVertexArray(0);

	glGenTextures(1, &rs->tex_y);
	glBindTexture(GL_TEXTURE_2D, rs->tex_y);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glGenTextures(1, &rs->tex_uv);
	glBindTexture(GL_TEXTURE_2D, rs->tex_uv);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	return true;
}

// Destroy GL objects only. Must NOT touch the EGL/XCB handles — the caller
// still needs rs->dpy / ctx / surf / conn to tear those down afterwards.
static void destroy_gl_resources(RenderState* rs) {
	if (rs->program)
		glDeleteProgram(rs->program);
	if (rs->hud_program)
		glDeleteProgram(rs->hud_program);
	if (rs->vao)
		glDeleteVertexArrays(1, &rs->vao);
	if (rs->vbo)
		glDeleteBuffers(1, &rs->vbo);
	if (rs->tex_y)
		glDeleteTextures(1, &rs->tex_y);
	if (rs->tex_uv)
		glDeleteTextures(1, &rs->tex_uv);
	if (rs->hud_tex)
		glDeleteTextures(1, &rs->hud_tex);
	if (rs->hud_vao)
		glDeleteVertexArrays(1, &rs->hud_vao);
	if (rs->hud_vbo)
		glDeleteBuffers(1, &rs->hud_vbo);
	rs->program = rs->hud_program = 0;
	rs->vao = rs->vbo = 0;
	rs->tex_y = rs->tex_uv = 0;
	rs->hud_tex = rs->hud_vao = rs->hud_vbo = 0;
	rs->u_tex_y_loc = rs->u_tex_uv_loc = -1;
	rs->hud_last_text[0] = '\0';
	rs->hud_cache_w = rs->hud_cache_h = 0;
}

// Release the EGL context/surface. Leaves rs->dpy alone: eglTerminate is
// deferred to finalize, after every other element has released its GPU state.
static void egl_teardown(RenderState* rs) {
	if (rs->dpy == EGL_NO_DISPLAY)
		return;
	eglMakeCurrent(rs->dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
	if (rs->surf != EGL_NO_SURFACE) {
		eglDestroySurface(rs->dpy, rs->surf);
		rs->surf = EGL_NO_SURFACE;
	}
	if (rs->ctx != EGL_NO_CONTEXT) {
		eglDestroyContext(rs->dpy, rs->ctx);
		rs->ctx = EGL_NO_CONTEXT;
	}
}

// Create XCB window (or use external handle)
static bool init_xcb_window(GstMagmaEGLVideoSink* self, RenderState* rs) {
	const char* disp = getenv("DISPLAY");
	if (!disp)
		disp = ":0";
	g_strlcpy(rs->display_name, disp, sizeof(rs->display_name));

	rs->conn = xcb_connect(disp, nullptr);
	if (xcb_connection_has_error(rs->conn)) {
		GST_ERROR_OBJECT(self, "XCB: connect to %s failed", disp);
		return false;
	}

	rs->screen = xcb_aux_get_screen(rs->conn, 0);
	if (!rs->screen) {
		GST_ERROR_OBJECT(self, "XCB: no screen");
		return false;
	}

	int w = self->win_width > 0 ? self->win_width : 960;
	int h = self->win_height > 0 ? self->win_height : 540;

	if (self->handle_set) {
		// Use application-provided window
		rs->win = (xcb_window_t)(uintptr_t)self->window_handle;

		xcb_get_geometry_cookie_t gc = xcb_get_geometry(rs->conn, rs->win);
		xcb_get_geometry_reply_t* gr = xcb_get_geometry_reply(rs->conn, gc, nullptr);
		if (gr) {
			w = gr->width;
			h = gr->height;
			free(gr);
		}

		// Select events on the existing window
		uint32_t evmask = XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_STRUCTURE_NOTIFY;
		xcb_change_window_attributes(rs->conn, rs->win, XCB_CW_EVENT_MASK, &evmask);
	} else {
		// Create our own window
		rs->win = xcb_generate_id(rs->conn);
		uint32_t mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
		uint32_t vals[] = {rs->screen->black_pixel, XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_STRUCTURE_NOTIFY | XCB_EVENT_MASK_KEY_PRESS};
		xcb_create_window(rs->conn, XCB_COPY_FROM_PARENT, rs->win, rs->screen->root, 0, 0, w, h, 0, XCB_WINDOW_CLASS_INPUT_OUTPUT, rs->screen->root_visual, mask, vals);

		// WM_DELETE_WINDOW
		static const char kDelAtom[] = "WM_DELETE_WINDOW";
		static const char kProtoAtom[] = "WM_PROTOCOLS";
		static const char kWinName[] = "mgmeglvideosink";
		xcb_intern_atom_cookie_t dc = xcb_intern_atom(rs->conn, 0, sizeof(kDelAtom) - 1, kDelAtom);
		xcb_intern_atom_reply_t* dr = xcb_intern_atom_reply(rs->conn, dc, nullptr);
		if (dr) {
			rs->wm_delete = dr->atom;
			free(dr);
		}
		xcb_intern_atom_cookie_t pc = xcb_intern_atom(rs->conn, 0, sizeof(kProtoAtom) - 1, kProtoAtom);
		xcb_intern_atom_reply_t* pr = xcb_intern_atom_reply(rs->conn, pc, nullptr);
		if (pr) {
			rs->wm_protocols = pr->atom;
			if (rs->wm_delete)
				xcb_change_property(rs->conn, XCB_PROP_MODE_REPLACE, rs->win, pr->atom, XCB_ATOM_ATOM, 32, 1, &rs->wm_delete);
			free(pr);
		}
		if (!rs->wm_delete || !rs->wm_protocols)
			GST_WARNING_OBJECT(self,
			                   "WM_DELETE_WINDOW/WM_PROTOCOLS intern failed "
			                   "— close-window detection disabled");
		xcb_change_property(rs->conn, XCB_PROP_MODE_REPLACE, rs->win, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8, sizeof(kWinName) - 1, kWinName);

		xcb_map_window(rs->conn, rs->win);
	}

	xcb_flush(rs->conn);
	self->win_width = w;
	self->win_height = h;
	GST_INFO_OBJECT(self, "XCB window 0x%x (%dx%d) on %s", (unsigned)rs->win, w, h, disp);
	return true;
}

// Create EGL display + context + surface
static bool init_egl_surface(GstMagmaEGLVideoSink* self, RenderState* rs) {
	// Resolve eglGetPlatformDisplayEXT for XCB platform
	auto eglGetPlatformDisplayEXT = (EGLDisplay(EGLAPIENTRY*)(EGLenum, void*, const EGLint*))eglGetProcAddress("eglGetPlatformDisplayEXT");
	if (eglGetPlatformDisplayEXT) {
		rs->dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_XCB_EXT, (void*)rs->conn, nullptr);
	}
	if (rs->dpy == EGL_NO_DISPLAY) {
		GST_WARNING_OBJECT(self, "EGL platform XCB not available");
		return false;
	}

	EGLint major, minor;
	if (!eglInitialize(rs->dpy, &major, &minor)) {
		GST_ERROR_OBJECT(self, "eglInitialize failed");
		return false;
	}
	GST_INFO_OBJECT(self, "EGL %d.%d", major, minor);

	const char* exts = eglQueryString(rs->dpy, EGL_EXTENSIONS);
	if (!exts || !strstr(exts, "EGL_EXT_image_dma_buf_import")) {
		GST_ERROR_OBJECT(self, "EGL_EXT_image_dma_buf_import missing");
		return false;
	}

	EGLint cfg_attrs[] = {EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE};
	EGLint n = 0;
	if (!eglChooseConfig(rs->dpy, cfg_attrs, &rs->cfg, 1, &n) || n == 0) {
		cfg_attrs[9] = EGL_OPENGL_ES2_BIT;
		if (!eglChooseConfig(rs->dpy, cfg_attrs, &rs->cfg, 1, &n) || n == 0) {
			GST_ERROR_OBJECT(self, "EGL: no suitable config");
			return false;
		}
	}

	// Try GLES 3.2, 3.1, 3.0, 2.0
	struct {
		int maj, min;
	} trials[] = {{3, 2}, {3, 1}, {3, 0}, {2, 0}};
	for (auto& t : trials) {
		EGLint a[] = {EGL_CONTEXT_MAJOR_VERSION, t.maj, EGL_CONTEXT_MINOR_VERSION, t.min, EGL_NONE};
		rs->ctx = eglCreateContext(rs->dpy, rs->cfg, EGL_NO_CONTEXT, a);
		if (rs->ctx != EGL_NO_CONTEXT) {
			GST_INFO_OBJECT(self, "GLES %d.%d", t.maj, t.min);
			break;
		}
	}
	if (rs->ctx == EGL_NO_CONTEXT) {
		GST_ERROR_OBJECT(self, "EGL context creation failed (0x%x)", eglGetError());
		return false;
	}

	rs->surf = eglCreateWindowSurface(rs->dpy, rs->cfg, (EGLNativeWindowType)(uintptr_t)rs->win, nullptr);
	if (rs->surf == EGL_NO_SURFACE) {
		GST_ERROR_OBJECT(self, "EGL surface failed (0x%x)", eglGetError());
		return false;
	}

	if (!eglMakeCurrent(rs->dpy, rs->surf, rs->surf, rs->ctx)) {
		GST_ERROR_OBJECT(self, "eglMakeCurrent failed");
		return false;
	}

	GST_INFO_OBJECT(self, "GL: %s | %s | %s", glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION));
	return true;
}

// Draw the currently bound Y/UV textures as a fullscreen quad.
// Shared by the normal render path and the XCB expose handler.
static void draw_video_quad(RenderState* rs, int vp_w, int vp_h) {
	glViewport(0, 0, vp_w, vp_h);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(rs->program);
	glBindVertexArray(rs->vao);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rs->tex_y);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, rs->tex_uv);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
	glUseProgram(0);
}

// Import dmabuf FD → EGLImages, bind to textures, render quad
// (glEGLImageTargetTexture2DOES is provided directly by libepoxy — no
//  per-frame eglGetProcAddress needed.)
static bool render_frame(GstMagmaEGLVideoSink* self, RenderState* rs, int dmabuf_fd, int w, int h, int stride, gsize uv_offset, const char* hud_text) {
	if (w <= 0 || h <= 0 || stride <= 0 || dmabuf_fd < 0)
		return false;

	// Import Y plane (R8)
	int dup_y = fcntl(dmabuf_fd, F_DUPFD_CLOEXEC, 0);
	if (dup_y < 0)
		return false;
	EGLAttrib y_attrs[] = {EGL_WIDTH,
	                       (EGLAttrib)w,
	                       EGL_HEIGHT,
	                       (EGLAttrib)h,
	                       EGL_LINUX_DRM_FOURCC_EXT,
	                       (EGLAttrib)DRM_FORMAT_R8,
	                       EGL_DMA_BUF_PLANE0_FD_EXT,
	                       (EGLAttrib)dup_y,
	                       EGL_DMA_BUF_PLANE0_OFFSET_EXT,
	                       0,
	                       EGL_DMA_BUF_PLANE0_PITCH_EXT,
	                       (EGLAttrib)stride,
	                       EGL_NONE};
	EGLImageKHR img_y = eglCreateImage(rs->dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, nullptr, y_attrs);

	if (uv_offset <= 0)
		uv_offset = (gsize)stride * h;

	// Import UV plane (GR88 at half res)
	int dup_uv = fcntl(dmabuf_fd, F_DUPFD_CLOEXEC, 0);
	if (dup_uv < 0) {
		if (img_y != EGL_NO_IMAGE_KHR)
			eglDestroyImageKHR(rs->dpy, img_y);
		close(dup_y);
		return false;
	}
	EGLAttrib uv_attrs[] = {EGL_WIDTH,
	                        (EGLAttrib)(w / 2),
	                        EGL_HEIGHT,
	                        (EGLAttrib)(h / 2),
	                        EGL_LINUX_DRM_FOURCC_EXT,
	                        (EGLAttrib)DRM_FORMAT_GR88,
	                        EGL_DMA_BUF_PLANE0_FD_EXT,
	                        (EGLAttrib)dup_uv,
	                        EGL_DMA_BUF_PLANE0_OFFSET_EXT,
	                        (EGLAttrib)uv_offset,
	                        EGL_DMA_BUF_PLANE0_PITCH_EXT,
	                        (EGLAttrib)stride,
	                        EGL_NONE};
	EGLImageKHR img_uv = eglCreateImage(rs->dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, nullptr, uv_attrs);

	/* EGL_EXT_image_dma_buf_import: EGL takes its own reference to the buffer,
	 * so the FDs must be closed here. Leaking them exhausted the FD table and
	 * pinned every decoder dmabuf for the lifetime of the pipeline. */
	close(dup_y);
	close(dup_uv);

	if (img_y == EGL_NO_IMAGE_KHR || img_uv == EGL_NO_IMAGE_KHR) {
		GST_ERROR_OBJECT(self, "EGLImage failed: Y=%s UV=%s (err=0x%x)", img_y != EGL_NO_IMAGE_KHR ? "OK" : "FAIL", img_uv != EGL_NO_IMAGE_KHR ? "OK" : "FAIL", eglGetError());
		if (img_y != EGL_NO_IMAGE_KHR)
			eglDestroyImageKHR(rs->dpy, img_y);
		if (img_uv != EGL_NO_IMAGE_KHR)
			eglDestroyImageKHR(rs->dpy, img_uv);
		return false;
	}

	// Bind EGLImages to GL textures
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rs->tex_y);
	glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img_y);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, rs->tex_uv);
	glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img_uv);

	// Destroy EGLImages (textures keep a reference to the underlying data)
	eglDestroyImageKHR(rs->dpy, img_y);
	eglDestroyImageKHR(rs->dpy, img_uv);

	// Render
	draw_video_quad(rs, self->win_width, self->win_height);

	if (hud_text && *hud_text)
		hud_render(rs, hud_text, self->win_width, self->win_height);

	eglSwapBuffers(rs->dpy, rs->surf);

	rs->last_w = w;
	rs->last_h = h;
	rs->has_valid = true;
	return true;
}

// Process XCB events (non-blocking)
static void process_xcb(GstMagmaEGLVideoSink* self, RenderState* rs) {
	xcb_generic_event_t* ev;

	// Apply a resize requested by set_caps (streaming thread). The resulting
	// CONFIGURE_NOTIFY below is what actually updates win_width/win_height.
	gint rw = g_atomic_int_get(&self->pending_resize_w);
	gint rh = g_atomic_int_get(&self->pending_resize_h);
	if (rw > 0 && rh > 0) {
		g_atomic_int_set(&self->pending_resize_w, 0);
		g_atomic_int_set(&self->pending_resize_h, 0);
		uint32_t vals[] = {(uint32_t)rw, (uint32_t)rh};
		xcb_configure_window(rs->conn, rs->win, XCB_CONFIG_WINDOW_WIDTH | XCB_CONFIG_WINDOW_HEIGHT, vals);
		xcb_flush(rs->conn);
		GST_INFO_OBJECT(self, "resizing window to %dx%d", rw, rh);
	}

	while ((ev = xcb_poll_for_event(rs->conn))) {
		uint8_t type = ev->response_type & ~0x80;
		if (type == XCB_CLIENT_MESSAGE) {
			auto* cm = (xcb_client_message_event_t*)ev;
			/* Only a WM_PROTOCOLS/WM_DELETE_WINDOW message means "close me".
			 * Matching on data32[0] alone made unrelated client messages (WM
			 * pings, sync requests, Xdnd, IME) tear the stream down, and a
			 * failed atom intern (wm_delete == 0) matched almost everything. */
			bool is_close = rs->wm_delete != 0 && rs->wm_protocols != 0 && cm->type == rs->wm_protocols && cm->format == 32 && cm->window == rs->win && cm->data.data32[0] == rs->wm_delete;
			if (is_close) {
				if (!rs->eos_sent) {
					rs->eos_sent = true;
					GST_INFO_OBJECT(self, "window close requested — sending EOS");
					gst_element_send_event(GST_ELEMENT(self), gst_event_new_eos());
				}
			} else {
				GST_DEBUG_OBJECT(self,
				                 "ignoring client message type=%u format=%u data32[0]=%u "
				                 "(wm_protocols=%u wm_delete=%u)",
				                 (unsigned)cm->type,
				                 (unsigned)cm->format,
				                 (unsigned)cm->data.data32[0],
				                 (unsigned)rs->wm_protocols,
				                 (unsigned)rs->wm_delete);
			}
		} else if (type == XCB_CONFIGURE_NOTIFY) {
			auto* ce = (xcb_configure_notify_event_t*)ev;
			self->win_width = ce->width;
			self->win_height = ce->height;
		} else if (type == XCB_EXPOSE) {
			// Redraw last frame if we have one
			if (rs->has_valid) {
				draw_video_quad(rs, self->win_width, self->win_height);
				// Re-render last HUD if visible
				if (self->show_hud && rs->hud_last_text[0])
					hud_render(rs, rs->hud_last_text, self->win_width, self->win_height);
				eglSwapBuffers(rs->dpy, rs->surf);
			}
		}
		free(ev);
	}
}

// ─── Render thread entry point ────────────────────────────────────────
static gpointer render_thread_func(gpointer data) {
	GstMagmaEGLVideoSink* self = GST_MAGMA_EGL_VIDEO_SINK(data);
	RenderState rs;

	if (!init_xcb_window(self, &rs))
		goto fail;
	if (!init_egl_surface(self, &rs))
		goto fail;
	if (!init_gl_resources(&rs))
		goto fail;
	if (!init_hud_resources(&rs)) {
		GST_WARNING_OBJECT(self, "HUD resources init failed, continuing without HUD");
	}

	// vsync
	eglSwapInterval(rs.dpy, self->vsync ? 1 : 0);

	GST_INFO_OBJECT(self, "render thread running");

	while (g_atomic_int_get(&self->render_thread_running)) {
		// Process XCB events (non-blocking)
		process_xcb(self, &rs);

		// Pop a frame (block up to 8ms)
		gpointer item = g_async_queue_timeout_pop(self->frame_queue, 8000);
		if (!item)
			continue; // timeout, poll events
		if (item == queue_wakeup_sentinel())
			break; // stop() woke us
		GstBuffer* buf = (GstBuffer*)item;

		// Import and render — use stride/offset from GstVideoMeta
		GstMemory* mem = gst_buffer_peek_memory(buf, 0);
		if (mem && gst_is_dmabuf_memory(mem)) {
			gint fd = gst_dmabuf_memory_get_fd(mem);
			if (fd >= 0) {
				gint stride = self->in_stride;
				gsize uv_offset = (gsize)stride * self->in_height;
				GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
				if (vmeta && vmeta->n_planes > 0) {
					stride = vmeta->stride[0]; // actual buffer stride
					if (vmeta->n_planes > 1)
						uv_offset = vmeta->offset[1];
				} else {
					GST_WARNING_OBJECT(self, "vmeta=%p n_planes=%d — using fallback stride=%d uv_offset=%zu", vmeta, vmeta ? vmeta->n_planes : 0, stride, uv_offset);
				}
				GST_LOG_OBJECT(self, "frame %dx%d stride=%d uv_offset=%zu fd=%d (vmeta=%p n_planes=%d)", self->in_width, self->in_height, stride, uv_offset, fd, vmeta, vmeta ? vmeta->n_planes : 0);
				int dup_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
				if (dup_fd >= 0) {
					// Build HUD text (must be ready before the swap inside render_frame)
					char hud_line[128] = "";
					if (self->show_hud) {
						guint64 frame_count = 0;
						guint64 drop_count = 0;
						gdouble fps_avg = 0.0;
						gdouble pot_avg = 0.0;
						g_mutex_lock(&self->fps_mutex);
						guint64 now = now_us();
						if (self->fps_last_time > 0) {
							guint64 delta = now - self->fps_last_time;
							if (delta > 0) {
								gdouble inst = 1000000.0 / (gdouble)delta;
								if (self->fps_avg < 0.001)
									self->fps_avg = inst;
								else
									self->fps_avg += 0.1 * (inst - self->fps_avg);
								self->fps_instant = inst;
							}
						}
						self->fps_last_time = now;
						frame_count = self->frames_rendered;
						drop_count = (guint64)g_atomic_int_get(&self->frames_dropped);
						g_mutex_unlock(&self->fps_mutex);
						fps_avg = self->fps_avg;
						pot_avg = self->pot_avg;
						snprintf(hud_line,
						         sizeof(hud_line),
						         "ACT:%5.1f POT:%5.1f  DROP:%lu/%lu  %dx%d",
						         fps_avg,
						         pot_avg,
						         (unsigned long)drop_count,
						         (unsigned long)(drop_count + frame_count),
						         self->in_width,
						         self->in_height);
					}

					bool ok = render_frame(self, &rs, dup_fd, self->in_width, self->in_height, stride, uv_offset, self->show_hud ? hud_line : "");
					if (!ok)
						GST_WARNING_OBJECT(self, "render_frame failed");
					close(dup_fd);
					if (ok) {
						g_mutex_lock(&self->fps_mutex);
						self->frames_rendered++;
						g_mutex_unlock(&self->fps_mutex);
					}
				}
			}
		}
		gst_buffer_unref(buf);
	}

	// Cleanup GL/EGL — destroy GL resources WHILE context is still current
	destroy_gl_resources(&rs);
	egl_teardown(&rs);
	// Stash for deferred cleanup in finalize (after HIP/MIGraphX release)
	self->egl_display = rs.dpy;
	self->xcb_conn = rs.conn;
	self->xcb_win = rs.win;
	self->owns_window = !self->handle_set;
	GST_INFO_OBJECT(self, "render thread done");
	return nullptr;

fail:
	egl_teardown(&rs);
	if (rs.dpy != EGL_NO_DISPLAY)
		eglTerminate(rs.dpy);
	if (rs.conn)
		xcb_disconnect(rs.conn);
	g_atomic_int_set(&self->render_thread_running, 0);
	return nullptr;
}

// ─── GstVideoOverlay interface ────────────────────────────────────────
static void gst_magma_egl_video_sink_set_window_handle(GstVideoOverlay* overlay, guintptr handle) {
	GstMagmaEGLVideoSink* self = GST_MAGMA_EGL_VIDEO_SINK(overlay);
	self->window_handle = handle;
	self->handle_set = TRUE;
	GST_INFO_OBJECT(self, "external window handle: 0x%lx", (unsigned long)handle);
}

static void gst_magma_egl_video_sink_video_overlay_init(gpointer g_iface, gpointer iface_data) {
	(void)iface_data;
	GstVideoOverlayInterface* iface = (GstVideoOverlayInterface*)g_iface;
	iface->set_window_handle = gst_magma_egl_video_sink_set_window_handle;
}

// ─── Properties ───────────────────────────────────────────────────────
static void gst_magma_egl_video_sink_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
	GstMagmaEGLVideoSink* self = GST_MAGMA_EGL_VIDEO_SINK(object);
	switch (prop_id) {
	case PROP_VSYNC:
		self->vsync = g_value_get_boolean(value);
		break;
	case PROP_SHOW_HUD:
		self->show_hud = g_value_get_boolean(value);
		break;
	case PROP_WIN_W:
		self->win_width = g_value_get_int(value);
		self->win_size_explicit = TRUE;
		break;
	case PROP_WIN_H:
		self->win_height = g_value_get_int(value);
		self->win_size_explicit = TRUE;
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}
}

static void gst_magma_egl_video_sink_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
	GstMagmaEGLVideoSink* self = GST_MAGMA_EGL_VIDEO_SINK(object);
	switch (prop_id) {
	case PROP_VSYNC:
		g_value_set_boolean(value, self->vsync);
		break;
	case PROP_SHOW_HUD:
		g_value_set_boolean(value, self->show_hud);
		break;
	case PROP_WIN_W:
		g_value_set_int(value, self->win_width);
		break;
	case PROP_WIN_H:
		g_value_set_int(value, self->win_height);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}
}

// ─── GstBaseSink ──────────────────────────────────────────────────────
static gboolean gst_magma_egl_video_sink_start(GstBaseSink* sink) {
	GstMagmaEGLVideoSink* self = GST_MAGMA_EGL_VIDEO_SINK(sink);
	self->frame_queue = g_async_queue_new();
	g_atomic_int_set(&self->render_thread_running, 1);
	self->render_thread = g_thread_new("mgmegl-render", render_thread_func, self);
	if (!self->render_thread) {
		g_atomic_int_set(&self->render_thread_running, 0);
		g_async_queue_unref(self->frame_queue);
		self->frame_queue = nullptr;
		return FALSE;
	}
	return TRUE;
}

static gboolean gst_magma_egl_video_sink_stop(GstBaseSink* sink) {
	GstMagmaEGLVideoSink* self = GST_MAGMA_EGL_VIDEO_SINK(sink);
	if (self->render_thread) {
		g_atomic_int_set(&self->render_thread_running, 0);
		// Wake the render thread. g_async_queue_push() rejects NULL
		// (g_return_if_fail), so use a non-NULL sentinel address.
		g_async_queue_push(self->frame_queue, queue_wakeup_sentinel());
		g_thread_join(self->render_thread);
		self->render_thread = nullptr;
	}
	if (self->frame_queue) {
		// Drain and unref any queued frames
		gpointer p;
		while ((p = g_async_queue_try_pop(self->frame_queue))) {
			if (p != queue_wakeup_sentinel())
				gst_buffer_unref(GST_BUFFER(p));
		}
		g_async_queue_unref(self->frame_queue);
		self->frame_queue = nullptr;
	}
	return TRUE;
}

static gboolean gst_magma_egl_video_sink_set_caps(GstBaseSink* sink, GstCaps* caps) {
	GstMagmaEGLVideoSink* self = GST_MAGMA_EGL_VIDEO_SINK(sink);
	GstVideoInfo info;
	if (!gst_video_info_from_caps(&info, caps)) {
		GST_ERROR_OBJECT(self, "failed to parse caps");
		return FALSE;
	}
	self->in_width = GST_VIDEO_INFO_WIDTH(&info);
	self->in_height = GST_VIDEO_INFO_HEIGHT(&info);
	self->in_stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
	if (self->in_stride <= 0)
		self->in_stride = self->in_width;

	// The window already exists (created in start(), before caps arrive).
	// Don't silently reassign win_width/win_height — that desynchronises the
	// viewport from the real window. Ask the render thread to resize instead,
	// and only when the user didn't pick a size explicitly.
	if (!self->win_size_explicit && !self->handle_set) {
		g_atomic_int_set(&self->pending_resize_w, self->in_width);
		g_atomic_int_set(&self->pending_resize_h, self->in_height);
	}

	GST_INFO_OBJECT(self, "configured %dx%d stride=%d %s", self->in_width, self->in_height, self->in_stride, GST_VIDEO_INFO_NAME(&info));
	return TRUE;
}

static GstFlowReturn gst_magma_egl_video_sink_render(GstBaseSink* sink, GstBuffer* buf) {
	GstMagmaEGLVideoSink* self = GST_MAGMA_EGL_VIDEO_SINK(sink);

	if (!g_atomic_int_get(&self->render_thread_running) || !self->frame_queue)
		return GST_FLOW_FLUSHING;

	// Only accept DMABuf-backed buffers. gst_buffer_peek_memory() returns NULL
	// for empty (gap) buffers, and gst_is_dmabuf_memory() dereferences its
	// argument — so the NULL case must short-circuit here.
	GstMemory* mem = gst_buffer_get_size(buf) > 0 ? gst_buffer_peek_memory(buf, 0) : nullptr;
	if (!mem || !gst_is_dmabuf_memory(mem)) {
		GST_WARNING_OBJECT(self, "non-DMABuf buffer received, skipping");
		return GST_FLOW_OK;
	}

	// Update stride from current buffer's video meta
	GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
	if (vmeta && vmeta->stride[0] > 0)
		self->in_stride = (gint)vmeta->stride[0];

	// Push frame to render thread (ref'd). If queue is full (>= 2 entries),
	// drop the oldest pending to stay real-time.
	gst_buffer_ref(buf);
	if (g_async_queue_length(self->frame_queue) >= 2) {
		gpointer old = g_async_queue_try_pop(self->frame_queue);
		if (old) {
			gst_buffer_unref(GST_BUFFER(old));
			g_atomic_int_inc(&self->frames_dropped);
		}
	}
	g_async_queue_push(self->frame_queue, buf);

	// Potential FPS: frame arrival rate from upstream (independent of vsync)
	g_mutex_lock(&self->fps_mutex);
	guint64 now = now_us();
	if (self->pot_last_time > 0) {
		guint64 delta = now - self->pot_last_time;
		if (delta > 0) {
			gdouble inst = 1000000.0 / (gdouble)delta;
			if (self->pot_avg < 0.001)
				self->pot_avg = inst;
			else
				self->pot_avg += 0.1 * (inst - self->pot_avg);
			self->pot_instant = inst;
		}
	}
	self->pot_last_time = now;
	g_mutex_unlock(&self->fps_mutex);

	return GST_FLOW_OK;
}

// ─── init / finalize / class_init ─────────────────────────────────────
static void gst_magma_egl_video_sink_init(GstMagmaEGLVideoSink* self) {
	self->in_width = 0;
	self->in_height = 0;
	self->in_stride = 0;
	self->vsync = TRUE;
	self->show_hud = FALSE;
	self->win_width = 960;
	self->win_height = 540;
	self->win_size_explicit = FALSE;
	self->pending_resize_w = 0;
	self->pending_resize_h = 0;
	self->window_handle = 0;
	self->handle_set = FALSE;
	self->frame_queue = nullptr;
	g_atomic_int_set(&self->render_thread_running, 0);
	self->render_thread = nullptr;
	g_mutex_init(&self->fps_mutex);
	self->fps_last_time = 0;
	self->fps_instant = 0.0;
	self->fps_avg = 0.0;
	self->pot_last_time = 0;
	self->pot_instant = 0.0;
	self->pot_avg = 0.0;
	self->frames_rendered = 0;
	self->frames_dropped = 0;
	self->egl_display = EGL_NO_DISPLAY;
	self->xcb_conn = nullptr;
	self->xcb_win = 0;
	self->owns_window = FALSE;
}

static void gst_magma_egl_video_sink_finalize(GObject* object) {
	GstMagmaEGLVideoSink* self = GST_MAGMA_EGL_VIDEO_SINK(object);
	if (self->render_thread)
		gst_magma_egl_video_sink_stop(GST_BASE_SINK(self));
	// Deferred GPU resource cleanup — by now all pipeline elements
	// have released their HIP/MIGraphX contexts.
	if (self->egl_display != EGL_NO_DISPLAY) {
		eglTerminate(self->egl_display);
		self->egl_display = EGL_NO_DISPLAY;
	}
	if (self->xcb_conn) {
		// Only destroy windows we created ourselves; an app-provided
		// handle belongs to the caller.
		if (self->owns_window && self->xcb_win)
			xcb_destroy_window(self->xcb_conn, self->xcb_win);
		xcb_disconnect(self->xcb_conn);
		self->xcb_conn = nullptr;
	}
	g_mutex_clear(&self->fps_mutex);
	G_OBJECT_CLASS(gst_magma_egl_video_sink_parent_class)->finalize(object);
}

static void gst_magma_egl_video_sink_class_init(GstMagmaEGLVideoSinkClass* klass) {
	GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
	GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
	GstBaseSinkClass* sink_class = GST_BASE_SINK_CLASS(klass);

	gobject_class->set_property = gst_magma_egl_video_sink_set_property;
	gobject_class->get_property = gst_magma_egl_video_sink_get_property;
	gobject_class->finalize = gst_magma_egl_video_sink_finalize;

	/* NOTE: must not be called "sync" — GstBaseSink already installs that
	 * property, and a duplicate name is rejected with a GLib critical, which
	 * left vsync permanently forced on (capping presentation at the refresh
	 * rate). To run uncapped, set BOTH vsync=false and sync=false. */
	g_object_class_install_property(gobject_class,
	                                PROP_VSYNC,
	                                g_param_spec_boolean("vsync",
	                                                     "Sync to vblank",
	                                                     "Synchronise buffer swaps to vertical blank. Set false "
	                                                     "(together with the base sink's sync=false) to render "
	                                                     "as fast as the pipeline allows",
	                                                     TRUE,
	                                                     GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

	g_object_class_install_property(
	    gobject_class,
	    PROP_SHOW_HUD,
	    g_param_spec_boolean("show-hud", "Show HUD", "Overlay debug HUD (FPS, dropped frames, resolution)", FALSE, GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

	g_object_class_install_property(
	    gobject_class, PROP_WIN_W, g_param_spec_int("window-width", "Window width", "Width of the video window (default 960)", 64, 8192, 960, GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

	g_object_class_install_property(
	    gobject_class,
	    PROP_WIN_H,
	    g_param_spec_int("window-height", "Window height", "Height of the video window (default 540)", 64, 8192, 540, GParamFlags(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

	static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

	gst_element_class_add_static_pad_template(element_class, &sink_template);

	gst_element_class_set_static_metadata(element_class,
	                                      "Magma EGL Video Sink",
	                                      "Sink/Video",
	                                      "EGL-based X11 video sink with zero-copy dmabuf import, "
	                                      "desktop-window compatible (no DRM master needed)",
	                                      "Magma");

	sink_class->start = GST_DEBUG_FUNCPTR(gst_magma_egl_video_sink_start);
	sink_class->stop = GST_DEBUG_FUNCPTR(gst_magma_egl_video_sink_stop);
	sink_class->set_caps = GST_DEBUG_FUNCPTR(gst_magma_egl_video_sink_set_caps);
	sink_class->render = GST_DEBUG_FUNCPTR(gst_magma_egl_video_sink_render);

	GST_DEBUG_CATEGORY_INIT(mgmeglvideosink_debug, "mgmeglvideosink", 0, "Magma EGL Video Sink");
}

// ─── Plugin entry point ───────────────────────────────────────────────
static gboolean plugin_init(GstPlugin* plugin) {
	return gst_element_register(plugin, "mgmeglvideosink", GST_RANK_NONE, GST_TYPE_MAGMA_EGL_VIDEO_SINK);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, mgmeglvideosink, "Magma EGL Video Sink Plugin", plugin_init, "0.1.0", "LGPL", "magma", "https://imeguras.eu.org")
