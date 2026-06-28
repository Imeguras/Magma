#define GST_CHECK_DISABLE_ASSERT_OVERRIDES
#include <gst/check/gstcheck.h>
#include <gst/video/video.h>
#include <gst/allocators/gstdmabuf.h>
#include "mgminfer.hpp"
#include "magma-meta.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <xf86drm.h>
#include <gbm.h>
#include <hip/hip_runtime.h>

/* ---------- function pointers loaded from .so via dlsym ---------- */

static MagmaTensorMeta* (*so_add_tensor_meta)(GstBuffer*, GstMemory*, gint, gint, gint) = NULL;
static GType (*so_inference_meta_api_type)(void) = NULL;

static MagmaInferenceMeta*
so_get_inference_meta(GstBuffer *buf)
{
    if (!so_inference_meta_api_type) return NULL;
    return (MagmaInferenceMeta*)gst_buffer_get_meta(buf, so_inference_meta_api_type());
}

static void
load_meta_funcs(void)
{
    if (so_add_tensor_meta)
        return;
    void *h = dlopen(TEST_PLUGIN_DIR "/libmgminfer.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen(TEST_PLUGIN_DIR "/libmgminfer.so", RTLD_LAZY | RTLD_NOW);
    g_assert(h != NULL);
    so_add_tensor_meta = (MagmaTensorMeta* (*)(GstBuffer*, GstMemory*, gint, gint, gint))
        dlsym(h, "magma_buffer_add_tensor_meta");
    so_inference_meta_api_type = (GType (*)(void))
        dlsym(h, "magma_inference_meta_api_get_type");
    g_assert(so_add_tensor_meta != NULL);
    g_assert(so_inference_meta_api_type != NULL);
    dlclose(h);
}

/* ---------- test dimensions ---------- */
#define NET_W       32
#define NET_H       32

/* ---------- helpers ---------- */

static GPtrArray *captured_bufs = NULL;

static int
open_drm(void)
{
    for (int i = 0; i < 64; i++) {
        char path[64];
        g_snprintf(path, sizeof(path), "/dev/dri/renderD%d", 128 + i);
        int fd = open(path, O_RDWR);
        if (fd < 0) continue;
        drmVersionPtr ver = drmGetVersion(fd);
        if (ver) { drmFreeVersion(ver); return fd; }
        close(fd);
    }
    return -1;
}

static GstMemory *
create_dmabuf_mem(gsize bytes)
{
    int drm_fd = open_drm();
    if (drm_fd < 0) return NULL;

    struct gbm_device *gbm = gbm_create_device(drm_fd);
    if (!gbm) { close(drm_fd); return NULL; }

    struct gbm_bo *bo = gbm_bo_create(gbm, bytes, 1, GBM_FORMAT_R8,
                                       GBM_BO_USE_RENDERING);
    if (!bo) { gbm_device_destroy(gbm); close(drm_fd); return NULL; }

    int fd = gbm_bo_get_fd(bo);
    gbm_bo_destroy(bo);
    gbm_device_destroy(gbm);
    close(drm_fd);

    if (fd < 0) return NULL;

    GstAllocator *dma = gst_dmabuf_allocator_new();
    GstMemory *mem = gst_dmabuf_allocator_alloc(dma, fd, bytes);
    gst_object_unref(dma);
    return mem;
}

static GstBuffer *
make_test_buffer(int w, int h, GstMemory *tensor_mem,
                 gint t_w, gint t_h, gint t_c)
{
    int drm_fd = open_drm();
    if (drm_fd < 0) return NULL;

    struct gbm_device *gbm = gbm_create_device(drm_fd);
    if (!gbm) { close(drm_fd); return NULL; }

    struct gbm_bo *bo = gbm_bo_create(gbm, w, h * 3 / 2, GBM_FORMAT_R8,
                                       GBM_BO_USE_RENDERING);
    if (!bo) { gbm_device_destroy(gbm); close(drm_fd); return NULL; }

    int nv12_fd = gbm_bo_get_fd(bo);
    int stride = gbm_bo_get_stride(bo);
    gbm_bo_destroy(bo);
    gbm_device_destroy(gbm);
    close(drm_fd);

    if (nv12_fd < 0) return NULL;

    gsize nv12_size = (gsize)stride * h * 3 / 2;
    GstAllocator *da = gst_dmabuf_allocator_new();
    GstMemory *nv12_mem = gst_dmabuf_allocator_alloc(da, nv12_fd, nv12_size);
    gst_object_unref(da);
    if (!nv12_mem) return NULL;

    GstBuffer *buf = gst_buffer_new();
    gst_buffer_append_memory(buf, nv12_mem);
    gst_buffer_add_video_meta_full(
        buf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12,
        w, h, 2,
        (gsize[]){0, (gsize)(stride * h)},
        (gint[]){stride, stride});

    if (tensor_mem)
        so_add_tensor_meta(buf, tensor_mem, t_w, t_h, t_c);

    return buf;
}

/* ---------- pad templates ---------- */
static GstStaticPadTemplate test_src_tmpl = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

static GstStaticPadTemplate test_sink_tmpl = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

static GstFlowReturn
capture_chain(GstPad *pad, GstObject *parent, GstBuffer *buf)
{
    g_ptr_array_add(captured_bufs, gst_buffer_ref(buf));
    return GST_FLOW_OK;
}

/* ---------- tests ---------- */

GST_START_TEST(test_attaches_meta)
{
    load_meta_funcs();

    GError *err = NULL;
    GstPlugin *plugin = gst_plugin_load_file(TEST_PLUGIN_DIR "/libmgminfer.so", &err);
    ck_assert_msg(plugin != NULL, "load libmgminfer.so: %s",
                  err ? err->message : "unknown");
    if (err) g_error_free(err);

    GstElement *e = gst_element_factory_make("mgminfer", "infer");
    ck_assert(e != NULL);
    g_object_set(e, "inference-interval", 1u, NULL);

    gsize tensor_bytes = (gsize)NET_W * NET_H * 3 * sizeof(float);
    GstMemory *tensor_mem = create_dmabuf_mem(tensor_bytes);
    ck_assert(tensor_mem != NULL);

    GstPad *push_pad = gst_pad_new_from_static_template(&test_src_tmpl, "testsrc");
    ck_assert(push_pad != NULL);
    gst_pad_set_active(push_pad, TRUE);

    GstPad *ele_sink = gst_element_get_static_pad(e, "sink");
    ck_assert(ele_sink != NULL);
    ck_assert(gst_pad_link(push_pad, ele_sink) == GST_PAD_LINK_OK);
    gst_object_unref(ele_sink);

    GstPad *capture_pad = gst_pad_new_from_static_template(&test_sink_tmpl, "testcap");
    ck_assert(capture_pad != NULL);
    gst_pad_set_active(capture_pad, TRUE);

    GstPad *ele_src = gst_element_get_static_pad(e, "src");
    ck_assert(ele_src != NULL);
    ck_assert(gst_pad_link(ele_src, capture_pad) == GST_PAD_LINK_OK);
    gst_object_unref(ele_src);
    gst_pad_set_chain_function(capture_pad, capture_chain);

    gst_element_set_state(e, GST_STATE_PAUSED);

    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:DMABuf),format=NV12,width=64,height=48");
    gchar *sid = gst_pad_create_stream_id(push_pad, e, NULL);
    ck_assert(gst_pad_push_event(push_pad, gst_event_new_stream_start(sid)));
    g_free(sid);
    ck_assert(gst_pad_push_event(push_pad, gst_event_new_caps(caps)));
    GstSegment seg;
    gst_segment_init(&seg, GST_FORMAT_TIME);
    ck_assert(gst_pad_push_event(push_pad, gst_event_new_segment(&seg)));
    gst_caps_unref(caps);

    GstBuffer *buf = make_test_buffer(64, 48, tensor_mem, NET_W, NET_H, 3);
    ck_assert(buf != NULL);

    GstFlowReturn ret = gst_pad_push(push_pad, buf);
    ck_assert_int_eq(ret, GST_FLOW_OK);

    /* verify MagmaInferenceMeta on output */
    ck_assert_int_eq(captured_bufs->len, 1);
    GstBuffer *out = (GstBuffer *)g_ptr_array_index(captured_bufs, 0);
    MagmaInferenceMeta *m = so_get_inference_meta(out);
    ck_assert_msg(m != NULL, "output buffer has no MagmaInferenceMeta");
    ck_assert_int_eq(m->num_objects, 1);
    ck_assert(m->objects_gpu != NULL);
    ck_assert(gst_is_dmabuf_memory(m->objects_gpu));

    /* read back objects_gpu DMABuf */
    int obj_fd = gst_dmabuf_memory_get_fd(m->objects_gpu);
    ck_assert_int_ge(obj_fd, 0);

    gsize obj_bytes = (gsize)m->num_objects * sizeof(MagmaInferObjectGPU);

    hipExternalMemoryHandleDesc odesc{};
    odesc.type = hipExternalMemoryHandleTypeOpaqueFd;
    odesc.handle.fd = obj_fd;
    odesc.size = obj_bytes;

    hipExternalMemory_t oext;
    ck_assert_int_eq(hipImportExternalMemory(&oext, &odesc), hipSuccess);

    hipExternalMemoryBufferDesc obdesc{};
    obdesc.offset = 0;
    obdesc.size = obj_bytes;

    hipDeviceptr_t d_obj;
    ck_assert_int_eq(hipExternalMemoryGetMappedBuffer(&d_obj, oext, &obdesc), hipSuccess);

    MagmaInferObjectGPU host_obj;
    ck_assert_int_eq(hipMemcpy(&host_obj, d_obj, obj_bytes, hipMemcpyDeviceToHost), hipSuccess);
    (void)hipDestroyExternalMemory(oext);

    ck_assert_int_eq(host_obj.class_id, 1);
    ck_assert_msg(fabsf(host_obj.confidence - 1.0f) < 0.0001f,
                  "confidence = %g (expected 1.0)", host_obj.confidence);
    ck_assert_msg(fabsf(host_obj.x - 0.5f) < 0.0001f, "x = %g", host_obj.x);
    ck_assert_msg(fabsf(host_obj.y - 0.5f) < 0.0001f, "y = %g", host_obj.y);

    gst_element_set_state(e, GST_STATE_NULL);
    gst_object_unref(push_pad);
    gst_object_unref(capture_pad);
    g_ptr_array_set_size(captured_bufs, 0);
    gst_memory_unref(tensor_mem);
    gst_object_unref(plugin);
    gst_object_unref(e);
}
GST_END_TEST

GST_START_TEST(test_no_tensor_meta)
{
    load_meta_funcs();

    GError *err = NULL;
    GstPlugin *plugin = gst_plugin_load_file(TEST_PLUGIN_DIR "/libmgminfer.so", &err);
    ck_assert(plugin != NULL);
    if (err) g_error_free(err);

    GstElement *e = gst_element_factory_make("mgminfer", NULL);
    ck_assert(e != NULL);
    g_object_set(e, "inference-interval", 1u, NULL);

    GstPad *push_pad = gst_pad_new_from_static_template(&test_src_tmpl, "testsrc");
    gst_pad_set_active(push_pad, TRUE);
    GstPad *ele_sink = gst_element_get_static_pad(e, "sink");
    ck_assert(gst_pad_link(push_pad, ele_sink) == GST_PAD_LINK_OK);
    gst_object_unref(ele_sink);

    GstPad *capture_pad = gst_pad_new_from_static_template(&test_sink_tmpl, "testcap");
    gst_pad_set_active(capture_pad, TRUE);
    GstPad *ele_src = gst_element_get_static_pad(e, "src");
    ck_assert(gst_pad_link(ele_src, capture_pad) == GST_PAD_LINK_OK);
    gst_object_unref(ele_src);
    gst_pad_set_chain_function(capture_pad, capture_chain);

    gst_element_set_state(e, GST_STATE_PAUSED);

    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:DMABuf),format=NV12,width=64,height=48");
    gchar *sid = gst_pad_create_stream_id(push_pad, e, NULL);
    gst_pad_push_event(push_pad, gst_event_new_stream_start(sid));
    g_free(sid);
    gst_pad_push_event(push_pad, gst_event_new_caps(caps));
    GstSegment seg;
    gst_segment_init(&seg, GST_FORMAT_TIME);
    gst_pad_push_event(push_pad, gst_event_new_segment(&seg));
    gst_caps_unref(caps);

    GstBuffer *buf = make_test_buffer(64, 48, NULL, 0, 0, 0);
    ck_assert(buf != NULL);

    GstFlowReturn ret = gst_pad_push(push_pad, buf);
    ck_assert_int_eq(ret, GST_FLOW_OK);

    ck_assert_int_eq(captured_bufs->len, 1);
    GstBuffer *out = (GstBuffer *)g_ptr_array_index(captured_bufs, 0);
    MagmaInferenceMeta *m = so_get_inference_meta(out);
    ck_assert_msg(m == NULL, "buffer without tensor meta must not get inference meta");

    gst_element_set_state(e, GST_STATE_NULL);
    gst_object_unref(push_pad);
    gst_object_unref(capture_pad);
    g_ptr_array_set_size(captured_bufs, 0);
    gst_object_unref(plugin);
    gst_object_unref(e);
}
GST_END_TEST

GST_START_TEST(test_infer_interval)
{
    load_meta_funcs();

    GError *err = NULL;
    GstPlugin *plugin = gst_plugin_load_file(TEST_PLUGIN_DIR "/libmgminfer.so", &err);
    ck_assert(plugin != NULL);
    if (err) g_error_free(err);

    GstElement *e = gst_element_factory_make("mgminfer", NULL);
    ck_assert(e != NULL);
    g_object_set(e, "inference-interval", 3u, NULL);

    gsize tensor_bytes = (gsize)NET_W * NET_H * 3 * sizeof(float);
    GstMemory *tensor_mem = create_dmabuf_mem(tensor_bytes);
    ck_assert(tensor_mem != NULL);

    GstPad *push_pad = gst_pad_new_from_static_template(&test_src_tmpl, "testsrc");
    gst_pad_set_active(push_pad, TRUE);
    GstPad *ele_sink = gst_element_get_static_pad(e, "sink");
    ck_assert(gst_pad_link(push_pad, ele_sink) == GST_PAD_LINK_OK);
    gst_object_unref(ele_sink);

    GstPad *capture_pad = gst_pad_new_from_static_template(&test_sink_tmpl, "testcap");
    gst_pad_set_active(capture_pad, TRUE);
    GstPad *ele_src = gst_element_get_static_pad(e, "src");
    ck_assert(gst_pad_link(ele_src, capture_pad) == GST_PAD_LINK_OK);
    gst_object_unref(ele_src);
    gst_pad_set_chain_function(capture_pad, capture_chain);

    gst_element_set_state(e, GST_STATE_PAUSED);

    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:DMABuf),format=NV12,width=64,height=48");
    gchar *sid = gst_pad_create_stream_id(push_pad, e, NULL);
    gst_pad_push_event(push_pad, gst_event_new_stream_start(sid));
    g_free(sid);
    gst_pad_push_event(push_pad, gst_event_new_caps(caps));
    GstSegment seg;
    gst_segment_init(&seg, GST_FORMAT_TIME);
    gst_pad_push_event(push_pad, gst_event_new_segment(&seg));
    gst_caps_unref(caps);

    for (int i = 0; i < 5; i++) {
        GstBuffer *b = make_test_buffer(64, 48, tensor_mem, NET_W, NET_H, 3);
        ck_assert(b != NULL);
        GstFlowReturn r = gst_pad_push(push_pad, b);
        ck_assert_int_eq(r, GST_FLOW_OK);
    }

    ck_assert_int_eq(captured_bufs->len, 5);

    /* interval=3: frames 0 and 3 should have meta; 1,2,4 should not */
    for (guint i = 0; i < 5; i++) {
        GstBuffer *b = (GstBuffer *)g_ptr_array_index(captured_bufs, i);
        MagmaInferenceMeta *m = so_get_inference_meta(b);
        if (i == 0 || i == 3) {
            ck_assert_msg(m != NULL, "frame %u should have inference meta", i);
        } else {
            ck_assert_msg(m == NULL, "frame %u should NOT have inference meta", i);
        }
    }

    gst_element_set_state(e, GST_STATE_NULL);
    gst_object_unref(push_pad);
    gst_object_unref(capture_pad);
    g_ptr_array_set_size(captured_bufs, 0);
    gst_memory_unref(tensor_mem);
    gst_object_unref(plugin);
    gst_object_unref(e);
}
GST_END_TEST

/* ---------- suite ---------- */
static Suite *
mgminfer_pipe_suite(void)
{
    Suite *s = suite_create("mgminfer_tensor_pipe");
    TCase *tc = tcase_create("general");
    tcase_add_test(tc, test_attaches_meta);
    tcase_add_test(tc, test_no_tensor_meta);
    tcase_add_test(tc, test_infer_interval);
    suite_add_tcase(s, tc);
    return s;
}

int
main(int argc, char *argv[])
{
    captured_bufs = g_ptr_array_new();
    gst_init(&argc, &argv);
    Suite *s = mgminfer_pipe_suite();
    SRunner *sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    int nf = srunner_ntests_failed(sr);
    srunner_free(sr);
    g_ptr_array_unref(captured_bufs);
    return nf ? EXIT_FAILURE : EXIT_SUCCESS;
}
