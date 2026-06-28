#define GST_CHECK_DISABLE_ASSERT_OVERRIDES
#include <gst/check/gstcheck.h>
#include <gst/video/video.h>
#include <gst/allocators/gstdmabuf.h>
#include "magma-meta.h"
#include "mgmpreproc.hpp"

#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <xf86drm.h>
#include <gbm.h>
#include <hip/hip_runtime.h>

/* ---------- test dimensions ---------- */
#define SRC_W       64
#define SRC_H       48
#define NET_W       32
#define NET_H       32

/* ---------- helpers ---------- */

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

static int
create_dmabuf(int w, int h, int *out_stride)
{
    int drm_fd = open_drm();
    if (drm_fd < 0) return -1;

    struct gbm_device *gbm = gbm_create_device(drm_fd);
    if (!gbm) { close(drm_fd); return -1; }

    struct gbm_bo *bo = gbm_bo_create(gbm, w, h * 3 / 2, GBM_FORMAT_R8,
                                       GBM_BO_USE_RENDERING);
    if (!bo) { gbm_device_destroy(gbm); close(drm_fd); return -1; }

    *out_stride = gbm_bo_get_stride(bo);
    int fd = gbm_bo_get_fd(bo);

    gbm_bo_destroy(bo);
    gbm_device_destroy(gbm);
    close(drm_fd);
    return fd;
}

static gboolean
fill_nv12_gradient(int dmabuf_fd, int w, int h, int stride)
{
    gsize total = (gsize)stride * h * 3 / 2;

    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = dmabuf_fd;
    desc.size = total;

    hipExternalMemory_t ext_mem;
    if (hipImportExternalMemory(&ext_mem, &desc) != hipSuccess)
        return FALSE;

    hipExternalMemoryBufferDesc bdesc{};
    bdesc.offset = 0;
    bdesc.size = total;

    hipDeviceptr_t d_ptr;
    if (hipExternalMemoryGetMappedBuffer(&d_ptr, ext_mem, &bdesc) != hipSuccess) {
        (void)hipDestroyExternalMemory(ext_mem);
        return FALSE;
    }

    uint8_t *cpu = (uint8_t *)g_malloc0(total);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            cpu[y * stride + x] = (uint8_t)((float)y / h * 255.0f + 0.5f);
    for (int y = 0; y < h / 2; y++)
        for (int x = 0; x < w; x++)
            cpu[stride * h + y * stride + x] = 128;

    hipError_t err = hipMemcpy(d_ptr, cpu, total, hipMemcpyHostToDevice);
    g_free(cpu);
    (void)hipDestroyExternalMemory(ext_mem);

    return err == hipSuccess;
}

/* ---------- pad templates matching mgmpreproc ---------- */
static GstStaticPadTemplate test_src_tmpl = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

static GstStaticPadTemplate test_sink_tmpl = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

/* ---------- chain function for capture pad ---------- */

static GstBuffer *captured_buf = NULL;

static GstFlowReturn
capture_chain(GstPad *pad, GstObject *parent, GstBuffer *buf)
{
    if (captured_buf)
        gst_buffer_unref(captured_buf);
    captured_buf = gst_buffer_ref(buf);
    return GST_FLOW_OK;
}

/* ---------- tests ---------- */

GST_START_TEST(test_tensor_non_null)
{
    /* ── load plugin ── */
    GError *err = NULL;
    GstPlugin *plugin = gst_plugin_load_file(TEST_PLUGIN_DIR "/libmgmpreproc.so",
                                              &err);
    ck_assert_msg(plugin != NULL, "load libmgmpreproc.so: %s",
                  err ? err->message : "unknown");
    if (err) g_error_free(err);

    GstElement *e = gst_element_factory_make("mgmpreproc", "preproc");
    ck_assert(e != NULL);
    GstMagmaPreproc *self = (GstMagmaPreproc *)e;

    g_object_set(e, "net-width", NET_W, "net-height", NET_H,
                 "scale-factor", 1.0f / 255.0f, NULL);

    /* ── create DMABuf and fill NV12 ── */
    int stride;
    int fd = create_dmabuf(SRC_W, SRC_H, &stride);
    ck_assert_int_ge(fd, 0);

    gsize buf_size = (gsize)stride * SRC_H * 3 / 2;

    gboolean ok = fill_nv12_gradient(fd, SRC_W, SRC_H, stride);
    ck_assert(ok);

    /* ── build caps ── */
    gchar caps_str[256];
    g_snprintf(caps_str, sizeof(caps_str),
        "video/x-raw(memory:DMABuf),format=NV12,width=%d,height=%d",
        SRC_W, SRC_H);
    GstCaps *caps = gst_caps_from_string(caps_str);
    ck_assert(caps != NULL);

    /* ── set up test pads ── */
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

    /* ── PAUSED ── */
    gst_element_set_state(e, GST_STATE_PAUSED);

    {
        gchar *sid = gst_pad_create_stream_id(push_pad, e, NULL);
        ck_assert(sid != NULL);
        gst_pad_push_event(push_pad, gst_event_new_stream_start(sid));
        g_free(sid);
    }
    gst_pad_push_event(push_pad, gst_event_new_caps(caps));
    {
        GstSegment seg;
        gst_segment_init(&seg, GST_FORMAT_TIME);
        gst_pad_push_event(push_pad, gst_event_new_segment(&seg));
    }
    gst_caps_unref(caps);

    /* ── wrap fd in GstBuffer ── */
    GstAllocator *dma_alloc = gst_dmabuf_allocator_new();
    ck_assert(dma_alloc != NULL);
    GstMemory *dma_mem = gst_dmabuf_allocator_alloc(dma_alloc, fd, buf_size);
    gst_object_unref(dma_alloc);
    ck_assert(dma_mem != NULL);

    GstBuffer *buf = gst_buffer_new();
    gst_buffer_append_memory(buf, dma_mem);
    gst_buffer_add_video_meta_full(
        buf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12,
        SRC_W, SRC_H, 2,
        (gsize[]){0, (gsize)(stride * SRC_H)},
        (gint[]){stride, stride});

    /* ── push buffer ── */
    GstFlowReturn ret = gst_pad_push(push_pad, buf);
    ck_assert(ret == GST_FLOW_OK);

    /* ── prove the GPU pipeline was exercised ── */
    ck_assert(self->in_width == SRC_W && self->in_height == SRC_H);
    ck_assert(self->imported == TRUE);
    ck_assert(self->hip_stream != NULL);
    ck_assert(self->kernel_ready == TRUE);

    /* ── check tensor content via MagmaTensorMeta on output buffer ── */
    ck_assert(captured_buf != NULL);

    MagmaTensorMeta *meta = magma_buffer_get_tensor_meta(captured_buf);
    ck_assert_msg(meta != NULL, "output buffer has no MagmaTensorMeta");
    ck_assert_int_eq(meta->width, NET_W);
    ck_assert_int_eq(meta->height, NET_H);
    ck_assert_int_eq(meta->channels, 3);
    ck_assert(meta->tensor_mem != NULL);
    ck_assert(gst_is_dmabuf_memory(meta->tensor_mem));

    /* ── read tensor data back from DMABuf ── */
    int tensor_fd = gst_dmabuf_memory_get_fd(meta->tensor_mem);
    ck_assert_int_ge(tensor_fd, 0);

    gsize n_tensor = (gsize)NET_W * NET_H * 3;
    gsize tensor_bytes = n_tensor * sizeof(float);

    hipExternalMemoryHandleDesc tdesc{};
    tdesc.type = hipExternalMemoryHandleTypeOpaqueFd;
    tdesc.handle.fd = tensor_fd;
    tdesc.size = tensor_bytes;

    hipExternalMemory_t text_mem;
    hipError_t herr = hipImportExternalMemory(&text_mem, &tdesc);
    ck_assert_int_eq(herr, hipSuccess);

    hipExternalMemoryBufferDesc tbdesc{};
    tbdesc.offset = 0;
    tbdesc.size = tensor_bytes;

    hipDeviceptr_t d_tensor;
    herr = hipExternalMemoryGetMappedBuffer(&d_tensor, text_mem, &tbdesc);
    ck_assert_int_eq(herr, hipSuccess);

    float *host_tensor = (float *)g_malloc0(tensor_bytes);
    herr = hipMemcpy(host_tensor, d_tensor, tensor_bytes, hipMemcpyDeviceToHost);
    ck_assert_int_eq(herr, hipSuccess);

    (void)hipDestroyExternalMemory(text_mem);

    /* fail if tensor was never written */
    float sum = 0.0f;
    for (gsize i = 0; i < n_tensor; i++)
        sum += host_tensor[i];
    ck_assert_msg(sum > 0.0f, "tensor is all zeros — nv12_to_rgb_normalized kernel "
                  "did not write to tensor DMABuf");

    /* all values must be in [0, 1] */
    for (gsize i = 0; i < n_tensor; i++) {
        ck_assert_msg(host_tensor[i] >= 0.0f && host_tensor[i] <= 1.0f,
                      "host_tensor[%zu] = %g (expected [0, 1])", i, host_tensor[i]);
    }

    g_free(host_tensor);

    /* ── stop ── */
    gst_element_set_state(e, GST_STATE_NULL);

    gst_object_unref(push_pad);
    gst_object_unref(capture_pad);
    if (captured_buf)
        gst_buffer_unref(captured_buf);
    captured_buf = NULL;
    gst_object_unref(plugin);
    gst_object_unref(e);
}
GST_END_TEST

static Suite *
mgmpreproc_tensor_suite(void)
{
    Suite *s = suite_create("mgmpreproc_tensor");
    TCase *tc = tcase_create("general");
    tcase_add_test(tc, test_tensor_non_null);
    suite_add_tcase(s, tc);
    return s;
}

int
main(int argc, char *argv[])
{
    gst_init(&argc, &argv);
    Suite *s = mgmpreproc_tensor_suite();
    SRunner *sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    int nf = srunner_ntests_failed(sr);
    srunner_free(sr);
    return nf ? EXIT_FAILURE : EXIT_SUCCESS;
}
