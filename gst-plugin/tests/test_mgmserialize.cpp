#define GST_CHECK_DISABLE_ASSERT_OVERRIDES
#include <gst/check/gstcheck.h>
#include <gst/video/video.h>
#include "magma-infer-meta.h"
#include "magma-meta.h"

#include <cstdlib>
#include <cstring>
#include <string>

/* ---------- helpers ---------- */

static GstBuffer *
make_buffer_with_inference_meta(guint num_objects, guint sw, guint sh)
{
    GstBuffer *buf = gst_buffer_new();
    gst_buffer_add_video_meta_full(buf, GST_VIDEO_FRAME_FLAG_NONE,
        GST_VIDEO_FORMAT_NV12, sw, sh, 2,
        (gsize[]){0, (gsize)(sw * sh)},
        (gint[]){sw, sw});

    MagmaInferenceMeta *m = magma_buffer_add_inference_meta(buf, sw, sh);
    g_assert(m != NULL);

    if (num_objects > 0) {
        gsize bytes = num_objects * sizeof(MagmaInferObjectGPU);
        MagmaInferObjectGPU *objs = (MagmaInferObjectGPU *)g_malloc0(bytes);
        for (guint i = 0; i < num_objects; i++) {
            objs[i].class_id = i + 1;
            objs[i].confidence = 0.5f + i * 0.1f;
            objs[i].x = 0.1f * (i + 1);
            objs[i].y = 0.2f * (i + 1);
            objs[i].width = 0.3f;
            objs[i].height = 0.4f;
        }

        GstMemory *mem = gst_memory_new_wrapped(
            GST_MEMORY_FLAG_READONLY, objs, bytes, 0, bytes, objs, g_free);
        m->num_objects = num_objects;
        m->objects_gpu = mem;
    }

    return buf;
}

static int
count_json_objects(const gchar *json)
{
    /* crude count: count occurrences of "class_id" in the object array */
    const gchar *p = json;
    int count = 0;
    while ((p = strstr(p, "\"class_id\"")) != NULL) {
        count++;
        p++;
    }
    return count;
}

static gboolean
find_json_value(const gchar *json, const gchar *key, gchar *out, int out_size)
{
    gchar pattern[256];
    g_snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const gchar *p = strstr(json, pattern);
    if (!p) return FALSE;
    p = strchr(p, ':');
    if (!p) return FALSE;
    p++;
    while (*p == ' ') p++;
    int i = 0;
    while (*p && *p != ',' && *p != '}' && *p != ']' && i < out_size - 1) {
        out[i++] = *p++;
    }
    out[i] = '\0';
    return TRUE;
}

/* ---------- pad templates ---------- */
static GstStaticPadTemplate test_src_tmpl = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-raw(memory:DMABuf),format=(string)NV12"));

static GstStaticPadTemplate test_sink_tmpl = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS("application/x-magma-msg"));

/* ---------- capture ---------- */
static GPtrArray *captured = NULL;

static GstFlowReturn
capture_chain(GstPad *pad, GstObject *parent, GstBuffer *buf)
{
    g_ptr_array_add(captured, gst_buffer_ref(buf));
    return GST_FLOW_OK;
}

/* ---------- setup helper ---------- */
static GstElement *
setup_pipeline(GstElement *e, GstPad **push_pad, GstPad **capture_pad)
{
    *push_pad = gst_pad_new_from_static_template(&test_src_tmpl, "testsrc");
    gst_pad_set_active(*push_pad, TRUE);

    GstPad *ele_sink = gst_element_get_static_pad(e, "sink");
    g_assert(gst_pad_link(*push_pad, ele_sink) == GST_PAD_LINK_OK);
    gst_object_unref(ele_sink);

    *capture_pad = gst_pad_new_from_static_template(&test_sink_tmpl, "testcap");
    gst_pad_set_active(*capture_pad, TRUE);

    GstPad *ele_src = gst_element_get_static_pad(e, "src");
    g_assert(gst_pad_link(ele_src, *capture_pad) == GST_PAD_LINK_OK);
    gst_object_unref(ele_src);
    gst_pad_set_chain_function(*capture_pad, capture_chain);

    gst_element_set_state(e, GST_STATE_PAUSED);

    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:DMABuf),format=NV12,width=64,height=48");
    gchar *sid = gst_pad_create_stream_id(*push_pad, e, NULL);
    gst_pad_push_event(*push_pad, gst_event_new_stream_start(sid));
    g_free(sid);
    gst_pad_push_event(*push_pad, gst_event_new_caps(caps));
    GstSegment seg;
    gst_segment_init(&seg, GST_FORMAT_TIME);
    gst_pad_push_event(*push_pad, gst_event_new_segment(&seg));
    gst_caps_unref(caps);

    return e;
}

/* ---------- test: serialize JSON with 2 objects ---------- */
GST_START_TEST(test_serialize_json)
{
    GError *err = NULL;
    GstPlugin *plugin = gst_plugin_load_file(
        TEST_PLUGIN_DIR "/libmgmserialize.so", &err);
    ck_assert_msg(plugin != NULL, "load libmgmserialize.so: %s",
        err ? err->message : "unknown");
    if (err) g_error_free(err);

    GstElement *e = gst_element_factory_make("mgmserialize", NULL);
    ck_assert(e != NULL);

    GstPad *push_pad, *cap_pad;
    setup_pipeline(e, &push_pad, &cap_pad);

    GstBuffer *buf = make_buffer_with_inference_meta(2, 640, 480);
    GstFlowReturn ret = gst_pad_push(push_pad, buf);
    ck_assert_int_eq(ret, GST_FLOW_OK);

    ck_assert_int_eq(captured->len, 1);
    GstBuffer *out = (GstBuffer *)g_ptr_array_index(captured, 0);

    /* verify caps */
    GstCaps *outcaps = gst_pad_get_current_caps(cap_pad);
    ck_assert(outcaps != NULL);
    GstStructure *s = gst_caps_get_structure(outcaps, 0);
    ck_assert_str_eq(gst_structure_get_name(s), "application/x-magma-msg");
    gst_caps_unref(outcaps);

    /* read back JSON */
    GstMapInfo info;
    ck_assert(gst_buffer_map(out, &info, GST_MAP_READ));
    const gchar *json = (const gchar *)info.data;

    /* verify structure */
    ck_assert_str_eq(("" + std::to_string(count_json_objects(json))).c_str(), "2");
    gchar val[64];
    ck_assert(find_json_value(json, "source_width", val, sizeof(val)));
    ck_assert_str_eq(val, "640");

    gst_buffer_unmap(out, &info);

    gst_element_set_state(e, GST_STATE_NULL);
    gst_object_unref(push_pad);
    gst_object_unref(cap_pad);
    g_ptr_array_set_size(captured, 0);
    gst_object_unref(plugin);
    gst_object_unref(e);
}
GST_END_TEST

/* ---------- test: serialize JSON with 0 objects ---------- */
GST_START_TEST(test_serialize_json_empty)
{
    GError *err = NULL;
    GstPlugin *plugin = gst_plugin_load_file(
        TEST_PLUGIN_DIR "/libmgmserialize.so", &err);
    ck_assert(plugin != NULL);
    if (err) g_error_free(err);

    GstElement *e = gst_element_factory_make("mgmserialize", NULL);
    ck_assert(e != NULL);

    GstPad *push_pad, *cap_pad;
    setup_pipeline(e, &push_pad, &cap_pad);

    GstBuffer *buf = make_buffer_with_inference_meta(0, 640, 480);
    GstFlowReturn ret = gst_pad_push(push_pad, buf);
    ck_assert_int_eq(ret, GST_FLOW_OK);

    ck_assert_int_eq(captured->len, 1);
    GstBuffer *out = (GstBuffer *)g_ptr_array_index(captured, 0);

    GstMapInfo info;
    ck_assert(gst_buffer_map(out, &info, GST_MAP_READ));
    const gchar *json = (const gchar *)info.data;
    ck_assert_msg(count_json_objects(json) == 0, "expected 0 objects, got json: %s", json);
    gst_buffer_unmap(out, &info);

    gst_element_set_state(e, GST_STATE_NULL);
    gst_object_unref(push_pad);
    gst_object_unref(cap_pad);
    g_ptr_array_set_size(captured, 0);
    gst_object_unref(plugin);
    gst_object_unref(e);
}
GST_END_TEST

/* ---------- test: serialize JSON with no inference meta at all ---------- */
GST_START_TEST(test_serialize_no_meta)
{
    GError *err = NULL;
    GstPlugin *plugin = gst_plugin_load_file(
        TEST_PLUGIN_DIR "/libmgmserialize.so", &err);
    ck_assert(plugin != NULL);
    if (err) g_error_free(err);

    GstElement *e = gst_element_factory_make("mgmserialize", NULL);
    ck_assert(e != NULL);

    GstPad *push_pad, *cap_pad;
    setup_pipeline(e, &push_pad, &cap_pad);

    /* buffer without any inference meta */
    GstBuffer *buf = gst_buffer_new();
    gst_buffer_add_video_meta_full(buf, GST_VIDEO_FRAME_FLAG_NONE,
        GST_VIDEO_FORMAT_NV12, 64, 48, 2,
        (gsize[]){0, (gsize)(64 * 48)},
        (gint[]){64, 64});

    GstFlowReturn ret = gst_pad_push(push_pad, buf);
    ck_assert_int_eq(ret, GST_FLOW_OK);

    ck_assert_int_eq(captured->len, 1);
    GstBuffer *out = (GstBuffer *)g_ptr_array_index(captured, 0);

    GstMapInfo info;
    ck_assert(gst_buffer_map(out, &info, GST_MAP_READ));
    const gchar *json = (const gchar *)info.data;
    ck_assert_msg(count_json_objects(json) == 0, "expected 0 objects for no meta, got: %s", json);
    gst_buffer_unmap(out, &info);

    gst_element_set_state(e, GST_STATE_NULL);
    gst_object_unref(push_pad);
    gst_object_unref(cap_pad);
    g_ptr_array_set_size(captured, 0);
    gst_object_unref(plugin);
    gst_object_unref(e);
}
GST_END_TEST

/* ---------- suite ---------- */
static Suite *
mgmserialize_suite(void)
{
    Suite *s = suite_create("mgmserialize");
    TCase *tc = tcase_create("general");
    tcase_add_test(tc, test_serialize_json);
    tcase_add_test(tc, test_serialize_json_empty);
    tcase_add_test(tc, test_serialize_no_meta);
    suite_add_tcase(s, tc);
    return s;
}

int
main(int argc, char *argv[])
{
    captured = g_ptr_array_new();
    gst_init(&argc, &argv);
    Suite *s = mgmserialize_suite();
    SRunner *sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    int nf = srunner_ntests_failed(sr);
    srunner_free(sr);
    g_ptr_array_unref(captured);
    return nf ? EXIT_FAILURE : EXIT_SUCCESS;
}
