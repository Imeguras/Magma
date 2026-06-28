#include <gst/gst.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define TEST_ASSERT(cond, fmt, ...)                                                                                                                                                                    \
    do {                                                                                                                                                                                               \
        if (!(cond)) {                                                                                                                                                                                 \
            g_printerr("FAIL (%s:%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);                                                                                                                  \
            failures++;                                                                                                                                                                                \
        } else {                                                                                                                                                                                       \
            g_print("  PASS\n");                                                                                                                                                                       \
        }                                                                                                                                                                                              \
    } while (0)

static int failures = 0;

static GstPlugin* plugin_infer = NULL;
static GstPlugin* plugin_preproc = NULL;
static GstPlugin* plugin_convert = NULL;

static void load_all_plugins(void) {
    if (plugin_infer)
        return;

    const char* dir = TEST_PLUGIN_DIR;
    const char* names[] = {"mgminfer", "mgmpreproc", "mgmvideoconvert"};
    GstPlugin** plugs[] = {&plugin_infer, &plugin_preproc, &plugin_convert};

    for (int i = 0; i < 3; i++) {
        gchar path[1024];
        g_snprintf(path, sizeof(path), "%s/lib%s.so", dir, names[i]);
        GError* error = NULL;
        *plugs[i] = gst_plugin_load_file(path, &error);
        if (error) {
            g_printerr("FAIL: failed to load %s: %s\n", path, error->message);
            g_error_free(error);
            failures++;
        } else {
            g_print("  loaded %s\n", names[i]);
        }
    }
}

static void unload_all_plugins(void) {
    if (plugin_infer) {
        gst_object_unref(plugin_infer);
        plugin_infer = NULL;
    }
    if (plugin_preproc) {
        gst_object_unref(plugin_preproc);
        plugin_preproc = NULL;
    }
    if (plugin_convert) {
        gst_object_unref(plugin_convert);
        plugin_convert = NULL;
    }
}

static void test_mgminfer_create(void) {
    g_print("  test_mgminfer_create ...");
    GstElement* e = gst_element_factory_make("mgminfer", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgminfer");
    if (e)
        gst_object_unref(e);
}

static void test_mgminfer_defaults(void) {
    g_print("  test_mgminfer_defaults ...");
    GstElement* e = gst_element_factory_make("mgminfer", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgminfer");
    if (!e)
        return;

    gchar* model_path = NULL;
    g_object_get(e, "model-path", &model_path, NULL);
    TEST_ASSERT(model_path == NULL, "model-path should be NULL, got %s", model_path);
    g_free(model_path);

    guint interval;
    g_object_get(e, "inference-interval", &interval, NULL);
    TEST_ASSERT(interval == 1, "inference-interval should be 1, got %u", interval);

    gst_object_unref(e);
}

static void test_mgminfer_set_model_path(void) {
    g_print("  test_mgminfer_set_model_path ...");
    GstElement* e = gst_element_factory_make("mgminfer", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgminfer");
    if (!e)
        return;

    g_object_set(e, "model-path", "/some/model.onnx", NULL);
    gchar* model_path = NULL;
    g_object_get(e, "model-path", &model_path, NULL);
    TEST_ASSERT(g_strcmp0(model_path, "/some/model.onnx") == 0, "expected /some/model.onnx, got %s", model_path);
    g_free(model_path);
    gst_object_unref(e);
}

static void test_mgminfer_set_inference_interval(void) {
    g_print("  test_mgminfer_set_inference_interval ...");
    GstElement* e = gst_element_factory_make("mgminfer", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgminfer");
    if (!e)
        return;

    g_object_set(e, "inference-interval", 15u, NULL);
    guint interval;
    g_object_get(e, "inference-interval", &interval, NULL);
    TEST_ASSERT(interval == 15, "expected 15, got %u", interval);
    gst_object_unref(e);
}

static void test_mgminfer_metadata(void) {
    g_print("  test_mgminfer_metadata ...");
    GstElement* e = gst_element_factory_make("mgminfer", "myinfer");
    TEST_ASSERT(e != NULL, "failed to create mgminfer");
    if (!e)
        return;

    const gchar* name = gst_element_get_name(e);
    TEST_ASSERT(g_strcmp0(name, "myinfer") == 0, "expected name 'myinfer', got %s", name);
    gst_object_unref(e);
}

static void test_mgmpreproc_create(void) {
    g_print("  test_mgmpreproc_create ...");
    GstElement* e = gst_element_factory_make("mgmpreproc", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgmpreproc");
    if (e)
        gst_object_unref(e);
}

static void test_mgmpreproc_defaults(void) {
    g_print("  test_mgmpreproc_defaults ...");
    GstElement* e = gst_element_factory_make("mgmpreproc", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgmpreproc");
    if (!e)
        return;

    gint val;
    g_object_get(e, "net-width", &val, NULL);
    TEST_ASSERT(val == 224, "net-width should be 224, got %d", val);

    g_object_get(e, "net-height", &val, NULL);
    TEST_ASSERT(val == 224, "net-height should be 224, got %d", val);

    gfloat fval;
    g_object_get(e, "scale-factor", &fval, NULL);
    TEST_ASSERT(fabsf(fval - 1.0f / 255.0f) < 0.0001f, "scale-factor should be 1/255≈0.0039, got %f", fval);

    gst_object_unref(e);
}

static void test_mgmpreproc_set_net_width(void) {
    g_print("  test_mgmpreproc_set_net_width ...");
    GstElement* e = gst_element_factory_make("mgmpreproc", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgmpreproc");
    if (!e)
        return;

    g_object_set(e, "net-width", 640, NULL);
    gint val;
    g_object_get(e, "net-width", &val, NULL);
    TEST_ASSERT(val == 640, "expected 640, got %d", val);
    gst_object_unref(e);
}

static void test_mgmpreproc_set_net_height(void) {
    g_print("  test_mgmpreproc_set_net_height ...");
    GstElement* e = gst_element_factory_make("mgmpreproc", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgmpreproc");
    if (!e)
        return;

    g_object_set(e, "net-height", 480, NULL);
    gint val;
    g_object_get(e, "net-height", &val, NULL);
    TEST_ASSERT(val == 480, "expected 480, got %d", val);
    gst_object_unref(e);
}

static void test_mgmpreproc_set_scale_factor(void) {
    g_print("  test_mgmpreproc_set_scale_factor ...");
    GstElement* e = gst_element_factory_make("mgmpreproc", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgmpreproc");
    if (!e)
        return;

    g_object_set(e, "scale-factor", 0.0078125f, NULL);
    gfloat val;
    g_object_get(e, "scale-factor", &val, NULL);
    TEST_ASSERT(fabsf(val - 0.0078125f) < 0.0001f, "expected 0.0078125, got %f", val);
    gst_object_unref(e);
}

static void test_mgmvideoconvert_create(void) {
    g_print("  test_mgmvideoconvert_create ...");
    GstElement* e = gst_element_factory_make("mgmvideoconvert", NULL);
    TEST_ASSERT(e != NULL, "failed to create mgmvideoconvert");
    if (e)
        gst_object_unref(e);
}

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);

    load_all_plugins();

    g_print("--- mgminfer ---\n");
    test_mgminfer_create();
    test_mgminfer_defaults();
    test_mgminfer_set_model_path();
    test_mgminfer_set_inference_interval();
    test_mgminfer_metadata();

    g_print("--- mgmpreproc ---\n");
    test_mgmpreproc_create();
    test_mgmpreproc_defaults();
    test_mgmpreproc_set_net_width();
    test_mgmpreproc_set_net_height();
    test_mgmpreproc_set_scale_factor();

    g_print("--- mgmvideoconvert ---\n");
    test_mgmvideoconvert_create();

    unload_all_plugins();

    g_print("\n%d failure(s)\n", failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
