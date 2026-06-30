#include <gst/check/gstcheck.h>
#include "magma-infer-meta.h"

GST_START_TEST(test_meta_api_type_is_valid) {
    GType t = magma_inference_meta_api_get_type();
    ck_assert(t != 0);
    GType t2 = magma_inference_meta_api_get_type();
    ck_assert(t == t2);
}
GST_END_TEST

GST_START_TEST(test_meta_info_is_registered) {
    const GstMetaInfo* info = magma_inference_meta_get_info();
    ck_assert(info != NULL);
    const GstMetaInfo* info2 = magma_inference_meta_get_info();
    ck_assert(info == info2);
}
GST_END_TEST

GST_START_TEST(test_buffer_add_inference_meta) {
    GstBuffer* buf = gst_buffer_new();
    ck_assert(buf != NULL);

    MagmaInferenceMeta* m = magma_buffer_add_inference_meta(buf, 1920, 1080);
    ck_assert(m != NULL);
    ck_assert_int_eq(m->source_width, 1920);
    ck_assert_int_eq(m->source_height, 1080);
    ck_assert_int_eq(m->num_objects, 0);
    ck_assert(m->objects_gpu == NULL);
    ck_assert(m->output_tensors == NULL);

    gst_buffer_unref(buf);
}
GST_END_TEST

GST_START_TEST(test_buffer_get_inference_meta_found) {
    GstBuffer* buf = gst_buffer_new();
    magma_buffer_add_inference_meta(buf, 640, 480);
    MagmaInferenceMeta* m = magma_buffer_get_inference_meta(buf);
    ck_assert(m != NULL);
    ck_assert_int_eq(m->source_width, 640);
    ck_assert_int_eq(m->source_height, 480);
    gst_buffer_unref(buf);
}
GST_END_TEST

GST_START_TEST(test_buffer_get_inference_meta_not_found) {
    GstBuffer* buf = gst_buffer_new();
    MagmaInferenceMeta* m = magma_buffer_get_inference_meta(buf);
    ck_assert(m == NULL);
    gst_buffer_unref(buf);
}
GST_END_TEST

GST_START_TEST(test_meta_init_defaults) {
    GstBuffer* buf = gst_buffer_new();
    magma_buffer_add_inference_meta(buf, 800, 600);
    MagmaInferenceMeta* m = magma_buffer_get_inference_meta(buf);
    ck_assert(m != NULL);
    ck_assert_int_eq(m->num_objects, 0);
    ck_assert(m->objects_gpu == NULL);
    ck_assert(m->output_tensors == NULL);
    gst_buffer_unref(buf);
}
GST_END_TEST

GST_START_TEST(test_meta_transform_func_is_registered) {
    const GstMetaInfo* info = magma_inference_meta_get_info();
    ck_assert(info != NULL);
    ck_assert(info->transform_func != NULL);
}
GST_END_TEST

GST_START_TEST(test_meta_attached_only_once) {
    GstBuffer* buf = gst_buffer_new();
    magma_buffer_add_inference_meta(buf, 1, 1);
    magma_buffer_add_inference_meta(buf, 2, 2);

    MagmaInferenceMeta* m = magma_buffer_get_inference_meta(buf);
    ck_assert(m != NULL);
    ck_assert_int_eq(m->source_width, 1);
    ck_assert_int_eq(m->source_height, 1);

    gst_buffer_unref(buf);
}
GST_END_TEST

static Suite* inference_meta_suite(void) {
    Suite* s = suite_create("magma_inference_meta");
    TCase* tc = tcase_create("general");
    tcase_add_test(tc, test_meta_api_type_is_valid);
    tcase_add_test(tc, test_meta_info_is_registered);
    tcase_add_test(tc, test_buffer_add_inference_meta);
    tcase_add_test(tc, test_buffer_get_inference_meta_found);
    tcase_add_test(tc, test_buffer_get_inference_meta_not_found);
    tcase_add_test(tc, test_meta_init_defaults);
    tcase_add_test(tc, test_meta_transform_func_is_registered);
    tcase_add_test(tc, test_meta_attached_only_once);
    suite_add_tcase(s, tc);
    return s;
}

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);
    Suite* s = inference_meta_suite();
    SRunner* sr = srunner_create(s);
    srunner_run_all(sr, CK_NORMAL);
    int nf = srunner_ntests_failed(sr);
    srunner_free(sr);
    return nf ? EXIT_FAILURE : EXIT_SUCCESS;
}
