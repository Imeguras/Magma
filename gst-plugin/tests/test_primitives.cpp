#include <gst/check/gstcheck.h>
#include <cstring>
#include "magma-primitives.h"
#include "magma-infer-meta.h"

GST_START_TEST(test_primitive_list_init_reset) {
	MagmaPrimitiveList pl;
	magma_primitive_list_init(&pl);
	ck_assert_int_eq(pl.total_primitives, 0);
	ck_assert(pl.rects == NULL);

	magma_primitive_list_add_rect(&pl, 10, 20, 100, 50, 0xFFFF0000, 0);
	ck_assert_int_eq(pl.total_primitives, 1);
	ck_assert_int_eq(pl.num_rects, 1);
	ck_assert_int_eq(pl.rects[0].x, 10);
	ck_assert_int_eq(pl.rects[0].y, 20);
	ck_assert_int_eq(pl.rects[0].w, 100);
	ck_assert_int_eq(pl.rects[0].h, 50);
	ck_assert_int_eq(pl.rects[0].color, 0xFFFF0000);
	ck_assert_int_eq(pl.rects[0].filled, 0);

	magma_primitive_list_reset(&pl);
	ck_assert_int_eq(pl.total_primitives, 0);
	ck_assert_int_eq(pl.num_rects, 0);
	/* buffers preserved for reuse */
	ck_assert(pl.rects != NULL);
	ck_assert_int_ge(pl.rects_cap, 1);

	magma_primitive_list_destroy(&pl);
	ck_assert(pl.rects == NULL);
}
GST_END_TEST

GST_START_TEST(test_primitive_list_all_types) {
	MagmaPrimitiveList pl;
	magma_primitive_list_init(&pl);

	int verts[] = {10, 20, 30, 40, 50, 60};
	magma_primitive_list_add_rect(&pl, 0, 0, 100, 100, 0xFF00FF00, 1);
	magma_primitive_list_add_polyline(&pl, verts, 3, 0xFF0000FF, 2);
	magma_primitive_list_add_polygon(&pl, verts, 3, 0xFF00FFFF, 0xFFFF0000);
	magma_primitive_list_add_point(&pl, 50, 50, 0xFFFF00FF, 5);
	magma_primitive_list_add_text(&pl, 10, 10, 0xFFFFFFFF, "hello", 5);
	magma_primitive_list_add_arrow(&pl, 0, 0, 100, 100, 0xFFFF0000, 2, 10);

	ck_assert_int_eq(pl.total_primitives, 6);
	ck_assert_int_eq(pl.num_rects, 1);
	ck_assert_int_eq(pl.num_polylines, 1);
	ck_assert_int_eq(pl.num_polygons, 1);
	ck_assert_int_eq(pl.num_points, 1);
	ck_assert_int_eq(pl.num_texts, 1);
	ck_assert_int_eq(pl.num_arrows, 1);

	/* verify polyline vertex arena */
	ck_assert_int_eq(pl.polylines[0].num_verts, 3);
	ck_assert_int_ge(pl.vertex_arena_used, 6);
	ck_assert_int_eq(pl.vertex_arena[0], 10);
	ck_assert_int_eq(pl.vertex_arena[5], 60);

	/* verify text content */
	ck_assert_str_eq(pl.texts[0].text, "hello");

	magma_primitive_list_destroy(&pl);
}
GST_END_TEST

GST_START_TEST(test_primitive_list_growth) {
	MagmaPrimitiveList pl;
	magma_primitive_list_init(&pl);

	/* Add many rects to force growth */
	for (int i = 0; i < 500; i++) {
		int r = magma_primitive_list_add_rect(&pl, i, i, 10, 10, 0xFF000000, 0);
		ck_assert_int_eq(r, 0);
	}
	ck_assert_int_eq(pl.total_primitives, 500);
	ck_assert_int_eq(pl.num_rects, 500);
	ck_assert_int_ge(pl.rects_cap, 500);

	magma_primitive_list_destroy(&pl);
}
GST_END_TEST

GST_START_TEST(test_primitive_list_add_text_truncation) {
	MagmaPrimitiveList pl;
	magma_primitive_list_init(&pl);

	const char* long_text = "this is a very long label that exceeds MAGMA_LABEL_MAX";
	magma_primitive_list_add_text(&pl, 0, 0, 0xFFFFFFFF, long_text, (int)strlen(long_text));
	ck_assert_int_eq(pl.num_texts, 1);
	ck_assert_int_eq((int)strlen(pl.texts[0].text), MAGMA_LABEL_MAX - 1);

	magma_primitive_list_destroy(&pl);
}
GST_END_TEST

GST_START_TEST(test_primitive_list_invalid_polyline) {
	MagmaPrimitiveList pl;
	magma_primitive_list_init(&pl);

	/* Single vertex → not a polyline */
	int verts[] = {10, 20};
	int r = magma_primitive_list_add_polyline(&pl, verts, 1, 0xFF000000, 1);
	ck_assert_int_eq(r, 0); /* returns 0 (no-op) */
	ck_assert_int_eq(pl.num_polylines, 0);

	magma_primitive_list_destroy(&pl);
}
GST_END_TEST

GST_START_TEST(test_default_converter_empty) {
	MagmaPrimitiveList out;
	magma_primitive_list_init(&out);

	MagmaToPrimitivesParams params{};
	params.type_id = "magma.detection.v1";
	params.source_width = 640;
	params.source_height = 480;
	params.roi_w = 640;
	params.roi_h = 480;

	int r = magma_default_detection_to_primitives(&params, &out);
	ck_assert_int_eq(r, 0);
	ck_assert_int_eq(out.total_primitives, 0);

	magma_primitive_list_destroy(&out);
}
GST_END_TEST

GST_START_TEST(test_default_converter_with_objects) {
	MagmaPrimitiveList out;
	magma_primitive_list_init(&out);

	/* Create a CPU data buffer with 2 detection objects */
	MagmaInferObjectGPU objs[2];
	objs[0].class_id = 0;
	objs[0].confidence = 0.95f;
	objs[0].x = 0.1f;
	objs[0].y = 0.2f;
	objs[0].width = 0.3f;
	objs[0].height = 0.4f;

	objs[1].class_id = 1;
	objs[1].confidence = 0.85f;
	objs[1].x = 0.5f;
	objs[1].y = 0.6f;
	objs[1].width = 0.2f;
	objs[1].height = 0.3f;

	MagmaToPrimitivesParams params{};
	params.type_id = "magma.detection.v1";
	params.source_width = 640;
	params.source_height = 480;
	params.roi_w = 640;
	params.roi_h = 480;
	params.cpu_data = objs;
	params.cpu_data_size = (int)sizeof(objs);

	int r = magma_default_detection_to_primitives(&params, &out);
	ck_assert_int_eq(r, 0);
	ck_assert_int_ge(out.total_primitives, 2); /* 2 rects + 2 texts */
	ck_assert_int_ge(out.num_rects, 2);
	ck_assert_int_ge(out.num_texts, 2);

	/* Verify coordinate remapping */
	ck_assert_int_eq(out.rects[0].x, 64); /* 0.1 * 640 */
	ck_assert_int_eq(out.rects[0].y, 96); /* 0.2 * 480 -> but uses roi_w=640... */
	/* Actually the default converter does:
	 * sx = roi_x + obj->x * (roi_w > 0 ? roi_w : source_width)
	 * But roi_x=0 and roi_w=640, source_width=640. So:
	 *   sx = 0 + 0.1 * 640 = 64 ✓
	 *   sy = 0 + 0.2 * 480 = 96 ✓ (but also divided by roi_h used as scale) */
	ck_assert_int_eq(out.rects[0].w, 192); /* 0.3 * 640 */
	ck_assert_int_eq(out.rects[0].h, 192); /* 0.4 * 480 */

	magma_primitive_list_destroy(&out);
}
GST_END_TEST

static Suite* primitives_suite(void) {
	Suite* s = suite_create("magma_primitives");
	TCase* tc = tcase_create("general");
	tcase_add_test(tc, test_primitive_list_init_reset);
	tcase_add_test(tc, test_primitive_list_all_types);
	tcase_add_test(tc, test_primitive_list_growth);
	tcase_add_test(tc, test_primitive_list_add_text_truncation);
	tcase_add_test(tc, test_primitive_list_invalid_polyline);
	tcase_add_test(tc, test_default_converter_empty);
	tcase_add_test(tc, test_default_converter_with_objects);
	suite_add_tcase(s, tc);
	return s;
}

int main(int argc, char* argv[]) {
	gst_init(&argc, &argv);
	Suite* s = primitives_suite();
	SRunner* sr = srunner_create(s);
	srunner_run_all(sr, CK_NORMAL);
	int nf = srunner_ntests_failed(sr);
	srunner_free(sr);
	return nf ? EXIT_FAILURE : EXIT_SUCCESS;
}
