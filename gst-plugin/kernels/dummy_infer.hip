struct InferObject {
    unsigned int class_id;
    float confidence;
    float x, y, width, height;
};

extern "C" __global__ void dummy_infer(InferObject* output) {
    output->class_id = 1;
    output->confidence = 1.0f;
    output->x = 0.5f;
    output->y = 0.5f;
    output->width = 1.0f;
    output->height = 1.0f;
}
