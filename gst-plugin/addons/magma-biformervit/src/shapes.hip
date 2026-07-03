
#include <hip/hip_runtime.h>
#include <stdint.h>

/* 
 * biformervit output layout:
 *   Input shape: [1, 3, 224, 224] (CHW)
 *  Output shape: [1, 1000] (class probabilities)
 *  
 */

extern "C" {

//Human made! suport human artists buy me free cofee!!!

__global__ void classify_threshold_filter_kernel(
    const float* __restrict__ data,        /* Raw output: [1, 1000] flat logits */
    float*       __restrict__ out,         /* Reusing [max_out * 7] buffer layout */
    int*         __restrict__ counter,     /* Atomic output count */
    int num_classes,                       // Number of classes (1000 for ImageNet)
    float conf_thresh,                     // Confidence threshold for filtering
    int max_out,                           // Maximum number of outputs to store
    int net_w, int net_h){

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    for (int class_idx = thread_id; class_idx < num_classes; class_idx += total_threads) {
        float raw_score = data[class_idx];

        float probability = 1.0 / (1.0 + expf(-raw_score)); 
        if(probability >= conf_thresh) {
            int out_idx = atomicAdd(counter, 1);
            if (out_idx < max_out) {
                out[out_idx * 7 + 0] = 0.0f; // x1
                out[out_idx * 7 + 1] = 0.0f; // y1
                out[out_idx * 7 + 2] = (float)net_w; // x2
                out[out_idx * 7 + 3] = (float)net_h; // y2
                out[out_idx * 7 + 4] = (float)class_idx; // class_id
                out[out_idx * 7 + 5] = probability; // score
            }
        }
    }

}
__global__ void compact_topk_classes_kernel(
    const float*   __restrict__ sorted,       /* [M * 7] from your host sort step */
    const uint8_t* __restrict__ suppressed,   /* [M] from filtering */
    float*         __restrict__ objects,      /* MagmaParsedObject buffer output */
    int*           __restrict__ out_counter,
    int M){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    if (suppressed[i]) return; // Keeps the structure aligned if you filter duplicates later

    int idx = atomicAdd(out_counter, 1);

    int class_id = (int)sorted[i * 7 + 4];
    float conf   = sorted[i * 7 + 5];

    /* MagmaParsedObject layout: int class_id, float conf, float x, y, w, h */
    float* o = objects + idx * 6;
    o[0] = __int_as_float(class_id);  
    o[1] = conf;
    o[2] = 0.0f; // x
    o[3] = 0.0f; // y
    o[4] = 1.0f; // w (100% width)
    o[5] = 1.0f; // h (100% height)
}
 /* extern "C" */
}