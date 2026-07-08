#pragma once
#include <hip/hip_runtime.h>

inline hipStream_t magma_get_shared_hip_stream() {
    static hipStream_t stream = nullptr;
    if (!stream) {
        (void)hipStreamCreate(&stream);
    }
    return stream;
}
