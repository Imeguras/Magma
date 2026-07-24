#pragma once

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <cstddef>

struct HipKernel {
    hipModule_t module;
    hipFunction_t func;
};

HipKernel compile_kernel(const char* hip_source_path, const char* entry_point,
                         const char* common_path = nullptr);

HipKernel compile_kernel_from_string(const char* source, size_t source_len,
                                     const char* entry_point,
                                     const char* name_for_log = nullptr);

void kernel_cache_clear();
