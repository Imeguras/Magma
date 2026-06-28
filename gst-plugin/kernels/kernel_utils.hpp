#pragma once

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

struct HipKernel {
    hipModule_t module;
    hipFunction_t func;
};

HipKernel compile_kernel(const char* hip_source_path, const char* entry_point);
