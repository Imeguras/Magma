#pragma once

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

struct HipKernel {
    hipModule_t module;
    hipFunction_t func;
};

/**
 * @brief Compile a HIP kernel at runtime via hiprtc.
 * @param hip_source_path  Path to the main .hip kernel file
 * @param entry_point      Name of the __global__ function to extract
 * @param common_path      Optional path to a common/header file prepended to the source (nullptr to skip)
 */
HipKernel compile_kernel(const char* hip_source_path, const char* entry_point,
                         const char* common_path = nullptr);
