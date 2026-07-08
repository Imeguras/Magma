#pragma once

/**
 * @file kernel_utils.hpp
 * @brief HIP kernel compilation and dispatch utilities.
 *
 * Provides runtime JIT compilation of HIP kernels via hiprtc,
 * used by all Magma elements that run custom GPU kernels.
 */

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

/**
 * @brief A compiled HIP kernel ready for dispatch.
 */
struct HipKernel {
    hipModule_t module;  /**< HIP module handle */
    hipFunction_t func;  /**< Compiled kernel function handle */
};

/**
 * @brief Compile a HIP kernel at runtime via hiprtc.
 *
 * Reads the source file, optionally prepends a common header,
 * compiles with hiprtc, and extracts the entry point function.
 *
 * @param hip_source_path  Path to the main .hip kernel file
 * @param entry_point      Name of the __global__ function to extract
 * @param common_path      Optional path to a common/header file prepended
 *                         to the source (nullptr to skip)
 * @return HipKernel struct. If compilation fails, func is nullptr.
 */
HipKernel compile_kernel(const char* hip_source_path, const char* entry_point,
                         const char* common_path = nullptr);
