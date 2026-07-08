#pragma once

/**
 * @file magma-hip-stream.hpp
 * @brief Shared HIP stream utility.
 *
 * Provides a process-wide singleton HIP stream for plugins that
 * do not require their own stream context.
 */

#include <hip/hip_runtime.h>

/**
 * @brief Get the process-wide shared HIP stream.
 *
 * Lazily created on first call. Suitable for plugins that perform
 * infrequent GPU operations and don't need per-element stream
 * isolation.
 *
 * @return hipStream_t (never null after first call)
 */
inline hipStream_t magma_get_shared_hip_stream() {
    static hipStream_t stream = nullptr;
    if (!stream) {
        (void)hipStreamCreate(&stream);
    }
    return stream;
}
