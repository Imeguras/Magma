#include "kernel_utils.hpp"

#include <cstdio>
#include <string>
#include <vector>

static std::string load_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string src(static_cast<size_t>(len), '\0');
    fread(&src[0], 1, src.size(), f);
    fclose(f);
    return src;
}

HipKernel compile_kernel(const char* hip_source_path, const char* entry_point) {
    HipKernel result = {nullptr, nullptr};

    std::string source = load_file(hip_source_path);
    if (source.empty()) {
        fprintf(stderr, "kernel_utils: failed to read %s\n", hip_source_path);
        return result;
    }

    hiprtcProgram prog;
    hiprtcResult r = hiprtcCreateProgram(&prog, source.c_str(), hip_source_path, 0, nullptr, nullptr);
    if (r != HIPRTC_SUCCESS) {
        fprintf(stderr, "kernel_utils: hiprtcCreateProgram failed: %d\n", r);
        return result;
    }

    int device_idx;
    hipError_t err = hipGetDevice(&device_idx);
    if (err != hipSuccess) {
        fprintf(stderr, "kernel_utils: hipGetDevice failed: %s\n", hipGetErrorString(err));
        hiprtcDestroyProgram(&prog);
        return result;
    }

    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, device_idx);
    if (err != hipSuccess) {
        fprintf(stderr, "kernel_utils: hipGetDeviceProperties failed: %s\n", hipGetErrorString(err));
        hiprtcDestroyProgram(&prog);
        return result;
    }

    std::string arch_str(props.gcnArchName);
    std::string arch_opt = std::string("--gpu-architecture=") + arch_str;
    const char* opts[] = {arch_opt.c_str()};

    r = hiprtcCompileProgram(prog, 1, opts);
    if (r != HIPRTC_SUCCESS) {
        size_t log_size;
        hiprtcGetProgramLogSize(prog, &log_size);
        std::string log;
        if (log_size > 0) {
            log.resize(log_size);
            hiprtcGetProgramLog(prog, &log[0]);
        }
        fprintf(stderr, "kernel_utils: hiprtcCompileProgram failed (arch=%s): %s\n",
                arch_str.c_str(), log.c_str());
        hiprtcDestroyProgram(&prog);
        return result;
    }

    size_t code_size;
    r = hiprtcGetCodeSize(prog, &code_size);
    if (r != HIPRTC_SUCCESS) {
        fprintf(stderr, "kernel_utils: hiprtcGetCodeSize failed: %d\n", r);
        hiprtcDestroyProgram(&prog);
        return result;
    }

    std::vector<char> code(code_size);
    r = hiprtcGetCode(prog, code.data());
    if (r != HIPRTC_SUCCESS) {
        fprintf(stderr, "kernel_utils: hiprtcGetCode failed: %d\n", r);
        hiprtcDestroyProgram(&prog);
        return result;
    }

    hiprtcDestroyProgram(&prog);

    err = hipModuleLoadData(&result.module, code.data());
    if (err != hipSuccess) {
        fprintf(stderr, "kernel_utils: hipModuleLoadData(%s) failed: %s\n",
                entry_point, hipGetErrorString(err));
        result.module = nullptr;
        return result;
    }

    err = hipModuleGetFunction(&result.func, result.module, entry_point);
    if (err != hipSuccess) {
        fprintf(stderr, "kernel_utils: hipModuleGetFunction(%s) failed: %s\n",
                entry_point, hipGetErrorString(err));
        (void)hipModuleUnload(result.module);
        result.module = nullptr;
        result.func = nullptr;
        return result;
    }

    return result;
}
