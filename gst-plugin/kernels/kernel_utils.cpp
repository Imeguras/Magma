#include "kernel_utils.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cstdlib>

#ifdef _WIN32
#include <direct.h>
#define mkdir(p, m) _mkdir(p)
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ── minimal SHA-256 (FIPS 180-4) ────────────────────────────── */
#define SHA256_BLOCK_SIZE 64

struct Sha256Ctx {
    uint8_t data[SHA256_BLOCK_SIZE];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];
};

static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
static uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
static uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
static uint32_t ep0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
static uint32_t ep1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
static uint32_t sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
static uint32_t sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

static void sha256_transform(Sha256Ctx* ctx) {
    uint32_t m[64], a, b, c, d, e, f, g, h;
    for (int i = 0; i < 16; i++)
        m[i] = (ctx->data[i * 4] << 24) | (ctx->data[i * 4 + 1] << 16) |
               (ctx->data[i * 4 + 2] << 8) | ctx->data[i * 4 + 3];
    for (int i = 16; i < 64; i++)
        m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + ep1(e) + ch(e, f, g) + K256[i] + m[i];
        uint32_t t2 = ep0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(Sha256Ctx* ctx) {
    ctx->datalen = 0; ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
}

static void sha256_update(Sha256Ctx* ctx, const void* buf, size_t len) {
    auto* data = (const uint8_t*)buf;
    for (size_t i = 0; i < len; i++) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == SHA256_BLOCK_SIZE) {
            sha256_transform(ctx);
            ctx->bitlen += SHA256_BLOCK_SIZE * 8;
            ctx->datalen = 0;
        }
    }
}

static void sha256_final(Sha256Ctx* ctx, uint8_t hash[32]) {
    ctx->bitlen += ctx->datalen * 8;
    ctx->data[ctx->datalen++] = 0x80;
    if (ctx->datalen > 56) {
        while (ctx->datalen < SHA256_BLOCK_SIZE)
            ctx->data[ctx->datalen++] = 0;
        sha256_transform(ctx);
        ctx->datalen = 0;
    }
    while (ctx->datalen < 56)
        ctx->data[ctx->datalen++] = 0;
    ctx->data[56] = ctx->bitlen >> 56;
    ctx->data[57] = ctx->bitlen >> 48;
    ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[59] = ctx->bitlen >> 32;
    ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[61] = ctx->bitlen >> 16;
    ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[63] = ctx->bitlen;
    sha256_transform(ctx);
    for (int i = 0; i < 4; i++) {
        hash[i]      = ctx->state[0] >> (24 - i * 8);
        hash[4 + i]  = ctx->state[1] >> (24 - i * 8);
        hash[8 + i]  = ctx->state[2] >> (24 - i * 8);
        hash[12 + i] = ctx->state[3] >> (24 - i * 8);
        hash[16 + i] = ctx->state[4] >> (24 - i * 8);
        hash[20 + i] = ctx->state[5] >> (24 - i * 8);
        hash[24 + i] = ctx->state[6] >> (24 - i * 8);
        hash[28 + i] = ctx->state[7] >> (24 - i * 8);
    }
}

static void hash_to_hex(const uint8_t hash[32], char out[65]) {
    const char* hex = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[i * 2]     = hex[hash[i] >> 4];
        out[i * 2 + 1] = hex[hash[i] & 0xf];
    }
    out[64] = '\0';
}

/* ── file I/O helpers ─────────────────────────────────────────── */

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

static bool write_file(const char* path, const void* data, size_t len) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    size_t written = fwrite(data, 1, len, f);
    fclose(f);
    return written == len;
}

static bool read_file(const char* path, std::vector<char>* out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    out->resize(static_cast<size_t>(len));
    fread(out->data(), 1, out->size(), f);
    fclose(f);
    return true;
}

/* ── cache directory ──────────────────────────────────────────── */

static std::string get_cache_dir() {
    const char* env = getenv("MAGMA_KERNEL_CACHE_DIR");
    if (env && env[0])
        return env;
    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0]) {
        std::string dir = std::string(xdg) + "/magma/kernels";
        return dir;
    }
    const char* home = getenv("HOME");
    if (home && home[0])
        return std::string(home) + "/.cache/magma/kernels";
    return "/tmp/magma-kernel-cache";
}

static bool ensure_dir(const char* dir) {
    struct stat st;
    if (stat(dir, &st) == 0 && S_ISDIR(st.st_mode))
        return true;
    if (mkdir(dir, 0755) != 0) {
        // try parent
        std::string s(dir);
        auto slash = s.rfind('/');
        if (slash != std::string::npos) {
            std::string parent = s.substr(0, slash);
            if (ensure_dir(parent.c_str()))
                return mkdir(dir, 0755) == 0;
        }
        return false;
    }
    return true;
}

/* ── cache logic ──────────────────────────────────────────────── */

static bool load_from_cache(const std::string& cache_path, hipModule_t* module) {
    std::vector<char> code;
    if (!read_file(cache_path.c_str(), &code))
        return false;
    hipError_t e = hipModuleLoadData(module, code.data());
    if (e != hipSuccess) {
        fprintf(stderr, "kernel_utils: cache load %s failed: %s, recompiling\n",
                cache_path.c_str(), hipGetErrorString(e));
        return false;
    }
    return true;
}

static void save_to_cache(const std::string& cache_path, const void* code, size_t size) {
    // atomic write: write to temp, rename
    std::string tmp = cache_path + ".tmp";
    if (write_file(tmp.c_str(), code, size)) {
        rename(tmp.c_str(), cache_path.c_str());
    }
}

static std::string compute_cache_key(const std::string& source, const std::string& arch) {
    Sha256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, source.data(), source.size());
    sha256_update(&ctx, arch.data(), arch.size());
    uint8_t hash[32];
    sha256_final(&ctx, hash);
    char hex[65];
    hash_to_hex(hash, hex);
    return std::string(hex);
}

static std::string get_arch_string() {
    int dev;
    hipError_t e = hipGetDevice(&dev);
    if (e != hipSuccess) return "unknown";
    hipDeviceProp_t props;
    e = hipGetDeviceProperties(&props, dev);
    if (e != hipSuccess) return "unknown";
    std::string arch(props.gcnArchName);
    // strip sub-arch colon suffix: "gfx1100:xnack-" → "gfx1100"
    auto colon = arch.find(':');
    if (colon != std::string::npos)
        arch.resize(colon);
    return arch;
}

/* ── public API ───────────────────────────────────────────────── */

HipKernel compile_kernel(const char* hip_source_path, const char* entry_point,
                         const char* common_path) {
    HipKernel result = {nullptr, nullptr};

    std::string source;
    if (common_path) {
        source = load_file(common_path);
        if (source.empty()) {
            fprintf(stderr, "kernel_utils: failed to read common source %s\n", common_path);
            return result;
        }
        source += '\n';
    }

    std::string kernel_src = load_file(hip_source_path);
    if (kernel_src.empty()) {
        fprintf(stderr, "kernel_utils: failed to read %s\n", hip_source_path);
        return result;
    }
    source += kernel_src;

    std::string arch = get_arch_string();
    std::string key = compute_cache_key(source, arch);
    std::string cachedir = get_cache_dir();
    std::string cache_path = cachedir + "/" + key + ".co";

    // try loading from cache first
    if (load_from_cache(cache_path, &result.module)) {
        hipError_t e = hipModuleGetFunction(&result.func, result.module, entry_point);
        if (e == hipSuccess)
            return result;
        // function not found — corrupted cache, unload and recompile
        (void)hipModuleUnload(result.module);
        result.module = nullptr;
        result.func = nullptr;
        fprintf(stderr, "kernel_utils: entry point '%s' not in cached code object, recompiling\n",
                entry_point);
    }

    // JIT compile via hiprtc
    hiprtcProgram prog;
    hiprtcResult r = hiprtcCreateProgram(&prog, source.c_str(), hip_source_path, 0, nullptr, nullptr);
    if (r != HIPRTC_SUCCESS) {
        fprintf(stderr, "kernel_utils: hiprtcCreateProgram failed: %d\n", r);
        return result;
    }

    std::string arch_opt = std::string("--gpu-architecture=") + arch;
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
                arch.c_str(), log.c_str());
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

    if (ensure_dir(cachedir.c_str()))
        save_to_cache(cache_path, code.data(), code.size());

    hipError_t err = hipModuleLoadData(&result.module, code.data());
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
    }

    return result;
}

HipKernel compile_kernel_from_string(const char* source, size_t source_len,
                                     const char* entry_point,
                                     const char* name_for_log) {
    HipKernel result = {nullptr, nullptr};
    std::string src(source, source_len);

    std::string arch = get_arch_string();
    std::string key = compute_cache_key(src, arch);
    std::string cachedir = get_cache_dir();
    std::string cache_path = cachedir + "/" + key + ".co";

    const char* label = name_for_log ? name_for_log : "inline";

    // try loading from cache first
    if (load_from_cache(cache_path, &result.module)) {
        hipError_t e = hipModuleGetFunction(&result.func, result.module, entry_point);
        if (e == hipSuccess)
            return result;
        (void)hipModuleUnload(result.module);
        result.module = nullptr;
        result.func = nullptr;
        fprintf(stderr, "kernel_utils: entry point '%s' not in cached '%s', recompiling\n",
                entry_point, label);
    }

    hiprtcProgram prog;
    hiprtcResult r = hiprtcCreateProgram(&prog, src.c_str(), label, 0, nullptr, nullptr);
    if (r != HIPRTC_SUCCESS) {
        fprintf(stderr, "kernel_utils: hiprtcCreateProgram(%s) failed: %d\n", label, r);
        return result;
    }

    std::string arch_opt = std::string("--gpu-architecture=") + arch;
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
        fprintf(stderr, "kernel_utils: hiprtcCompileProgram(%s) failed (arch=%s): %s\n",
                label, arch.c_str(), log.c_str());
        hiprtcDestroyProgram(&prog);
        return result;
    }

    size_t code_size;
    r = hiprtcGetCodeSize(prog, &code_size);
    if (r != HIPRTC_SUCCESS) {
        fprintf(stderr, "kernel_utils: hiprtcGetCodeSize(%s) failed: %d\n", label, r);
        hiprtcDestroyProgram(&prog);
        return result;
    }

    std::vector<char> code(code_size);
    r = hiprtcGetCode(prog, code.data());
    if (r != HIPRTC_SUCCESS) {
        fprintf(stderr, "kernel_utils: hiprtcGetCode(%s) failed: %d\n", label, r);
        hiprtcDestroyProgram(&prog);
        return result;
    }

    hiprtcDestroyProgram(&prog);

    if (ensure_dir(cachedir.c_str()))
        save_to_cache(cache_path, code.data(), code.size());

    hipError_t err = hipModuleLoadData(&result.module, code.data());
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
    }

    return result;
}

void kernel_cache_clear() {
    std::string cachedir = get_cache_dir();
    std::string glob = cachedir + "/*.co";
    (void)glob;
#if !defined(_WIN32)
    std::string cmd = "rm -f " + cachedir + "/*.co 2>/dev/null";
    (void)!system(cmd.c_str());
#endif
}
