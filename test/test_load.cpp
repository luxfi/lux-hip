// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Smoke test: dlopen the plugin, resolve lux_backend_init, query interface,
// then on AMD/ROCm hardware actually create a device, allocate buffers, compile
// a HIP vector-add kernel via hiprtc, dispatch it and verify the result.
//
// On a host without ROCm (e.g. macOS) is_available() returns 0 and the test
// passes after the identity check — the real compute path needs AMD hardware.

#include <lux/accel/backend_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {

void log_info(const char* msg)  { std::fprintf(stderr, "[info]  %s\n", msg); }
void log_warn(const char* msg)  { std::fprintf(stderr, "[warn]  %s\n", msg); }
void log_error(const char* msg) { std::fprintf(stderr, "[error] %s\n", msg); }
void log_debug(const char* msg) { std::fprintf(stderr, "[debug] %s\n", msg); }

void* core_alloc(size_t n) { return std::malloc(n); }
void  core_free(void* p)   { std::free(p); }

const void* get_kernel_bundle(const char*, size_t* sz) { if (sz) *sz = 0; return nullptr; }
const char* get_kernel_source(const char*) { return nullptr; }

// HIP compute kernel: c[i] = a[i] + b[i]
const char* kAddKernel = R"HIP(
extern "C" __global__ void vec_add(const float* a, const float* b, float* c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
)HIP";

} // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "./lux_hip.plugin";

    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    auto init = reinterpret_cast<lux_backend_init_fn>(dlsym(handle, LUX_BACKEND_INIT_SYMBOL));
    if (!init) {
        std::fprintf(stderr, "dlsym(%s) failed: %s\n", LUX_BACKEND_INIT_SYMBOL, dlerror());
        dlclose(handle);
        return 1;
    }

    lux_core_api_t core = {};
    core.api_version       = LUX_BACKEND_API_VERSION;
    core.log_debug         = log_debug;
    core.log_info          = log_info;
    core.log_warn          = log_warn;
    core.log_error         = log_error;
    core.alloc             = core_alloc;
    core.free              = core_free;
    core.get_kernel_bundle = get_kernel_bundle;
    core.get_kernel_source = get_kernel_source;

    lux_backend_interface_t* iface = init(&core);
    if (!iface) { std::fprintf(stderr, "init returned null\n"); dlclose(handle); return 1; }

    if (iface->api_version != LUX_BACKEND_API_VERSION ||
        iface->type != LUX_BACKEND_TYPE_HIP ||
        !iface->name || std::strcmp(iface->name, "hip") != 0) {
        std::fprintf(stderr, "interface identity mismatch (type=%d name=%s)\n",
                     (int)iface->type, iface->name ? iface->name : "(null)");
        dlclose(handle); return 1;
    }

    int avail = iface->is_available ? iface->is_available() : 0;
    std::fprintf(stderr, "hip is_available -> %d\n", avail);
    if (!avail) {
        std::fprintf(stderr, "OK: lux-hip plugin loaded and self-identified "
                             "(no ROCm runtime here; compute path needs AMD hardware)\n");
        dlclose(handle);
        return 0;
    }

    if (!iface->init || !iface->init()) { std::fprintf(stderr, "init() failed\n"); dlclose(handle); return 1; }

    int nDev = iface->get_device_count();
    if (nDev <= 0) { std::fprintf(stderr, "no hip devices\n"); iface->shutdown(); dlclose(handle); return 1; }

    lux_device_caps_t caps = {};
    iface->get_device_caps(0, &caps);
    std::fprintf(stderr, "hip device 0: %s (vendor=%s, total_mem=%llu, simd=%u)\n",
                 caps.name ? caps.name : "?", caps.vendor ? caps.vendor : "?",
                 (unsigned long long)caps.total_memory, caps.simd_width);

    void* dev = iface->create_device(0);
    void* queue = dev ? iface->create_queue(dev) : nullptr;
    if (!dev || !queue) { std::fprintf(stderr, "device/queue creation failed\n"); dlclose(handle); return 1; }

    constexpr uint32_t N = 1024;
    float a[N], b[N];
    for (uint32_t i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(2 * i); }

    lux_buffer_desc_t desc_in = { N * sizeof(float),
        LUX_BUFFER_USAGE_STORAGE | LUX_BUFFER_USAGE_COPY_SRC, nullptr };
    lux_buffer_desc_t desc_out = { N * sizeof(float),
        LUX_BUFFER_USAGE_STORAGE | LUX_BUFFER_USAGE_MAP_READ, nullptr };

    void* bufA = iface->create_buffer_with_data(dev, &desc_in, a);
    void* bufB = iface->create_buffer_with_data(dev, &desc_in, b);
    void* bufC = iface->create_buffer(dev, &desc_out);
    void* kernel = iface->create_kernel_from_source(dev, kAddKernel, "vec_add");
    if (!bufA || !bufB || !bufC || !kernel) {
        std::fprintf(stderr, "buffer/kernel setup failed\n"); dlclose(handle); return 1;
    }

    iface->kernel_set_buffer(kernel, 0, bufA, 0);
    iface->kernel_set_buffer(kernel, 1, bufB, 0);
    iface->kernel_set_buffer(kernel, 2, bufC, 0);
    iface->kernel_set_bytes (kernel, 3, &N, sizeof(N));
    iface->kernel_set_workgroup_size(kernel, 64, 1, 1);

    lux_dispatch_desc_t disp = { (N + 63) / 64, 1, 1, 64, 1, 1 };
    if (!iface->dispatch(queue, kernel, &disp) || !iface->queue_wait(queue)) {
        std::fprintf(stderr, "dispatch/wait failed\n"); return 1;
    }

    const float* out = static_cast<const float*>(iface->map_buffer(bufC));
    int errors = 0;
    if (!out) { std::fprintf(stderr, "map_buffer failed\n"); return 1; }
    for (uint32_t i = 0; i < N; i++) if (out[i] != a[i] + b[i]) errors++;
    iface->unmap_buffer(bufC);

    iface->destroy_kernel(kernel);
    iface->destroy_buffer(bufA);
    iface->destroy_buffer(bufB);
    iface->destroy_buffer(bufC);
    iface->destroy_queue(queue);
    iface->destroy_device(dev);
    iface->shutdown();
    dlclose(handle);

    if (errors) { std::fprintf(stderr, "FAIL: %d/%u mismatches\n", errors, N); return 1; }
    std::fprintf(stderr, "OK: %u-element vector add via HIP verified\n", N);
    return 0;
}
