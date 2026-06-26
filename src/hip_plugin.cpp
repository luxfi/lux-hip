// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// HIP (AMD ROCm) backend plugin for lux-accel.
//
// This is the ROCm sibling of lux-cuda. HIP's runtime + module + hiprtc APIs
// are a near 1:1 mirror of the CUDA driver/runtime + NVRTC surface the CUDA
// plugin uses, so the two backends stay byte-for-byte parity-testable from the
// same dispatch harness. Where CUDA uses an explicit primary context
// (cuCtxCreate/cuCtxSetCurrent), this plugin uses the non-deprecated runtime
// device model (hipSetDevice) — the modern, version-stable ROCm idiom.
//
// Build via the lux-hip CMake target. When the ROCm/HIP toolkit is not found
// at configure time (e.g. on macOS, which has no ROCm), the plugin compiles as
// a stub that reports is_available()==0 so the host can pick another backend.
//
// NOTE: HIP has no runtime on Apple/macOS. The real path below compiles and
// runs only on an AMD GPU with ROCm installed (build hosts: evo / spark).

#include <lux/accel/backend_api.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(LUX_HAS_HIP) && LUX_HAS_HIP
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#endif

namespace {

// Core API provided by lux-accel
const lux_core_api_t* g_core_api = nullptr;

inline void log_error(const char* msg) {
    if (g_core_api && g_core_api->log_error) g_core_api->log_error(msg);
}

#if defined(LUX_HAS_HIP) && LUX_HAS_HIP

// Cached device count
int g_device_count = -1;

// Forward declaration (referenced before definition)
void hip_destroy_buffer(void* buffer);

// =============================================================================
// HIP wrappers
// =============================================================================

struct HIPBuffer {
    void* ptr = nullptr;
    size_t size = 0;
    uint32_t usage = 0;
    bool managed = false;
};

struct HIPKernel {
    hipFunction_t function = nullptr;
    hipModule_t module = nullptr;
    std::string name;
    dim3 workgroup_size = dim3(256, 1, 1);
    std::unordered_map<uint32_t, hipDeviceptr_t> buffer_args;
    std::unordered_map<uint32_t, std::vector<uint8_t>> byte_args;
};

struct HIPQueue {
    hipStream_t stream = nullptr;
    int device_index = 0;
};

struct HIPDevice {
    int index = 0;
    std::string name;
};

// =============================================================================
// Backend interface implementation
// =============================================================================

int hip_is_available() {
    if (hipInit(0) != hipSuccess) return 0;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) return 0;
    return count > 0 ? 1 : 0;
}

int hip_init() {
    if (hipInit(0) != hipSuccess) return 0;
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) return 0;
    g_device_count = count;
    return g_device_count > 0 ? 1 : 0;
}

void hip_shutdown() {
    g_device_count = -1;
}

int hip_get_device_count() {
    if (g_device_count < 0) {
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess) count = 0;
        g_device_count = count;
    }
    return g_device_count;
}

int hip_get_device_caps(int index, lux_device_caps_t* caps) {
    if (!caps || index < 0 || index >= hip_get_device_count()) return 0;

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, index) != hipSuccess) return 0;

    static thread_local std::string name_storage;
    name_storage = prop.name;

    caps->name = name_storage.c_str();
    caps->vendor = "AMD";
    caps->is_discrete = prop.integrated ? 0 : 1;
    caps->is_unified_memory = prop.integrated ? 1 : 0;
    caps->total_memory = prop.totalGlobalMem;
    caps->max_buffer_size = prop.totalGlobalMem;
    caps->max_workgroup_size = static_cast<uint32_t>(prop.maxThreadsPerBlock);
    caps->simd_width = static_cast<uint32_t>(prop.warpSize);  // wavefront size (32/64)
    caps->supports_fp16 = 1;
    caps->supports_subgroups = 1;
    return 1;
}

void* hip_create_device(int index) {
    if (index < 0 || index >= hip_get_device_count()) return nullptr;
    if (hipSetDevice(index) != hipSuccess) return nullptr;

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, index) != hipSuccess) return nullptr;

    auto* device = new HIPDevice();
    device->index = index;
    device->name = prop.name;
    return device;
}

void hip_destroy_device(void* device) {
    delete static_cast<HIPDevice*>(device);
}

void* hip_create_queue(void* device) {
    auto* dev = static_cast<HIPDevice*>(device);
    if (!dev) return nullptr;
    if (hipSetDevice(dev->index) != hipSuccess) return nullptr;

    hipStream_t stream;
    if (hipStreamCreate(&stream) != hipSuccess) return nullptr;

    auto* queue = new HIPQueue();
    queue->stream = stream;
    queue->device_index = dev->index;
    return queue;
}

void hip_destroy_queue(void* queue) {
    auto* q = static_cast<HIPQueue*>(queue);
    if (q) {
        if (q->stream) hipStreamDestroy(q->stream);
        delete q;
    }
}

int hip_queue_submit(void* queue) {
    // HIP streams auto-submit on enqueue, matching CUDA streams.
    (void)queue;
    return 1;
}

int hip_queue_wait(void* queue) {
    auto* q = static_cast<HIPQueue*>(queue);
    if (!q) return 0;
    hipSetDevice(q->device_index);
    return hipStreamSynchronize(q->stream) == hipSuccess ? 1 : 0;
}

void* hip_create_buffer(void* device, const lux_buffer_desc_t* desc) {
    auto* dev = static_cast<HIPDevice*>(device);
    if (!dev || !desc) return nullptr;
    if (hipSetDevice(dev->index) != hipSuccess) return nullptr;

    void* ptr = nullptr;
    bool managed = (desc->usage & LUX_BUFFER_USAGE_MAP_READ) ||
                   (desc->usage & LUX_BUFFER_USAGE_MAP_WRITE);

    if (managed) {
        if (hipMallocManaged(&ptr, desc->size) != hipSuccess) return nullptr;
    } else {
        if (hipMalloc(&ptr, desc->size) != hipSuccess) return nullptr;
    }

    auto* buf = new HIPBuffer();
    buf->ptr = ptr;
    buf->size = desc->size;
    buf->usage = desc->usage;
    buf->managed = managed;
    return buf;
}

void* hip_create_buffer_with_data(void* device, const lux_buffer_desc_t* desc, const void* data) {
    auto* dev = static_cast<HIPDevice*>(device);
    if (!dev || !desc || !data) return nullptr;

    void* buffer = hip_create_buffer(device, desc);
    if (!buffer) return nullptr;

    auto* buf = static_cast<HIPBuffer*>(buffer);
    hipSetDevice(dev->index);
    if (hipMemcpyHtoD(buf->ptr, const_cast<void*>(data), desc->size) != hipSuccess) {
        hip_destroy_buffer(buffer);
        return nullptr;
    }
    return buffer;
}

void hip_destroy_buffer(void* buffer) {
    auto* buf = static_cast<HIPBuffer*>(buffer);
    if (buf) {
        if (buf->ptr) hipFree(buf->ptr);
        delete buf;
    }
}

void* hip_map_buffer(void* buffer) {
    auto* buf = static_cast<HIPBuffer*>(buffer);
    if (!buf) return nullptr;
    if (buf->managed) return buf->ptr;  // managed memory is host-addressable
    return nullptr;                      // device-only buffer: not mappable
}

void hip_unmap_buffer(void* buffer) {
    auto* buf = static_cast<HIPBuffer*>(buffer);
    if (buf && buf->managed) hipDeviceSynchronize();
}

void* hip_create_kernel_from_source(void* device, const char* source, const char* entry_point) {
    auto* dev = static_cast<HIPDevice*>(device);
    if (!dev || !source || !entry_point) return nullptr;
    if (hipSetDevice(dev->index) != hipSuccess) return nullptr;

    hiprtcProgram prog;
    if (hiprtcCreateProgram(&prog, source, "kernel.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS) {
        log_error("hiprtcCreateProgram failed");
        return nullptr;
    }

    // Target the running device's GPU arch (e.g. "gfx90a"). This is the
    // canonical ROCm hiprtc option, mirroring NVRTC's --gpu-architecture.
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, dev->index);
    std::string arch = std::string("--gpu-architecture=") + prop.gcnArchName;
    const char* opts[] = {arch.c_str()};

    hiprtcResult cres = hiprtcCompileProgram(prog, 1, opts);
    if (cres != HIPRTC_SUCCESS) {
        size_t log_size = 0;
        hiprtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        if (log_size) hiprtcGetProgramLog(prog, log.data());
        hiprtcDestroyProgram(&prog);
        log_error(log.empty() ? hiprtcGetErrorString(cres) : log.c_str());
        return nullptr;
    }

    size_t code_size = 0;
    hiprtcGetCodeSize(prog, &code_size);
    std::vector<char> code(code_size);
    hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);

    hipModule_t module;
    if (hipModuleLoadData(&module, code.data()) != hipSuccess) {
        log_error("hipModuleLoadData failed");
        return nullptr;
    }

    hipFunction_t fn;
    if (hipModuleGetFunction(&fn, module, entry_point) != hipSuccess) {
        hipModuleUnload(module);
        log_error("Entry point not found");
        return nullptr;
    }

    auto* kernel = new HIPKernel();
    kernel->function = fn;
    kernel->module = module;
    kernel->name = entry_point;
    return kernel;
}

void* hip_create_kernel_from_bundle(void* device, const void* bundle_data,
                                    size_t bundle_size, const char* entry_point) {
    auto* dev = static_cast<HIPDevice*>(device);
    if (!dev || !bundle_data || !entry_point) return nullptr;
    (void)bundle_size;
    if (hipSetDevice(dev->index) != hipSuccess) return nullptr;

    // Bundle is a precompiled HIP code object (HSACO / fat binary).
    hipModule_t module;
    if (hipModuleLoadData(&module, bundle_data) != hipSuccess) {
        log_error("hipModuleLoadData failed");
        return nullptr;
    }

    hipFunction_t fn;
    if (hipModuleGetFunction(&fn, module, entry_point) != hipSuccess) {
        hipModuleUnload(module);
        log_error("Entry point not found");
        return nullptr;
    }

    auto* kernel = new HIPKernel();
    kernel->function = fn;
    kernel->module = module;
    kernel->name = entry_point;
    return kernel;
}

void hip_destroy_kernel(void* kernel) {
    auto* k = static_cast<HIPKernel*>(kernel);
    if (k) {
        if (k->module) hipModuleUnload(k->module);
        delete k;
    }
}

void hip_kernel_set_buffer(void* kernel, uint32_t index, void* buffer, size_t offset) {
    auto* k = static_cast<HIPKernel*>(kernel);
    auto* buf = static_cast<HIPBuffer*>(buffer);
    if (k && buf) {
        k->buffer_args[index] =
            static_cast<hipDeviceptr_t>(static_cast<char*>(buf->ptr) + offset);
    }
}

void hip_kernel_set_bytes(void* kernel, uint32_t index, const void* data, size_t size) {
    auto* k = static_cast<HIPKernel*>(kernel);
    if (k && data) {
        k->byte_args[index].assign(static_cast<const uint8_t*>(data),
                                   static_cast<const uint8_t*>(data) + size);
    }
}

void hip_kernel_set_workgroup_size(void* kernel, uint32_t x, uint32_t y, uint32_t z) {
    auto* k = static_cast<HIPKernel*>(kernel);
    if (k) k->workgroup_size = dim3(x, y, z);
}

int hip_dispatch(void* queue, void* kernel, const lux_dispatch_desc_t* desc) {
    auto* q = static_cast<HIPQueue*>(queue);
    auto* k = static_cast<HIPKernel*>(kernel);
    if (!q || !k || !desc) return 0;
    hipSetDevice(q->device_index);

    // Build the kernelParams array: each slot points to the argument value.
    size_t max_idx = 0;
    for (const auto& [idx, _] : k->buffer_args) max_idx = std::max(max_idx, (size_t)idx);
    for (const auto& [idx, _] : k->byte_args) max_idx = std::max(max_idx, (size_t)idx);

    std::vector<void*> args(k->buffer_args.empty() && k->byte_args.empty() ? 0 : max_idx + 1, nullptr);
    for (auto& [idx, ptr] : k->buffer_args) args[idx] = &ptr;
    for (auto& [idx, data] : k->byte_args) args[idx] = data.data();

    hipError_t err = hipModuleLaunchKernel(
        k->function,
        desc->grid_x, desc->grid_y, desc->grid_z,
        k->workgroup_size.x, k->workgroup_size.y, k->workgroup_size.z,
        0, q->stream,
        args.empty() ? nullptr : args.data(), nullptr);

    if (err != hipSuccess) {
        log_error(hipGetErrorString(err));
        return 0;
    }
    return 1;
}

int hip_copy_buffer(void* queue, void* src, size_t src_offset,
                    void* dst, size_t dst_offset, size_t size) {
    auto* q = static_cast<HIPQueue*>(queue);
    auto* src_buf = static_cast<HIPBuffer*>(src);
    auto* dst_buf = static_cast<HIPBuffer*>(dst);
    if (!q || !src_buf || !dst_buf) return 0;
    hipSetDevice(q->device_index);

    void* src_ptr = static_cast<char*>(src_buf->ptr) + src_offset;
    void* dst_ptr = static_cast<char*>(dst_buf->ptr) + dst_offset;
    return hipMemcpyDtoDAsync(static_cast<hipDeviceptr_t>(dst_ptr),
                              static_cast<hipDeviceptr_t>(src_ptr),
                              size, q->stream) == hipSuccess ? 1 : 0;
}

int hip_fill_buffer(void* queue, void* buffer, size_t offset, size_t size, uint8_t value) {
    auto* q = static_cast<HIPQueue*>(queue);
    auto* buf = static_cast<HIPBuffer*>(buffer);
    if (!q || !buf) return 0;
    hipSetDevice(q->device_index);

    void* ptr = static_cast<char*>(buf->ptr) + offset;
    return hipMemsetD8Async(static_cast<hipDeviceptr_t>(ptr), value, size, q->stream) == hipSuccess ? 1 : 0;
}

#else  // !LUX_HAS_HIP — stub path (compiles everywhere; is_available()==0)

int hip_is_available() { return 0; }
int hip_init() { return 0; }
void hip_shutdown() {}
int hip_get_device_count() { return 0; }
int hip_get_device_caps(int, lux_device_caps_t*) { return 0; }
void* hip_create_device(int) { return nullptr; }
void hip_destroy_device(void*) {}
void* hip_create_queue(void*) { return nullptr; }
void hip_destroy_queue(void*) {}
int hip_queue_submit(void*) { return 0; }
int hip_queue_wait(void*) { return 0; }
void* hip_create_buffer(void*, const lux_buffer_desc_t*) { return nullptr; }
void* hip_create_buffer_with_data(void*, const lux_buffer_desc_t*, const void*) { return nullptr; }
void hip_destroy_buffer(void*) {}
void* hip_map_buffer(void*) { return nullptr; }
void hip_unmap_buffer(void*) {}
void* hip_create_kernel_from_source(void*, const char*, const char*) { return nullptr; }
void* hip_create_kernel_from_bundle(void*, const void*, size_t, const char*) { return nullptr; }
void hip_destroy_kernel(void*) {}
void hip_kernel_set_buffer(void*, uint32_t, void*, size_t) {}
void hip_kernel_set_bytes(void*, uint32_t, const void*, size_t) {}
void hip_kernel_set_workgroup_size(void*, uint32_t, uint32_t, uint32_t) {}
int hip_dispatch(void*, void*, const lux_dispatch_desc_t*) { return 0; }
int hip_copy_buffer(void*, void*, size_t, void*, size_t, size_t) { return 0; }
int hip_fill_buffer(void*, void*, size_t, size_t, uint8_t) { return 0; }

#endif // LUX_HAS_HIP

// =============================================================================
// Backend interface structure (identical layout for real + stub paths)
// =============================================================================

static lux_backend_interface_t g_hip_interface = {
    .type = LUX_BACKEND_TYPE_HIP,
    .name = "hip",
    .api_version = LUX_BACKEND_API_VERSION,

    .is_available = hip_is_available,
    .init = hip_init,
    .shutdown = hip_shutdown,

    .get_device_count = hip_get_device_count,
    .get_device_caps = hip_get_device_caps,

    .create_device = hip_create_device,
    .destroy_device = hip_destroy_device,

    .create_queue = hip_create_queue,
    .destroy_queue = hip_destroy_queue,
    .queue_submit = hip_queue_submit,
    .queue_wait = hip_queue_wait,

    .create_buffer = hip_create_buffer,
    .create_buffer_with_data = hip_create_buffer_with_data,
    .destroy_buffer = hip_destroy_buffer,
    .map_buffer = hip_map_buffer,
    .unmap_buffer = hip_unmap_buffer,

    .create_kernel_from_source = hip_create_kernel_from_source,
    .create_kernel_from_bundle = hip_create_kernel_from_bundle,
    .destroy_kernel = hip_destroy_kernel,
    .kernel_set_buffer = hip_kernel_set_buffer,
    .kernel_set_bytes = hip_kernel_set_bytes,
    .kernel_set_workgroup_size = hip_kernel_set_workgroup_size,

    .dispatch = hip_dispatch,
    .copy_buffer = hip_copy_buffer,
    .fill_buffer = hip_fill_buffer,
    // .create_buffer_from_host_ptr: no such field in lux_backend_interface_t;
    // create_buffer_with_data is the public host-data path (matches lux-cuda).
};

} // anonymous namespace

// =============================================================================
// Plugin entry point
// =============================================================================

extern "C" {

LUX_PLUGIN_EXPORT lux_backend_interface_t* lux_backend_init(const lux_core_api_t* core_api) {
    g_core_api = core_api;
    if (core_api && core_api->log_info) {
        core_api->log_info("HIP backend plugin loaded");
    }
    return &g_hip_interface;
}

} // extern "C"
