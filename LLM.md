# lux-hip — HIP (AMD ROCm) backend plugin for lux-accel

`lux-hip` (CMake project `lux-hip`, v0.1.x) is a **plugin** (shared library /
CMake `MODULE`, output `lux_hip.plugin`) that `lux-accel` loads at runtime via
`dlopen` + `lux_backend_init`. It links nothing from lux-accel — it only
consumes the `backend_api.h` header. It is the ROCm sibling of `lux-cuda`.

On systems without the ROCm/HIP toolkit it builds as a no-op stub whose
`is_available()` returns 0, so the host transparently picks another backend.

## Where things are

- `src/hip_plugin.cpp` — the whole backend. The real path (`#if LUX_HAS_HIP`)
  mirrors `lux-cuda` almost line-for-line: HIP runtime + module API + `hiprtc`
  map 1:1 onto CUDA runtime/driver + NVRTC, which is what keeps the two backends
  byte-for-byte parity-testable from the same harness. It uses the
  non-deprecated runtime device model (`hipSetDevice`) rather than explicit
  `hipCtx*` contexts.
- `test/test_load.cpp` — load/identity smoke test; runs a real HIP vector-add
  when ROCm hardware is present.
- `conanfile.py`, `CMakeLists.txt`, `.gitignore`.

## Hardware requirement (honest status)

HIP has **no runtime on Apple/macOS**. On this Mac the plugin compiles and is
exercised only via the **stub path** (load + self-identify as `type=HIP`,
`name="hip"`, `is_available()==0`). The real ROCm path compiles and runs only on
an **AMD GPU with ROCm installed** — that is a hardware limit, not skipped work.

To run the real backend (build hosts `evo` / `spark` with AMD + ROCm):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DLUXACCEL_INCLUDE_DIR=/path/to/lux-accel/include
cmake --build build -j
ctest --test-dir build --output-on-failure   # runs the HIP vector-add
```

CMake links `hip::host` (+ `hiprtc`) and defines `LUX_HAS_HIP=1` when
`find_package(hip)` resolves; otherwise it prints the stub notice.

## Registration

The backend type is registered in lux-accel's enums as `LUX_BACKEND_TYPE_HIP=4`
(`backend_api.h`), `LUX_BACKEND_HIP=4` (`c_api.h`), and
`lux::gpu::BackendType::HIP` (`lux-gpu/gpu.hpp`). The cross-backend conformance
harness (`lux-gpu-conformance`) lists it as one row in `kBackends[]`.
