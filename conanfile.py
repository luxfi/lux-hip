from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout


class LuxHip(ConanFile):
    name = "lux-hip"
    version = "0.1.0"
    license = "BSD-3-Clause"
    description = "HIP (AMD ROCm) backend plugin for lux-accel"
    package_type = "shared-library"  # Plugin - always shared
    settings = "os", "arch", "compiler", "build_type"
    options = {"fPIC": [True, False]}
    default_options = {"fPIC": True}

    exports_sources = (
        "CMakeLists.txt",
        "include/*",
        "src/*",
        "test/*",
        "LICENSE*",
    )

    # The plugin only needs backend_api.h from lux-accel at build time. It is
    # a host requires (not tool_requires) so CMakeDeps emits lux-accel's
    # include dirs — a build-context tool dep has its usage requirements
    # stripped, leaving <lux/accel/backend_api.h> unresolvable. lux-hip never
    # links lux-accel (it is a dlopen'd plugin, libs=[]). The ROCm/HIP toolkit
    # is located via CMake's find_package(hip); if not present, the plugin
    # builds as a stub that reports is_available()==0.
    def requirements(self):
        self.requires("lux-accel/0.1.3", headers=True, libs=False, run=False)

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()

        # Emits lux-accel-config.cmake so find_package(lux-accel CONFIG)
        # resolves and exposes <lux/accel/backend_api.h>.
        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        # Plugin - no CMake target export needed; consumers dlopen it.
        self.cpp_info.libs = []
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = ["lib/lux/plugins"]
