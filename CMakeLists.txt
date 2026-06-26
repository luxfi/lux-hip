cmake_minimum_required(VERSION 3.20)
project(lux-hip VERSION 0.1.0 LANGUAGES CXX)

# lux-hip is a PLUGIN (shared library) loaded by lux-accel at runtime.
# It links nothing from lux-accel — it only consumes backend_api.h headers.
# It is the ROCm sibling of lux-cuda.

# Locate lux-accel for the backend_api.h header. Allow override via
# LUXACCEL_INCLUDE_DIR or fall back to a sibling checkout / install tree.
find_package(lux-accel CONFIG QUIET)
if(NOT lux-accel_FOUND)
  if(NOT LUXACCEL_INCLUDE_DIR)
    set(LUXACCEL_INCLUDE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/../lux-accel/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/../install/include")
  endif()
endif()

# ROCm/HIP toolkit (optional — plugin builds as no-op stub on systems without HIP).
# CMAKE_PREFIX_PATH should include the ROCm install (e.g. /opt/rocm) on AMD hosts.
find_package(hip CONFIG QUIET)

add_library(lux_hip MODULE
  src/hip_plugin.cpp
)

target_compile_features(lux_hip PRIVATE cxx_std_17)

target_include_directories(lux_hip
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${LUXACCEL_INCLUDE_DIR}
)

if(lux-accel_FOUND AND TARGET lux::accel)
  target_include_directories(lux_hip PRIVATE
    $<TARGET_PROPERTY:lux::accel,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

# Link HIP runtime + hiprtc if available; otherwise build the stub path.
if(hip_FOUND)
  target_link_libraries(lux_hip PRIVATE hip::host)
  # hiprtc ships as a separate library next to the HIP runtime.
  find_library(HIPRTC_LIB NAMES hiprtc
    HINTS ${HIP_ROOT_DIR}/lib ${hip_DIR}/../../../lib /opt/rocm/lib)
  if(HIPRTC_LIB)
    target_link_libraries(lux_hip PRIVATE ${HIPRTC_LIB})
  endif()
  target_compile_definitions(lux_hip PRIVATE LUX_HAS_HIP=1)
  message(STATUS "lux-hip: HIP toolkit found, building real ROCm plugin")
else()
  message(STATUS "lux-hip: HIP toolkit NOT found, building stub plugin "
                 "(is_available() returns 0). Real ROCm path needs AMD hardware.")
endif()

# Plugin settings — hidden visibility, .plugin extension matching loader pattern.
set_target_properties(lux_hip PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  PREFIX ""
  SUFFIX ".plugin"
)

include(GNUInstallDirs)

install(TARGETS lux_hip
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/lux/plugins
)

# Smoke test — dlopens the plugin and verifies the entry point + identity.
option(LUX_HIP_BUILD_TESTS "Build the lux-hip plugin smoke test" ON)
if(LUX_HIP_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_load.cpp")
  add_executable(lux_hip_test_load test/test_load.cpp)
  target_compile_features(lux_hip_test_load PRIVATE cxx_std_17)
  target_include_directories(lux_hip_test_load PRIVATE ${LUXACCEL_INCLUDE_DIR})
  if(lux-accel_FOUND AND TARGET lux::accel)
    target_include_directories(lux_hip_test_load PRIVATE
      $<TARGET_PROPERTY:lux::accel,INTERFACE_INCLUDE_DIRECTORIES>)
  endif()
  if(NOT WIN32)
    target_link_libraries(lux_hip_test_load PRIVATE ${CMAKE_DL_LIBS})
  endif()
  add_dependencies(lux_hip_test_load lux_hip)

  enable_testing()
  add_test(NAME lux_hip_load
           COMMAND lux_hip_test_load $<TARGET_FILE:lux_hip>)
endif()
