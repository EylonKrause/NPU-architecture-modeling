cmake_minimum_required(VERSION 3.24)
project(NPUArchitectureModeling
  VERSION 0.4.0
  DESCRIPTION "SystemC/TLM-2.0 architecture model of a tiled INT8/INT4 NPU"
  HOMEPAGE_URL "https://github.com/EylonKrause/NPU-architecture-modeling"
  LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Release, Debug or RelWithDebInfo" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

option(NPU_FETCH_SYSTEMC "Fetch the pinned SystemC dependency" ON)
set(NPU_SANITIZERS "" CACHE STRING
  "Sanitizers for the project and a fetched SystemC: any of address;undefined;leak")
option(NPU_ENABLE_SANITIZERS "Deprecated: same as NPU_SANITIZERS=undefined" OFF)
option(NPU_WARNINGS_AS_ERRORS "Treat warnings in project code as errors" OFF)
option(NPU_COVERAGE "Instrument project code for gcov line coverage" OFF)
option(NPU_ENABLE_ONNX_TESTS "Test the optional ONNX importer against ONNX Runtime" OFF)
if(DEFINED ENV{NPU_TEST_TIMEOUT_SCALE})
  set(_npu_timeout_scale "$ENV{NPU_TEST_TIMEOUT_SCALE}")
else()
  set(_npu_timeout_scale 1)
endif()
set(NPU_TEST_TIMEOUT_SCALE "${_npu_timeout_scale}" CACHE STRING
  "Integer multiplier for test timeouts (sanitizer builds, slow hosts)")
if(NOT NPU_TEST_TIMEOUT_SCALE MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR
    "NPU_TEST_TIMEOUT_SCALE must be a positive integer, got '${NPU_TEST_TIMEOUT_SCALE}'")
endif()

include(GNUInstallDirs)
include(NpuCompileOptions)  # before SystemC: sanitizer flags must reach the dependency
include(NpuVersion)
include(SystemCSource)

if(NPU_FETCH_SYSTEMC)
  include(FetchContent)
  # Normal variables (CMP0077): they configure SystemC without touching the user's cache.
  foreach(_npu_option IN LISTS NPU_SYSTEMC_OPTIONS)
    string(REGEX MATCH "^([^=]+)=(.*)$" _npu_match "${_npu_option}")
    set(${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
  endforeach()
  # SystemC replaces a defaulted install prefix with /opt/systemc; keep the platform default.
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
  FetchContent_Declare(systemc
    URL "${NPU_SYSTEMC_URL}"
    URL_HASH SHA256=${NPU_SYSTEMC_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  # GNUInstallDirs has set CMAKE_INSTALL_DOCDIR to this project's directory, and SystemC's
  # cache default cannot override a normal variable: without this, its README, LICENSE and
  # NOTICE install as this project's documentation.
  set(_npu_docdir "${CMAKE_INSTALL_DOCDIR}")
  set(NPU_SYSTEMC_DOCDIR "${CMAKE_INSTALL_DATAROOTDIR}/doc/systemc")
  set(CMAKE_INSTALL_DOCDIR "${NPU_SYSTEMC_DOCDIR}")
  FetchContent_MakeAvailable(systemc)
  set(CMAKE_INSTALL_DOCDIR "${_npu_docdir}")
  npu_sanitize_systemc(systemc)
  if(MSVC)
    set(NPU_SYSTEMC_SOURCE_DIR "${systemc_SOURCE_DIR}")
    include(PatchSystemCMsvc)
  endif()
else()
  find_package(SystemCLanguage ${NPU_SYSTEMC_VERSION} CONFIG REQUIRED)
endif()

set(NPU_GENERATED_INCLUDE_DIR "${PROJECT_BINARY_DIR}/generated/include")
npu_generate_version_header("${NPU_GENERATED_INCLUDE_DIR}/npu/version.hpp")

add_library(npu_core STATIC src/core.cpp src/conv.cpp src/json.cpp src/options.cpp)
target_include_directories(npu_core PUBLIC include "${NPU_GENERATED_INCLUDE_DIR}")
npu_project_target(npu_core)

npu_add_tool(npu-sim src/main.cpp src/model.cpp src/memory.cpp)
npu_add_tool(npu-tlm-demo src/tlm_platform.cpp)
npu_add_tool(npu-transformer-demo src/transformer.cpp)
npu_add_tool(npu-mobile-demo src/mobile_platform.cpp)

include(CTest)
if(BUILD_TESTING)
  include(NpuTesting)

  npu_add_unit_test(core.arithmetic
    TARGET npu-core-tests SOURCES tests/core_tests.cpp TIMEOUT 60)
  npu_add_unit_test(core.weight-codec
    TARGET npu-compression-core-tests SOURCES tests/mobile_compression_core_tests.cpp TIMEOUT 60)
  npu_add_unit_test(core.config
    TARGET npu-config-tests SOURCES tests/config_tests.cpp TIMEOUT 60
    ARGS "${PROJECT_SOURCE_DIR}/configs/example.json")

  npu_add_python_test(sim.gemm-oracle
    SCRIPT tests/integration.py BINARY npu-sim TIMEOUT 180)
  npu_add_python_test(sim.quantization
    SCRIPT tests/quantization_tests.py BINARY npu-sim)
  npu_add_python_test(sim.memory-hierarchy
    SCRIPT tests/architecture_tests.py BINARY npu-sim)
  npu_add_python_test(sim.weight-compression
    SCRIPT tests/mobile_compression_tests.py BINARY npu-sim TIMEOUT 180)
  npu_add_python_test(sim.observability
    SCRIPT tests/systemc_observability_tests.py BINARY npu-sim)
  npu_add_python_test(sim.config-file
    SCRIPT tests/config_file_tests.py BINARY npu-sim)
  npu_add_python_test(sim.memory-model
    SCRIPT tests/memory_model_tests.py BINARY npu-sim TIMEOUT 180)
  npu_add_python_test(sim.cluster
    SCRIPT tests/cluster_tests.py BINARY npu-sim TIMEOUT 180)
  npu_add_python_test(sim.design-points
    SCRIPT tests/design_point_tests.py BINARY npu-sim TIMEOUT 180)
  npu_add_python_test(sim.conv-traffic
    SCRIPT tests/conv_traffic_tests.py BINARY npu-sim TIMEOUT 180)
  npu_add_python_test(graph.runtime
    SCRIPT tests/graph_runtime_tests.py BINARY npu-sim)
  npu_add_python_test(graph.fusion
    SCRIPT tests/graph_optimizer_tests.py BINARY npu-sim TIMEOUT 180)
  npu_add_python_test(soc.transformer
    SCRIPT tests/transformer_tests.py BINARY npu-transformer-demo TIMEOUT 180)
  npu_add_python_test(soc.multicore
    SCRIPT tests/mobile_platform_tests.py BINARY npu-mobile-demo TIMEOUT 180)
  npu_add_python_test(soc.tlm
    SCRIPT tests/tlm_platform_tests.py BINARY npu-tlm-demo)
  npu_add_python_test(live.protocol
    SCRIPT tests/live_protocol_tests.py BINARY npu-sim TIMEOUT 90)
  npu_add_python_test(live.dashboard
    SCRIPT tests/live_dashboard_tests.py BINARY npu-sim TIMEOUT 90)
  npu_add_python_test(tools.launcher
    SCRIPT tests/launcher_tests.py BINARY npu-sim TIMEOUT 90)
  npu_add_python_test(tools.energy
    SCRIPT tests/energy_tests.py BINARY npu-sim)
  npu_add_python_test(tools.autotune
    SCRIPT tests/autotune_tests.py BINARY npu-sim)

  set(_npu_install_args
    --cmake "${CMAKE_COMMAND}" --build-dir "${PROJECT_BINARY_DIR}" --config "$<CONFIG>"
    --bindir "${CMAKE_INSTALL_BINDIR}" --docdir "${CMAKE_INSTALL_DOCDIR}"
    --tools $<TARGET_FILE_NAME:npu-sim> $<TARGET_FILE_NAME:npu-tlm-demo>
            $<TARGET_FILE_NAME:npu-transformer-demo> $<TARGET_FILE_NAME:npu-mobile-demo>)
  if(NPU_FETCH_SYSTEMC)
    list(APPEND _npu_install_args --systemc-docdir "${NPU_SYSTEMC_DOCDIR}")
  endif()
  npu_add_python_test(build.install
    SCRIPT tests/install_tests.py ARGS ${_npu_install_args})
  npu_add_python_test(ci.systemc-cache
    SCRIPT tests/ci_cache_tests.py TIMEOUT 30)
  if("undefined" IN_LIST NPU_SANITIZERS)
    add_executable(npu-ubsan-vptr-probe tests/ubsan_vptr_probe.cpp)
    npu_project_target(npu-ubsan-vptr-probe)
    npu_add_python_test(build.ubsan-vptr
      SCRIPT tests/ubsan_vptr_tests.py BINARY npu-ubsan-vptr-probe TIMEOUT 30)
  endif()

  # The scripts under scripts/ run end to end on the bundled examples. The setup fixture
  # deletes the outputs these scripts refuse to overwrite, so ctest can be re-run.
  set(_npu_mobile_out "${PROJECT_BINARY_DIR}/examples-mobile")
  set(_npu_fused_out "${PROJECT_BINARY_DIR}/examples-fused-graph.json")
  add_test(NAME examples.clean-output
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_npu_mobile_out}" "${_npu_fused_out}")
  _npu_register_test(examples.clean-output 30)
  set_tests_properties(examples.clean-output PROPERTIES FIXTURES_SETUP examples-output)
  npu_add_python_test(examples.sweep
    SCRIPT scripts/sweep.py BINARY npu-sim)
  npu_add_python_test(examples.compare-modes
    SCRIPT scripts/compare_modes.py BINARY npu-sim ARGS --repeats 2)
  npu_add_python_test(examples.mlp
    SCRIPT scripts/mlp_demo.py BINARY npu-sim)
  npu_add_python_test(examples.graph-cnn
    SCRIPT scripts/graph_runtime.py BINARY npu-sim
    ARGS "${PROJECT_SOURCE_DIR}/examples/cnn.json")
  npu_add_python_test(examples.graph-residual
    SCRIPT scripts/graph_runtime.py BINARY npu-sim
    ARGS "${PROJECT_SOURCE_DIR}/examples/residual_mlp.json")
  npu_add_python_test(examples.graph-fusion
    SCRIPT scripts/graph_optimize.py BINARY npu-sim
    ARGS "${PROJECT_SOURCE_DIR}/examples/mobile_fusion.json" --verify --output "${_npu_fused_out}"
    FIXTURES_REQUIRED examples-output)
  npu_add_python_test(examples.mobile-suite
    SCRIPT scripts/mobile_suite.py
    ARGS --build-dir "$<TARGET_FILE_DIR:npu-sim>" --output-dir "${_npu_mobile_out}"
    TIMEOUT 180 FIXTURES_REQUIRED examples-output)
  npu_add_python_test(examples.autotune
    SCRIPT scripts/autotune.py BINARY npu-sim)

  if(NPU_ENABLE_ONNX_TESTS)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import onnx, onnxruntime"
      RESULT_VARIABLE _npu_onnx_import OUTPUT_QUIET ERROR_QUIET)
    if(NOT _npu_onnx_import EQUAL 0)
      message(FATAL_ERROR
        "ONNX tests require `pip install -r requirements-onnx.txt` for ${Python3_EXECUTABLE}")
    endif()
    npu_add_python_test(onnx.ort-differential
      SCRIPT tests/onnx_import_tests.py BINARY npu-sim TIMEOUT 180)
  endif()
  npu_add_python_test(tools.npuperf
    SCRIPT tests/npuperf_tests.py BINARY npu-sim TIMEOUT 300)
  npu_add_python_test(tools.analytic-model
    SCRIPT tests/analytic_model_tests.py BINARY npu-sim TIMEOUT 180)
  npu_add_python_test(tools.mapper
    SCRIPT tests/mapper_tests.py BINARY npu-sim TIMEOUT 180)
  npu_add_python_test(tools.studies
    SCRIPT tests/studies_tests.py BINARY npu-sim TIMEOUT 180)
endif()
