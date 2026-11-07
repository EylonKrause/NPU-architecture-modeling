cmake_minimum_required(VERSION 3.24)
project(NPUArchitectureModeling VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(NPU_FETCH_SYSTEMC "Fetch the pinned SystemC dependency" ON)
option(NPU_ENABLE_SANITIZERS "Enable undefined-behavior checks in project code" OFF)
option(NPU_ENABLE_ONNX_TESTS "Test optional ONNX importer against ONNX Runtime" OFF)

if(NPU_FETCH_SYSTEMC)
  include(FetchContent)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Static dependency" FORCE)
  set(ENABLE_EXAMPLES OFF CACHE BOOL "Do not build dependency examples" FORCE)
  set(ENABLE_REGRESSION OFF CACHE BOOL "Do not build dependency regressions" FORCE)
  FetchContent_Declare(systemc
    URL https://codeload.github.com/accellera-official/systemc/tar.gz/refs/tags/3.0.2
    URL_HASH SHA256=9b3693ed286aab958b9e5d79bb0ad3bc523bbc46931100553275352038f4a0c4
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(systemc)
  if(MSVC)
    set(NPU_SYSTEMC_SOURCE_DIR "${systemc_SOURCE_DIR}")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/PatchSystemCMsvc.cmake")
  endif()
else()
  find_package(SystemCLanguage 3.0 CONFIG REQUIRED)
endif()

# Version header: project version plus the configure-time git describe.
# GIT_OPTIONAL_LOCKS=0 keeps --dirty from rewriting the index.
set(NPU_GIT_DESCRIBE "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env GIT_OPTIONAL_LOCKS=0
            ${GIT_EXECUTABLE} describe --always --dirty --tags
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NPU_GIT_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NPU_GIT_RESULT ERROR_QUIET)
  if(NPU_GIT_RESULT EQUAL 0 AND NPU_GIT_OUTPUT)
    set(NPU_GIT_DESCRIBE "${NPU_GIT_OUTPUT}")
  endif()
endif()
configure_file(cmake/version.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/generated/npu/version.hpp @ONLY)

add_library(npu_core src/core.cpp src/json.cpp src/options.cpp)
target_include_directories(npu_core PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}/generated)
add_executable(npu-sim src/main.cpp src/model.cpp)
target_link_libraries(npu-sim PRIVATE npu_core SystemC::systemc)
target_include_directories(npu-sim SYSTEM PRIVATE
  $<TARGET_PROPERTY:SystemC::systemc,INTERFACE_INCLUDE_DIRECTORIES>)
add_executable(npu-tlm-demo src/tlm_platform.cpp)
target_link_libraries(npu-tlm-demo PRIVATE npu_core SystemC::systemc)
target_include_directories(npu-tlm-demo SYSTEM PRIVATE
  $<TARGET_PROPERTY:SystemC::systemc,INTERFACE_INCLUDE_DIRECTORIES>)
add_executable(npu-transformer-demo src/transformer.cpp)
target_link_libraries(npu-transformer-demo PRIVATE npu_core SystemC::systemc)
target_include_directories(npu-transformer-demo SYSTEM PRIVATE
  $<TARGET_PROPERTY:SystemC::systemc,INTERFACE_INCLUDE_DIRECTORIES>)
add_executable(npu-mobile-demo src/mobile_platform.cpp)
target_link_libraries(npu-mobile-demo PRIVATE npu_core SystemC::systemc)
target_include_directories(npu-mobile-demo SYSTEM PRIVATE
  $<TARGET_PROPERTY:SystemC::systemc,INTERFACE_INCLUDE_DIRECTORIES>)

foreach(target npu_core npu-sim npu-tlm-demo npu-transformer-demo npu-mobile-demo)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
  if(NPU_ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(${target} PRIVATE -fsanitize=undefined -fno-sanitize-recover=all)
    target_link_options(${target} PUBLIC -fsanitize=undefined)
  endif()
endforeach()

include(CTest)
if(BUILD_TESTING)
  add_executable(npu-core-tests tests/core_tests.cpp)
  target_link_libraries(npu-core-tests PRIVATE npu_core)
  if(NPU_ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(npu-core-tests PRIVATE -fsanitize=undefined -fno-sanitize-recover=all)
    target_link_options(npu-core-tests PRIVATE -fsanitize=undefined)
  endif()
  add_test(NAME arithmetic-and-validation COMMAND npu-core-tests)
  add_executable(npu-compression-core-tests tests/mobile_compression_core_tests.cpp)
  target_link_libraries(npu-compression-core-tests PRIVATE npu_core)
  if(NPU_ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(npu-compression-core-tests PRIVATE -fsanitize=undefined -fno-sanitize-recover=all)
    target_link_options(npu-compression-core-tests PRIVATE -fsanitize=undefined)
  endif()
  add_test(NAME mobile-compression-storage COMMAND npu-compression-core-tests)
  add_executable(npu-config-tests tests/config_tests.cpp)
  target_link_libraries(npu-config-tests PRIVATE npu_core)
  if(NPU_ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(npu-config-tests PRIVATE -fsanitize=undefined -fno-sanitize-recover=all)
    target_link_options(npu-config-tests PRIVATE -fsanitize=undefined)
  endif()
  add_test(NAME config-options-and-json
    COMMAND npu-config-tests ${CMAKE_CURRENT_SOURCE_DIR}/configs/example.json)
  find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
  add_test(NAME independent-integration
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration.py
            --binary $<TARGET_FILE:npu-sim>)
  add_test(NAME per-channel-and-int4
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/quantization_tests.py
            --binary $<TARGET_FILE:npu-sim>)
  add_test(NAME graph-inference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/graph_runtime_tests.py
            --binary $<TARGET_FILE:npu-sim>)
  add_test(NAME graph-compiler-fusion
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/graph_optimizer_tests.py
            --binary $<TARGET_FILE:npu-sim>)
  add_test(NAME mobile-weight-compression
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/mobile_compression_tests.py
            --binary $<TARGET_FILE:npu-sim>)
  add_test(NAME mobile-transformer
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/transformer_tests.py
            --binary $<TARGET_FILE:npu-transformer-demo>)
  add_test(NAME mobile-soc-platform
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/mobile_platform_tests.py
            --binary $<TARGET_FILE:npu-mobile-demo>)
  set_tests_properties(mobile-weight-compression mobile-transformer mobile-soc-platform
    graph-compiler-fusion PROPERTIES TIMEOUT 180)
  add_test(NAME config-file-cli
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/config_file_tests.py
            --binary $<TARGET_FILE:npu-sim>)
  add_test(NAME architecture-contract
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/architecture_tests.py
            --binary $<TARGET_FILE:npu-sim>)
  add_test(NAME energy-analysis
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/energy_tests.py)
  add_test(NAME mapping-search
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/autotune_tests.py
            --binary $<TARGET_FILE:npu-sim>)
  add_test(NAME systemc-observability
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/systemc_observability_tests.py
            --binary $<TARGET_FILE:npu-sim>)
  add_test(NAME systemc-tlm-platform
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tlm_platform_tests.py
            --binary $<TARGET_FILE:npu-tlm-demo>)
  add_test(NAME live-control-protocol
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/live_protocol_tests.py
            --binary $<TARGET_FILE:npu-sim>)
  add_test(NAME live-dashboard
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/live_dashboard_tests.py
            --binary $<TARGET_FILE:npu-sim>)
  add_test(NAME one-command-launcher
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/launcher_tests.py
            --binary $<TARGET_FILE:npu-sim>)
  set_tests_properties(one-command-launcher PROPERTIES TIMEOUT 90)
  set_tests_properties(live-dashboard live-control-protocol PROPERTIES TIMEOUT 90)
  if(NPU_ENABLE_ONNX_TESTS)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import onnx, onnxruntime"
      RESULT_VARIABLE NPU_ONNX_IMPORT_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(NOT NPU_ONNX_IMPORT_RESULT EQUAL 0)
      message(FATAL_ERROR "ONNX tests require pip install -r requirements-onnx.txt in the selected Python interpreter")
    endif()
    add_test(NAME onnx-runtime-conformance
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/onnx_import_tests.py
              --binary $<TARGET_FILE:npu-sim>)
    set_tests_properties(onnx-runtime-conformance PROPERTIES TIMEOUT 180)
  endif()
  set_tests_properties(independent-integration PROPERTIES TIMEOUT 180)
endif()
