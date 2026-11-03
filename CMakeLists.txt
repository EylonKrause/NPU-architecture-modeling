cmake_minimum_required(VERSION 3.24)
project(NPUArchitectureModeling VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(NPU_FETCH_SYSTEMC "Fetch the pinned SystemC dependency" ON)
option(NPU_ENABLE_SANITIZERS "Enable undefined-behavior checks in project code" OFF)

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
else()
  find_package(SystemCLanguage 3.0 CONFIG REQUIRED)
endif()

add_library(npu_core src/core.cpp)
target_include_directories(npu_core PUBLIC include)
add_executable(npu-sim src/main.cpp src/model.cpp)
target_link_libraries(npu-sim PRIVATE npu_core SystemC::systemc)
target_include_directories(npu-sim SYSTEM PRIVATE
  $<TARGET_PROPERTY:SystemC::systemc,INTERFACE_INCLUDE_DIRECTORIES>)

foreach(target npu_core npu-sim)
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
  find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
  add_test(NAME independent-integration
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration.py
            --binary $<TARGET_FILE:npu-sim>)
  set_tests_properties(independent-integration PROPERTIES TIMEOUT 180)
endif()
