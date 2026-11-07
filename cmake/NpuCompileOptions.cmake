# Compile settings for project targets. SystemC keeps its own warning flags; only the
# sanitizer instrumentation below reaches it, minus the checks npu_sanitize_systemc() drops.

# --- Sanitizers ----------------------------------------------------------------------------
# Applied with add_compile_options() before SystemC is added, so a fetched SystemC is
# instrumented together with the project: a project object freed while the SystemC scheduler
# still references it faults inside kernel code, which ASan only checks when it is
# instrumented.
if(NPU_ENABLE_SANITIZERS AND NPU_SANITIZERS STREQUAL "")
  message(DEPRECATION "NPU_ENABLE_SANITIZERS is deprecated; use -DNPU_SANITIZERS=undefined")
  set(NPU_SANITIZERS undefined)
endif()
if(NOT NPU_SANITIZERS STREQUAL "")
  if(MSVC)
    message(FATAL_ERROR "NPU_SANITIZERS is supported with GCC and Clang only")
  endif()
  foreach(_npu_sanitizer IN LISTS NPU_SANITIZERS)
    if(NOT _npu_sanitizer MATCHES "^(address|undefined|leak)$")
      message(FATAL_ERROR
        "NPU_SANITIZERS: unsupported sanitizer '${_npu_sanitizer}' (use address, undefined, leak)")
    endif()
  endforeach()
  list(JOIN NPU_SANITIZERS "," _npu_sanitize)
  set(_npu_sanitize_flags
    -fsanitize=${_npu_sanitize} -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:${_npu_sanitize_flags}>")
  add_link_options(-fsanitize=${_npu_sanitize})
  if(NOT NPU_FETCH_SYSTEMC)
    message(STATUS "NPU_SANITIZERS: the prebuilt SystemC package is not instrumented")
  endif()
endif()

# npu_sanitize_systemc(<target>): the exemptions a fetched SystemC needs, on its own objects
# only. SystemC invokes process bodies through sc_process_host member-function pointers, which
# UBSan's vptr check rejects (sc_process.h, reproduced with a one-method module), so every
# SystemC-driven test would abort in kernel code. Project code keeps the vptr check for its
# sc_module subclasses and payload casts; the build.ubsan-vptr test holds it to that.
function(npu_sanitize_systemc target)
  if("undefined" IN_LIST NPU_SANITIZERS)
    target_compile_options(${target} PRIVATE -fno-sanitize=vptr)
  endif()
endfunction()

# --- Warnings ------------------------------------------------------------------------------
add_library(npu_warnings INTERFACE)
if(MSVC)
  # /utf-8: sources are UTF-8; without it MSVC decodes them with the host code page.
  target_compile_options(npu_warnings INTERFACE /W4 /permissive- /utf-8)
else()
  target_compile_options(npu_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wold-style-cast
    -Wnon-virtual-dtor -Woverloaded-virtual -Wimplicit-fallthrough)
endif()

# Settings every project target gets: the warning set, NPU_WARNINGS_AS_ERRORS (through the
# COMPILE_WARNING_AS_ERROR property, so `cmake --compile-no-warning-as-error` can still
# override it) and gcov instrumentation.
function(npu_project_target target)
  target_link_libraries(${target} PRIVATE npu_warnings)
  set_target_properties(${target} PROPERTIES COMPILE_WARNING_AS_ERROR ${NPU_WARNINGS_AS_ERRORS})
  if(NPU_COVERAGE)
    if(MSVC)
      message(FATAL_ERROR "NPU_COVERAGE is supported with GCC and Clang only")
    endif()
    target_compile_options(${target} PRIVATE --coverage)
    target_link_options(${target} PRIVATE --coverage)
  endif()
endfunction()

# SystemC headers are SYSTEM includes so their warnings stay out of project diagnostics.
# (The FetchContent SYSTEM keyword needs CMake 3.25; this works from 3.24.)
function(npu_link_systemc target)
  target_link_libraries(${target} PRIVATE SystemC::systemc)
  target_include_directories(${target} SYSTEM PRIVATE
    $<TARGET_PROPERTY:SystemC::systemc,INTERFACE_INCLUDE_DIRECTORIES>)
endfunction()

# npu_add_tool(<name> <source>...): a SystemC executable linked with npu_core and installed
# to ${CMAKE_INSTALL_BINDIR}.
function(npu_add_tool name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE npu_core)
  npu_project_target(${name})
  npu_link_systemc(${name})
  install(TARGETS ${name} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endfunction()
