# Test registration.
#
# Names are <area>.<what> in lower-case dash style (sim.gemm-oracle, soc.tlm). The area is a
# ctest label, so `ctest -L soc` runs one area; C++ executables add the label `unit` and
# Python suites `python`. TIMEOUT is in seconds at NPU_TEST_TIMEOUT_SCALE=1 (default 120) and
# is multiplied by the scale, which is also exported to the test as NPU_TEST_TIMEOUT_SCALE so
# a suite can stretch its own internal waits by the same factor.
#
# Every test sees the tool paths as NPU_SIM, NPU_TLM_DEMO, NPU_TRANSFORMER_DEMO and
# NPU_MOBILE_DEMO, and the source root on PYTHONPATH.

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

function(_npu_register_test name timeout)
  if(NOT name MATCHES "^[a-z][a-z0-9]*(\\.[a-z0-9]+(-[a-z0-9]+)*)+$")
    message(FATAL_ERROR "test name '${name}' must be <area>.<what>, lower case, dash separated")
  endif()
  string(REGEX MATCH "^[a-z0-9]+" area "${name}")
  if(timeout STREQUAL "")
    set(timeout 120)
  endif()
  math(EXPR timeout "${timeout} * ${NPU_TEST_TIMEOUT_SCALE}")
  set(labels ${area} ${ARGN})
  set(environment
    "NPU_TEST_TIMEOUT_SCALE=${NPU_TEST_TIMEOUT_SCALE}"
    "NPU_SIM=$<TARGET_FILE:npu-sim>"
    "NPU_TLM_DEMO=$<TARGET_FILE:npu-tlm-demo>"
    "NPU_TRANSFORMER_DEMO=$<TARGET_FILE:npu-transformer-demo>"
    "NPU_MOBILE_DEMO=$<TARGET_FILE:npu-mobile-demo>")
  set_tests_properties(${name} PROPERTIES
    LABELS "${labels}"
    TIMEOUT ${timeout}
    ENVIRONMENT "${environment}"
    ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}")
endfunction()

# npu_add_unit_test(<name> TARGET <executable> SOURCES <file>... [TIMEOUT <s>] [LABELS <l>...])
function(npu_add_unit_test name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "TARGET;TIMEOUT" "SOURCES;LABELS")
  if(NOT arg_TARGET OR NOT arg_SOURCES)
    message(FATAL_ERROR "npu_add_unit_test(${name}): TARGET and SOURCES are required")
  endif()
  add_executable(${arg_TARGET} ${arg_SOURCES})
  target_link_libraries(${arg_TARGET} PRIVATE npu_core)
  npu_project_target(${arg_TARGET})
  add_test(NAME ${name} COMMAND ${arg_TARGET})
  _npu_register_test(${name} "${arg_TIMEOUT}" unit ${arg_LABELS})
endfunction()

# npu_add_python_test(<name> SCRIPT <path> [BINARY <tool>] [ARGS <arg>...] [TIMEOUT <s>]
#                     [LABELS <l>...] [FIXTURES_REQUIRED <f>...])
# Runs <path> (relative to the source root) from the build directory. BINARY passes
# --binary $<TARGET_FILE:<tool>> before ARGS.
function(npu_add_python_test name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "SCRIPT;BINARY;TIMEOUT" "ARGS;LABELS;FIXTURES_REQUIRED")
  if(NOT arg_SCRIPT)
    message(FATAL_ERROR "npu_add_python_test(${name}): SCRIPT is required")
  endif()
  set(command "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/${arg_SCRIPT}")
  if(arg_BINARY)
    list(APPEND command --binary "$<TARGET_FILE:${arg_BINARY}>")
  endif()
  add_test(NAME ${name} COMMAND ${command} ${arg_ARGS} WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")
  _npu_register_test(${name} "${arg_TIMEOUT}" python ${arg_LABELS})
  if(arg_FIXTURES_REQUIRED)
    set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED "${arg_FIXTURES_REQUIRED}")
  endif()
endfunction()
