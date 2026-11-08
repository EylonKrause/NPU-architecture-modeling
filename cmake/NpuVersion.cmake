# Generates npu/version.hpp from cmake/version.hpp.in.
#
# The git description is taken at configure time. The HEAD reflog is registered as a
# configure dependency, so a commit, checkout or reset re-runs CMake on the next build and
# refreshes the description. -DNPU_GIT_DESCRIBE=<text> overrides it (source archives).

set(NPU_GIT_DESCRIBE "" CACHE STRING
  "Override for the git description compiled into the tools (empty: run git describe)")

function(_npu_git_describe out_var)
  set(${out_var} "unknown" PARENT_SCOPE)
  find_package(Git QUIET)
  if(NOT GIT_FOUND)
    return()
  endif()
  # Describe the tree only at the root of its own checkout: inside a repository that vendors
  # it, git describe would report the enclosing repository.
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --show-prefix
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE result OUTPUT_VARIABLE prefix ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT result EQUAL 0 OR NOT prefix STREQUAL "")
    return()
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty --abbrev=12
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE result OUTPUT_VARIABLE describe ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT result EQUAL 0 OR describe STREQUAL "")
    return()
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --git-path logs/HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE result OUTPUT_VARIABLE reflog ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(result EQUAL 0 AND NOT reflog STREQUAL "")
    if(NOT IS_ABSOLUTE "${reflog}")
      set(reflog "${PROJECT_SOURCE_DIR}/${reflog}")
    endif()
    if(EXISTS "${reflog}")
      set_property(DIRECTORY "${PROJECT_SOURCE_DIR}" APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${reflog}")
    endif()
  endif()
  set(${out_var} "${describe}" PARENT_SCOPE)
endfunction()

function(npu_generate_version_header output)
  if(NOT NPU_GIT_DESCRIBE STREQUAL "")
    set(describe "${NPU_GIT_DESCRIBE}")
  else()
    _npu_git_describe(describe)
  endif()
  # The value lands in a C++ string literal: keep only characters git allows in tags and
  # hashes that need no escaping.
  string(REGEX REPLACE "[^A-Za-z0-9._+/-]" "_" NPU_GIT_DESCRIBE_VALUE "${describe}")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/version.hpp.in" "${output}" @ONLY)
  message(STATUS "NPU version ${PROJECT_VERSION} (${NPU_GIT_DESCRIBE_VALUE})")
endfunction()
