# Builds the pinned SystemC once and installs it to a prefix, for CI caches and for
# configuring the project with -DNPU_FETCH_SYSTEMC=OFF -DCMAKE_PREFIX_PATH=<prefix>.
#
#   cmake -DNPU_SYSTEMC_PREFIX=<install dir> [-DNPU_SYSTEMC_WORK=<scratch dir>]
#         [-DCMAKE_BUILD_TYPE=Release] [-DCMAKE_GENERATOR=Ninja] -P cmake/BuildSystemC.cmake
#
# Download URL, SHA256 and dependency options come from SystemCSource.cmake, the MSVC fix from
# PatchSystemCMsvc.cmake, so the result matches what FetchContent builds. The compilers are
# taken from CC/CXX as in any first CMake configure; the C++ standard is the project's (17),
# which the SystemC library ABI encodes.
cmake_minimum_required(VERSION 3.24)

if(NOT NPU_SYSTEMC_PREFIX)
  message(FATAL_ERROR "BuildSystemC.cmake: -DNPU_SYSTEMC_PREFIX=<install dir> is required")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/SystemCSource.cmake")
file(TO_CMAKE_PATH "${NPU_SYSTEMC_PREFIX}" _prefix)
get_filename_component(_prefix "${_prefix}" ABSOLUTE)
if(NOT NPU_SYSTEMC_WORK)
  set(NPU_SYSTEMC_WORK "${_prefix}-work")
endif()
file(TO_CMAKE_PATH "${NPU_SYSTEMC_WORK}" _work)
get_filename_component(_work "${_work}" ABSOLUTE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT CMAKE_GENERATOR)
  set(CMAKE_GENERATOR Ninja)
endif()

set(_archive "${_work}/systemc-${NPU_SYSTEMC_VERSION}.tar.gz")
file(DOWNLOAD "${NPU_SYSTEMC_URL}" "${_archive}"
  EXPECTED_HASH SHA256=${NPU_SYSTEMC_SHA256} TLS_VERIFY ON STATUS _status)
list(GET _status 0 _code)
if(NOT _code EQUAL 0)
  list(GET _status 1 _reason)
  message(FATAL_ERROR "SystemC download failed: ${_reason}")
endif()

file(REMOVE_RECURSE "${_work}/src" "${_work}/build")
file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_work}/src")
set(NPU_SYSTEMC_SOURCE_DIR "${_work}/src/systemc-${NPU_SYSTEMC_VERSION}")
if(NOT EXISTS "${NPU_SYSTEMC_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "SystemC archive did not unpack to ${NPU_SYSTEMC_SOURCE_DIR}")
endif()
# The patched file is compiled only under _MSC_VER, so patching on every host is a no-op
# elsewhere and keeps one source tree per version.
include("${CMAKE_CURRENT_LIST_DIR}/PatchSystemCMsvc.cmake")

set(_definitions
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_CXX_STANDARD=17
  -DCMAKE_CXX_STANDARD_REQUIRED=ON
  -DCMAKE_CXX_EXTENSIONS=OFF
  "-DCMAKE_INSTALL_PREFIX=${_prefix}")
foreach(_option IN LISTS NPU_SYSTEMC_OPTIONS)
  list(APPEND _definitions "-D${_option}")
endforeach()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${NPU_SYSTEMC_SOURCE_DIR}" -B "${_work}/build"
          -G "${CMAKE_GENERATOR}" ${_definitions}
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${_work}/build" --config ${CMAKE_BUILD_TYPE} --parallel
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${_work}/build" --config ${CMAKE_BUILD_TYPE}
  COMMAND_ERROR_IS_FATAL ANY)
message(STATUS "SystemC ${NPU_SYSTEMC_VERSION} installed to ${_prefix}")
