# Backport the official SystemC MSVC ostream compatibility fix:
# https://github.com/accellera-official/systemc/commit/3570c374f3c98e060d6267183ccb8de51aab5129
# The download remains pinned to SystemC 3.0.2. Never rewrite an unknown or
# locally modified dependency file; accept only the original or patched bytes.
if(NOT DEFINED NPU_SYSTEMC_SOURCE_DIR)
  message(FATAL_ERROR "NPU_SYSTEMC_SOURCE_DIR is required for the SystemC MSVC compatibility patch")
endif()
set(_npu_io_file "${NPU_SYSTEMC_SOURCE_DIR}/src/sysc/datatypes/int/sc_int64_io.cpp")
set(_npu_io_original "97432097e1b2866079342a22e21fa19f994044dc5937675bd9c256287338f235")
set(_npu_io_patched "850fc2370df345e6216f7e91fe07ad2040b46969a0b70b43aff7e900c116e4bd")
file(SHA256 "${_npu_io_file}" _npu_io_hash)
if(_npu_io_hash STREQUAL _npu_io_patched)
  return()
endif()
if(NOT _npu_io_hash STREQUAL _npu_io_original)
  message(FATAL_ERROR "Unexpected SystemC integer I/O source: refusing to overwrite a modified dependency")
endif()
file(READ "${_npu_io_file}" _npu_io_content)
string(REPLACE "    os.osfx();\n    return;" "    return;" _npu_io_content "${_npu_io_content}")
string(REPLACE "    os.osfx();" "    return;" _npu_io_content "${_npu_io_content}")
string(REPLACE "    if (os.opfx()) {" "    ::std::ostream::sentry sentry(os);\n    if (sentry) {"
       _npu_io_content "${_npu_io_content}")
string(SHA256 _npu_io_result "${_npu_io_content}")
if(NOT _npu_io_result STREQUAL _npu_io_patched)
  message(FATAL_ERROR "SystemC MSVC compatibility patch did not produce the expected source")
endif()
file(WRITE "${_npu_io_file}" "${_npu_io_content}")
message(STATUS "Applied official SystemC MSVC ostream compatibility fix")
