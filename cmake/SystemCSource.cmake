# Pinned SystemC release, read by CMakeLists.txt (FetchContent) and cmake/BuildSystemC.cmake
# (CI cache prefix).
set(NPU_SYSTEMC_VERSION 3.0.2)
set(NPU_SYSTEMC_URL
  "https://codeload.github.com/accellera-official/systemc/tar.gz/refs/tags/${NPU_SYSTEMC_VERSION}")
set(NPU_SYSTEMC_SHA256 9b3693ed286aab958b9e5d79bb0ad3bc523bbc46931100553275352038f4a0c4)

# Dependency options applied in both paths: a static library (the tools are single binaries,
# and npu_core has no DLL exports), no examples or regressions, and no Accellera banner on
# stderr, so every tool's diagnostics start with the tool's own name.
set(NPU_SYSTEMC_OPTIONS
  BUILD_SHARED_LIBS=OFF
  ENABLE_EXAMPLES=OFF
  ENABLE_REGRESSION=OFF
  DISABLE_COPYRIGHT_MESSAGE=ON)
