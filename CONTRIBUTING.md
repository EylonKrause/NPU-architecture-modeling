# Contributing

- Build and test with a preset (CMake 3.25+, Ninja): `cmake --workflow --preset release`. One area: `ctest --preset release -L sim` (areas: core, sim, graph, soc, live, tools, examples).
- Before a pull request run `cmake --workflow --preset ci-gcc` (warnings are errors) and, on Linux, `cmake --workflow --preset asan-ubsan`. On Windows run `ci-msvc` from a Visual Studio developer prompt.
- Format C++ with clang-format 21.1.8 and Python with ruff 0.16.10 (`ruff format`, `ruff check`), the versions CI uses.
- Register tests with `npu_add_unit_test` or `npu_add_python_test` (cmake/NpuTesting.cmake) under an `<area>.<what>` name. A change to timing or numerics updates the C++ and Python oracles in the same commit and says so in the message.
- Keep `build*/`, `results/` and `CMakeUserPresets.json` untracked. Contributions are licensed under Apache-2.0 (LICENSE).
