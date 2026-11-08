# Contributing

- Build and test with a preset (CMake 3.25+, Ninja, Python 3.10+): `cmake --workflow --preset release`. One area: `ctest --preset release -L sim`. Areas: `core`, `sim`, `graph`, `soc`, `live`, `tools`, `examples`, `build`, `ci`, `docs`, `onnx`; the labels `unit` and `python` select the C++ and Python suites, and `sweep` marks the design-point sweep.
- Before a pull request run `cmake --workflow --preset ci-gcc` (warnings are errors) and, on Linux, `cmake --workflow --preset asan-ubsan`. On Windows run `ci-msvc` from a Visual Studio developer prompt.
- Format C++ with clang-format 21.1.8 and Python with ruff 0.16.10 (`ruff format`, `ruff check`), the versions CI uses.
- Register tests with `npu_add_unit_test` or `npu_add_python_test` (cmake/NpuTesting.cmake) under an `<area>.<what>` name. A change to timing or numerics updates the C++ and Python oracles in the same commit and says so in the message.
- Generated files change only through their generators: `python3 -m npuperf.reproduce` (tables in docs/results.md, docs/figures/), `python3 -m npuperf.designs` (prints the design-point table of docs/design-points.md), `python3 tools/gen_cli_reference.py --write` (option tables in docs/usage.md) and `python3 tools/gen_workloads.py` (workloads/). CI and ctest compare each with a fresh run.
- Keep `build*/`, `results/` and `CMakeUserPresets.json` untracked. Contributions are licensed under Apache-2.0 (LICENSE).
