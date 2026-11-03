# Validation and experiment method

The test suite and experiment tools address different claims. A matching output does not validate timing, and matching architecture cycles do not establish a faster host simulator or physical hardware.

## Required checks

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/sweep.py --binary build/npu-sim --output-dir results/sweep
python3 scripts/compare_modes.py --binary build/npu-sim --output-dir results/modes
python3 scripts/mlp_demo.py --binary build/npu-sim
```

The core arithmetic tests cover signed zero-point correction, bias, rounding ties, saturation, ReLU, tile edges, deterministic generation, invalid parameters, and overflow handling. Independent Python checks should calculate expected values directly rather than reproduce only the C++ call graph. Compare full accumulators and quantized outputs; a hash is convenient evidence of identity but is not a substitute for a direct correctness comparison inside the executable.

Timing checks should include a single tile, one-slot serialization, overlapped tiles, memory-dominated and compute-dominated cases, non-divisible dimensions, and DMA arbitration. An independent event scheduler can validate the stated scheduling contract. The cycle and event implementations must also agree on final cycles, results, traffic, and stage spans. Agreement between two modes sharing scheduling code is a regression check, not independent proof of that code.

## Sanitizers

```sh
cmake -S . -B build-ubsan -DCMAKE_BUILD_TYPE=Debug -DNPU_ENABLE_SANITIZERS=ON
cmake --build build-ubsan -j2
ctest --test-dir build-ubsan --output-on-failure
```

The sanitizer option instruments the project's arithmetic, SystemC model, CLI and test code. SystemC is a separately built dependency; passing these checks does not certify every dependency or eliminate all possible C++ lifetime errors. Continuous integration builds with GCC and Clang and runs the sanitizer configuration.

## Architecture sweep

The sweep uses deterministic workload fixtures and pairs each alternative with a baseline for the same shape, seed, and quantization. It reports individual cycle counts, output hashes, correctness, configuration, and cycle ratios. Positive integer cycles are required. A modeled speedup ratio is

$$
r_i=\frac{C_{\mathrm{baseline},i}}{C_{\mathrm{candidate},i}}.
$$

For a complete set of comparable pairs, the aggregate is

$$
G=\exp\left(\frac{1}{N}\sum_{i=1}^{N}\log r_i\right).
$$

Never average unrelated workload results into an unsupported headline. These are modeled cycle ratios under one timing contract, not measured hardware speedups. Inspect memory and compute busy cycles and traffic when explaining a result. The script does not require every resource increase to improve performance; unchanged results can be the expected outcome when a different resource is limiting.

## Timing-mode experiment

`compare_modes.py` runs the same deterministic configurations in event and cycle modes, verifies architecture-result agreement, and repeats host measurements. It uses median subprocess wall time for each mode and case, retaining every measurement. Subprocess wall time includes process startup and is affected by host load, compiler settings, CPU scheduling, and system libraries. The executable's own host simulation duration, when available, has a narrower boundary and is retained separately.

Use sufficiently large simulated workloads when investigating event-versus-cycle overhead. Small cases can be dominated by startup noise. A host-runtime ratio says how quickly this executable simulates a case on this machine; it does not alter the modeled accelerator latency.

## Evidence and limits

Save the JSON and CSV reports alongside the exact executable build, compiler version, host, and command when publishing measurements. Report failed cases and regressions. Generated results are ignored by Git by default so results are not mistaken for portable guarantees.

The arithmetic follows the repository's explicit contract. There is no ONNX or TFLite conformance claim. Timing has no supplied RTL or silicon ground truth. Memory requests model service latency and bandwidth without physical bank conflicts, DRAM refresh, bus topology, or cache coherence. Real workload performance needs model calibration and validation against the target hardware before architectural decisions can be treated as hardware predictions.
