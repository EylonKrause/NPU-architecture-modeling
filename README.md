# NPU architecture modeling

A working C++17 and SystemC model of a configurable INT8 dense-layer accelerator. It combines a bit-exact arithmetic reference with an explicit load/compute/store timing contract, a shared DMA engine, bounded tile buffers, and reproducible architecture experiments.

The supported workload is dense matrix multiplication, bias, integer requantization, and optional fused ReLU. The model is intentionally small enough to inspect and extend. It is not a full machine-learning runtime, a replica of a commercial NPU, or a claim of RTL or silicon cycle accuracy.

## Build and run

You need a C++17 compiler, CMake 3.24 or newer, a build tool, and Python 3.10 or newer for the experiment and independent-check scripts. By default CMake fetches the pinned, hash-verified [SystemC 3.0.2](https://github.com/accellera-official/systemc/releases/tag/3.0.2) source, so the initial configuration needs internet access. To use an installed SystemC package, configure with `-DNPU_FETCH_SYSTEMC=OFF -DCMAKE_PREFIX_PATH=/path/to/systemc`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
build/npu-sim --help
build/npu-sim --m 16 --n 16 --k 32 --timing event
```

The executable writes one JSON result to stdout. Diagnostic messages go to stderr. Without an input file, data is generated deterministically from the seed, so the same workload can be compared across architecture configurations. You can also provide real input, weights, and bias through the text format described in [Examples](examples/README.md).

```sh
python3 scripts/sweep.py --binary build/npu-sim --output-dir results/sweep
python3 scripts/compare_modes.py --binary build/npu-sim --output-dir results/modes
python3 scripts/mlp_demo.py --binary build/npu-sim
```

These scripts use only Python's standard library. They fail if arithmetic agreement or paired output checks fail. The sweep compares target architecture cycles; the timing-mode experiment separately measures simulator host runtime. A faster simulator is not evidence of a faster accelerator.

## What the model represents

- Signed INT8 activations and weights, configurable zero points, INT32 bias and final accumulators, and an INT64 intermediate for overflow checking.
- Output tiling over rows and columns, including partial edge tiles. Each tile spans the full reduction dimension.
- A compute engine with a configurable MAC budget and pipeline overhead.
- One non-preemptive DMA engine shared by reads and writes, with request-boundary round-robin arbitration, transaction latency, and bandwidth.
- Bounded end-to-end buffer slots and a conservative SRAM capacity check.
- Two SystemC execution modes implementing the same stage timing: event jumps and single-cycle waits.
- Architecture-cycle, traffic, utilization, trace, correctness, and host-runtime reporting.

Data is reloaded for each output tile. There is no inter-tile cache, sparse execution, compression, convolution frontend, or compiler. The two-layer MLP example feeds one invocation's INT8 outputs into the next layer, demonstrating network inference without claiming cross-layer overlap or retained on-chip state. Matrix multiply-accumulate counts are reported as MACs, not doubled into operations without explanation.

## How to read a result

First check arithmetic agreement and the output hash. Then compare cycles, memory traffic, compute and DMA busy cycles, utilization, and buffer pressure. Change one architecture variable at a time while preserving shape, seed, and quantization. A second buffer can enable overlap, but a shared DMA engine still serializes transfers. More MAC capacity can leave a bandwidth-limited workload almost unchanged.

The scalar reference is a numerical baseline, not a hardware timing baseline. The one-slot configuration supplies the serial timing baseline. Event and cycle modes provide a simulator-performance comparison under an identical modeled architecture.

Read [Architecture](docs/ARCHITECTURE.md) for equations and limitations, [Validation](docs/VALIDATION.md) for the evidence standard, and [Examples](examples/README.md) for reproducible experiments. The [implementation contract](docs/IMPLEMENTATION-CONTRACT.md) defines the model's intended semantics.

See [verified results](docs/VALIDATED-RESULTS.md) for checked workload outputs and reproducible architecture comparisons.

## Main controls

| Options | Meaning |
| --- | --- |
| `--m`, `--n`, `--k` | Generated workload dimensions; file input supplies its own dimensions. |
| `--input PATH` | Supply activations, weights, and bias in the documented `NPU1` format. |
| `--tile-m`, `--tile-n` | Output tile dimensions; reduction spans the full input dimension. |
| `--macs-per-cycle`, `--pipeline-cycles` | Compute capacity and per-tile pipeline overhead. |
| `--memory-bytes-per-cycle`, `--memory-latency` | Shared DMA bandwidth and per-request service latency. |
| `--buffer-slots`, `--sram-bytes` | In-flight tile limit and accelerator SRAM capacity. |
| `--input-zero`, `--weight-zero`, `--output-zero` | Signed INT8 zero points. |
| `--multiplier`, `--shift`, `--relu` | Integer requantization and optional fused activation. |
| `--timing event`, `--timing cycle` | Two ways to execute the same stage timing contract. |
| `--frequency-mhz` | Convert architecture cycles to reported target time. |
| `--seed` | Deterministic fixture seed; unused for file-supplied data. |
| `--trace PATH` | Write load, compute, and store stage spans as CSV. |
| `--dump-output` | Include full accumulator and output arrays in the JSON result. |
