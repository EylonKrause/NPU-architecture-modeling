# Experiments

Run from the repository root after building `build/npu-sim`.

## User-supplied dense layer

```sh
build/npu-sim --input examples/tiny-dense.npu --input-zero 0 --weight-zero 0 --output-zero 0 --multiplier 1 --shift 0 --dump-output
```

The expected output array is `[125, -110]`. Adding `--relu` produces `[125, 0]`.

The text format starts with `NPU1 M N K`, followed by whitespace-separated activation values, weights, and bias. Activations have row-major shape specified by the row and reduction dimensions; weights have row-major shape specified by the reduction and output dimensions. Bias contains one INT32 value per output column. Activation and weight values must fit signed INT8. The header selects the dimensions, so `--m`, `--n`, and `--k` cannot accompany `--input`.

```text
NPU1 1 2 3
10 -5 20
1 2
-3 4
5 -6
0 10
```

The input is a plain data file, not executable code. The quantization parameters remain explicit CLI options. This is a dense-layer interchange format, not a parser for ONNX or TFLite models.

## Two-layer neural inference

```sh
python3 scripts/mlp_demo.py --binary build/npu-sim
```

The script runs a small quantized dense layer with ReLU, feeds its outputs into a second dense layer, and checks the independently calculated expected results. It prints per-layer cycles, traffic, numerical agreement, and outputs. Each layer is a separate simulator process; the sum of layer cycles excludes host launch time and does not model cross-layer overlap or SRAM retention.

## Arithmetic and tiling

```sh
build/npu-sim --m 17 --n 13 --k 31 --tile-m 8 --tile-n 8 --timing event
```

This exercises partial edge tiles. The numerical result must agree with the scalar reference. The modeled traffic includes reloads for each output tile.

## Serial baseline and overlap

```sh
build/npu-sim --m 32 --n 32 --k 64 --buffer-slots 1 --timing event
build/npu-sim --m 32 --n 32 --k 64 --buffer-slots 2 --timing event
```

Preserve the arithmetic fixture and compare elapsed cycles and resource utilization. Extra slots permit overlap but reserve more SRAM. They do not create a second DMA engine.

## Reproducible architecture sweep

```sh
python3 scripts/sweep.py --binary build/npu-sim --output-dir results/sweep
```

The built-in cases compare a one-slot baseline with additional buffering, then independently double compute capacity or memory bandwidth relative to that two-slot configuration. A combined configuration changes both resources. Every ratio names its baseline and keeps the workload fixed. CSV also retains each configuration's ratio to the serial baseline. JSON retains raw machine-readable results; CSV makes individual cases easy to inspect.

## Simulator execution overhead

```sh
python3 scripts/compare_modes.py --binary build/npu-sim --repeats 5 --output-dir results/modes
```

Event and cycle modes must have identical modeled cycles and outputs. Compare their host-runtime medians separately. The report makes no automatic claim that either mode always runs faster, and never describes host-runtime ratios as accelerator speedups.
