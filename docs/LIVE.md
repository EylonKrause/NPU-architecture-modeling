# Live SystemC execution

The live dashboard runs the compiled C++/SystemC accelerator and displays its
progress. Each inference computes signed integer matrix multiplication,
requantizes outputs, and compares the complete accumulator and output vectors
with the scalar reference. It is a running **architectural simulator**, not a
physical NPU device or a commercial NPU ISA emulator.

## One-command launch

```sh
python3 run.py
```

On Windows use `py -3 run.py`. This builds and checks the native model, starts
continuous inference and opens the dashboard. One-time compiler/CMake setup and
the initial SystemC download are explained in the [quick-start guide](QUICKSTART.md).
Use `--no-browser` on a headless machine or `--port 8766` if the default port is
busy. The launcher uses a separate `build-live` tree.

## Manual build and launch

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
python3 scripts/live_dashboard.py --binary build/npu-sim
```

Open **http://127.0.0.1:8765** in a browser on the same machine. A different local
port can be selected with `--port 8766`. The controller uses Python's standard
library; no web package, browser extension, cloud connection or external asset
is required. Stop the server with Ctrl+C. It closes only its own child simulator.
The server starts idle; choose a profile and press **Run**.

| Control | Effect |
| --- | --- |
| Run | Starts the selected profile and runs validated inferences back to back. |
| Pause | Stops issuing simulation advances. Simulated time and counters freeze. |
| Resume | Continues issuing advances to the same paused SystemC process. |
| Step | Advances the paused model by the cycle count beside it (1 to 10,000), or until the inference completes. |
| Stop | Ends the controller's own simulator child; the final snapshot remains visible. |
| Reset | Clears session counters and creates a fresh paused model with the selected profile. |

Each child models one inference. Once a completed output is validated, the next
inference starts in a **new SystemC process** with the next deterministic seed:
a stopped SystemC kernel cannot be restarted, and an unchanged tensor would not
be a new input. The controller keeps the eight latest completed snapshots, output
hashes included, in `recent_jobs`; the Jobs table shows their cycles, rates and
validation. Model counters reset per inference; the validated-inference count
spans the dashboard session. A failed comparison stops execution and is reported
as an error.

## Workloads

All four built-in profiles use a 64-by-64 output matrix with reduction length
128, 8-by-8 output tiles, reduction panels of 32 and two SRAM slots. The profile
menu shows each label with the bound class the analytic model gives it:

| Profile key | Menu label | Parameters | Analytic bound |
| --- | --- | --- | --- |
| `balanced` | Default machine | 64 MAC/cycle, 16 B/cycle, 12-cycle DMA latency | latency |
| `compute-bound` | 4 MAC/cycle | 4 MAC/cycle, 128 B/cycle | compute |
| `memory-bound` | 4 B/cycle | 256 MAC/cycle, 4 B/cycle, 24-cycle DMA latency | bandwidth |
| `palette-weights` | 2-bit palette weights | INT8 fixture with four weight values stored as packed 2-bit indices plus per-panel LUTs; decode cycles charged; activations change with each job seed | none (no analytic model for palette weights) |

The labels name model parameters; they do not describe measured silicon.

## Panels

| Panel | Shows | Source |
| --- | --- | --- |
| Title bar | Shape, tile, slots, panel buffers, peak rates, DMA latency, weight format, clock | `workload`, `machine` |
| Cycles, MAC/cycle, DRAM B/cycle, arithmetic intensity | Current job. "% of peak" divides executed MACs per cycle by the array peak; MAC/cycle and intensity count every MAC of the workload | `telemetry`, `machine` |
| Bound | Analytic bound class, the analytic lower bound in cycles and the share of it the last completed job attained. Without an analytic model (palette weights) it shows a dash and the MAC and DMA busy fractions | `machine.analysis` |
| Roofline | Compute roof, DRAM roof, the latency-inclusive DRAM roof when request latency lowers it, completed jobs, the current job and the analytic lower bound | `machine`, `recent_jobs` |
| Utilisation | MAC busy and DMA busy fractions of each advance over the last 200 advances. The snapshot counts DMA busy cycles for loads and stores together, so the chart shows the total. Focus the chart and use the arrow keys (or hover) to read one slice; End returns to the newest | `slices` |
| Pipeline | Tile in each of the load, compute and store stages over the latest six tiles | `timeline` |
| Output tiles | State of each output tile | `telemetry` |
| Jobs | Cycles, time, MAC/cycle, % of peak, % of the analytic lower bound, DRAM bytes and validation of the latest completed jobs | `recent_jobs`, `machine` |
| Counters | Every numeric snapshot counter; nested objects and arrays of objects (memory channels, per-core state) appear under their own names when a snapshot carries them | `telemetry` |

The machine description comes from the snapshot config when it echoes the whole
design (`machine.source` is `snapshot`). A schema_version 2 snapshot does: its
`config` is the complete configuration nested by option-table path
(`shape.m`, `arch.compute.macs_per_cycle`, `mapping.panel_buffers`, ...). A
schema_version 1 snapshot from an older build echoes the shape, tiling and weight
format only, so the controller completes the design from the profile's flags and
that simulator's behaviour for keys it does not echo: one panel buffer per slot,
one group, serial memory and one core (`machine.source` is `profile`).

The pipeline chart is sampled. Every advance samples the stages each 8 cycles
over its last 512 cycles, so a stage change appears up to 8 cycles late. A step
longer than 512 cycles runs its earlier cycles in one advance; the chart hatches
that interval as not sampled instead of drawing stages across it. Exact stage
times need stage-change events from the simulator.

Busy fractions divide busy cycles, including elapsed time in an active stage, by
simulated cycles. Completed MACs and byte counters increase when their stage
finishes, so those counters can move in steps. The first 16 input and output
values are previews; validation covers the complete output.

The controller advances 256 cycles roughly every 25 milliseconds of host time
and the browser polls at 250-millisecond intervals. This pacing makes execution
inspectable. It is **not** target throughput, hardware speed, or a deadline
guarantee. Target time uses the configured model clock.

## Local API

`GET /api/profiles` lists the fixed workload profiles with `label`,
`description` and analytic `bound` (null without an analytic model).

`GET /api/state` returns the last published view; it never waits on the
simulator. Fields:

| Field | Content |
| --- | --- |
| `state`, `profile`, `error` | Controller state (`idle`, `running`, `paused`, `stopped`, `error`), selected profile, last error |
| `telemetry` | Latest unmodified npu-sim snapshot (schema_version 2; 1 from older builds) |
| `workload` | `m`, `n`, `k`, `groups`, `weight_bits`, `weight_encoding`, `frequency_mhz` resolved from the snapshot config (null when absent) |
| `machine` | `source`, peak `macs_per_cycle` and `bytes_per_cycle`, `ridge`, `memory_model`, `memory_latency`, `pipeline_cycles`, `buffer_slots`, `panel_buffers`, `cores`, tile sizes, `frequency_mhz`, and `analysis` (null without an analytic model): `bound`, `lower_bound_cycles`, `resource_bound_cycles`, `estimate_cycles`, `compute_busy_cycles`, `memory_cycles`, `dram_bytes`, `arithmetic_intensity`, `effective_bytes_per_cycle` |
| `slice` | Counter deltas over the last advance |
| `slices` | The last 200 advances: `job`, `end_cycle`, `cycles`, `compute_busy_cycles`, `dma_busy_cycles`, their `_fraction`s, and `executed_macs`, `read_bytes`, `write_bytes` deltas |
| `timeline` | `stride_cycles` and up to 512 `transitions`: `job`, `cycle` and the tile in `load`, `compute` and `store` (null when idle). A transition with `unsampled_from` ends an unsampled interval that started at that cycle |
| `max_step` | Largest `cycles` a step accepts |
| `slice_cycles` | Cycles per scheduler tick while running |
| `jobs_completed`, `completed_job_cycles`, `completed_job_macs`, `completed_job_executed_macs` | Session totals over validated jobs |
| `recent_jobs` | The eight latest completed jobs, newest first: `job`, `profile` and the final `telemetry` snapshot |

`POST /api/control` accepts an `application/json` object:

```json
{"action":"start","profile":"balanced"}
```

Actions are `start`, `pause`, `resume`, `step`, `stop` and `reset`. Only start/reset
accept `profile`. Step accepts an integer `cycles` from 1 through 10,000; the page
offers 1, 8, 64, 256, 1024, 4096 and 10,000. Unknown fields, shell commands, paths
and executable selections are rejected. The executable path is a trusted
command-line setting at server startup, never an HTTP parameter.

The server binds only IPv4 loopback, checks the HTTP Host header to resist DNS
rebinding, rejects cross-origin controls and CORS preflights, and accepts JSON
control bodies of at most 4096 bytes. It has a restrictive content-security
policy and loads no third-party content. It exposes no remote control endpoint.
Do not put it behind a public reverse proxy without an authenticated boundary.

## Verification

```sh
python3 tests/live_dashboard_tests.py --binary build/npu-sim -v
```

Tests use the compiled SystemC executable. They verify increasing model time,
frozen paused state, exact stepping before completion, resumption, owned child
teardown, full-output validation with changing deterministic inputs, all four
profiles against their analytic lower bounds, the sampled timeline and its
unsampled intervals, rejected arbitrary commands, and HTTP browser boundaries.

The batch, graph and ONNX paths remain available; this dashboard is a live view
of the SystemC matrix accelerator, not a graph compiler.
