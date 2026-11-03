# Live SystemC execution

The live dashboard runs the compiled C++/SystemC accelerator and displays its
actual progress. Each inference computes signed integer matrix multiplication,
requantizes outputs, and compares the complete accumulator and output vectors
with the scalar reference. It is a running **architectural simulator**, not a
physical NPU device or a commercial NPU ISA emulator.

## Run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
python3 scripts/live_dashboard.py --binary build/npu-sim
```

Open **http://127.0.0.1:8765** in a browser on the same machine. A different local
port can be selected with `--port 8766`. The controller uses Python's standard
library; no web package, browser extension, cloud connection or external asset
is required. Stop the server with Ctrl+C. It closes only its own child simulator.
The server starts idle; press **Start continuous inference** to run workloads.

The controls have explicit semantics:

| Control | Effect |
| --- | --- |
| Start | Starts the selected workload and repeatedly runs independently validated inferences. |
| Pause | Stops issuing simulation advances. Simulated time and counters freeze. |
| Resume | Continues issuing advances to the same paused SystemC process. |
| Step | Advances the paused model by at most 256 target cycles, or until completion. |
| Stop | Ends the controller's own simulator child; the final snapshot remains visible. |
| Reset | Clears session counters and creates a fresh paused model with the selected profile. |

Each child models one inference. Once a completed output is validated, continuous
mode starts the next inference in a **new SystemC process** with the next
deterministic seed. This avoids pretending that a stopped SystemC kernel can be
restarted or that an unchanged tensor represents new input. The dashboard keeps
the eight latest output hashes and previews. Model counters reset per inference;
the validated-inference count spans the dashboard session. A failed comparison
stops execution and is reported as an error.

## Workloads and observability

All three built-in profiles use a bounded 64-by-64 output matrix with reduction
length 128, 8-by-8 output tiles, reduction panels of 32 and two SRAM slots:

| Profile | Purpose |
| --- | --- |
| Balanced INT8 | Default compute and shared DMA bandwidth assumptions. |
| Compute pressure | Four MACs per cycle and 128 memory bytes per cycle. |
| Memory pressure | 256 MACs per cycle, four memory bytes per cycle and 24-cycle DMA startup. |

These names describe selected model parameters; they do not assert measured
silicon bottlenecks. The UI reports simulated cycles, completed MACs, DMA bytes,
busy fractions, active stages, actual tile completion and real input/output
previews. The first 16 values are previews, while validation covers the complete
output. There is no synthetic activity animation. Busy fractions divide actual
busy cycles, including elapsed time in an active stage, by simulated cycles.
Completed MACs and byte counters increase when their corresponding stage
finishes, so those counters can move in steps.

The controller advances 256 cycles roughly every 25 milliseconds of host time
and the browser polls at 250-millisecond intervals. This pacing makes execution
inspectable. It is **not** target throughput, hardware speed, or a real-time
deadline guarantee. Target time uses the configured model clock; do not infer
hardware performance from how fast the browser animates.

## Local API

`GET /api/state` returns the current state, latest unmodified model telemetry and
bounded recent-job history. `GET /api/profiles` lists fixed workload profiles.

`POST /api/control` accepts an `application/json` object:

```json
{"action":"start","profile":"balanced"}
```

Actions are `start`, `pause`, `resume`, `step`, `stop` and `reset`. Only start/reset
accept `profile`. Step accepts an integer `cycles` from 1 through 10,000; the UI
uses 256. Arbitrary fields, shell commands, paths and executable selections are
rejected. The executable path is a trusted command-line setting at server
startup, never an HTTP parameter.

The server binds only IPv4 loopback, checks the HTTP Host header to resist DNS
rebinding, rejects cross-origin controls and CORS preflights, and accepts only
bounded JSON control bodies. It has a restrictive content-security policy and
does not load third-party content. It intentionally does not expose a remote
control endpoint. Do not put it behind a public reverse proxy without adding an
appropriate authenticated boundary.

## Verification

```sh
python3 tests/live_dashboard_tests.py --binary build/npu-sim -v
```

Tests use the actual compiled SystemC executable. They verify increasing model
time, frozen paused state, exact stepping before completion, resumption, owned
child teardown, both full-output validation and changing deterministic inputs,
all three profiles, rejected arbitrary commands, and HTTP browser boundaries.

The existing batch, graph and ONNX paths remain available; this dashboard is a
live view of the SystemC matrix accelerator, not a replacement graph compiler.
