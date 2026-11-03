# SystemC execution and observability

The executable runs a real Accellera SystemC discrete-event kernel. Its NPU
model has concurrent load, compute, store, and shared-DMA `SC_THREAD` processes.
They exchange jobs and end-to-end buffer tokens through bounded `sc_fifo`
channels; `sc_event` notifications implement DMA completion, arbitration, and
reduction-panel handoff. Blocking FIFOs model backpressure. The model performs
quantized integer inference and independently checks the resulting tensors.

This is an executable architecture model of an NPU, not a physical NPU device
or an RTL implementation. `event` timing skips directly to stage completions;
`cycle` timing advances the same stage contracts one architecture cycle at a
time. Cycle stepping does not add register-transfer or electrical detail.
See [the implementation contract](IMPLEMENTATION-CONTRACT.md) for the exact
memory, compute, cache, and overlap assumptions.

## Native waveform output

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/npu-sim --m 16 --n 16 --k 32 --tile-k 8 \
  --buffer-slots 2 --vcd npu.vcd --trace npu.csv --dump-output
```

`npu.vcd` is written by SystemC's `sc_create_vcd_trace_file` and `sc_trace`,
using actual `sc_signal` values. Open it in a waveform viewer such as GTKWave.
`npu.csv` remains the stage-completion trace, and stdout remains a single final
JSON result. The two trace formats observe the same simulation.

| Signal | Meaning |
| --- | --- |
| `done` | All output tiles have finished their stores. |
| `dma_busy`, `dma_read` | DMA ownership and read versus write direction. Direction is meaningful while busy. |
| `compute_busy` | The compute resource owns a reduction panel. |
| `dma_tile_id`, `dma_block_id` | Active DMA tile and reduction panel; meaningful while DMA is busy. |
| `compute_tile_id`, `compute_block_id` | Active compute tile and panel; meaningful while compute is busy. |
| `inflight_tiles` | End-to-end slots owned by tiles, including tiles waiting on resources. |
| `loaded_tiles`, `computed_tiles` | Jobs available in the two bounded interstage FIFOs. |
| `read_queue_depth`, `write_queue_depth` | Requests waiting for DMA arbitration. |
| `read_bytes`, `write_bytes`, `macs` | Completed-operation counters. |
| `dma_requests`, `dma_bursts` | Completed DMA requests and modeled burst count. |
| `dma_busy_cycles`, `compute_busy_cycles` | Accumulated occupancy, including elapsed time in an active operation when sampled. |
| `cycles` | Current architecture-cycle timestamp when a state transition is observed. |

The kernel uses one abstract nanosecond per architecture cycle. VCD uses
picosecond resolution, allowing delta-cycle transitions to be drawn between
those cycle boundaries. These fine trace offsets annotate zero-time ordering;
they are not physical circuit delays. Read the `cycles` signal or CSV cycle
columns for architecture timing. `--frequency-mhz` converts cycle counts to
physical-time estimates in the result report; it does not calibrate the model
against hardware. Event mode has no synthetic free-running clock. Its waveform
changes when modeled state changes, which keeps tracing useful without creating
an event for every inactive cycle.

A single passive `SC_METHOD` writes every observable signal. Scheduling threads
update their own ordinary state and notify `state_changed_` at zero simulation
time. The observer introduces delta-cycle evaluation only: it never waits,
changes the scheduler queues, claims a resource, or adds architecture cycles.
Counters for bytes and MACs commit when their operation finishes, rather than
inventing a per-cycle transfer or execution distribution inside a coarse stage.
Delta-cycle tracing preserves final same-time updates, including the final
DMA counters and `done` transition, before the simulation becomes quiescent.
Use the CSV trace for exact completed-operation identities.

## Live execution protocol

```sh
./build/npu-sim --m 4 --n 5 --k 7 --tile-m 2 --tile-n 3 --tile-k 3 \
  --live-stdio --vcd live.vcd --progress-jsonl live.jsonl
```

The process emits one initial JSON snapshot at cycle zero, then waits for
newline-delimited commands on stdin. Simulation time does not advance while
the host waits for a command. All commands and kernel execution run on the
same thread.

| Command | Response |
| --- | --- |
| `advance 100` | Run the actual SystemC kernel for the requested slice and return a snapshot. After completion, retain the final model timestamp. |
| `snapshot` | Return the current paused state without advancing time. |
| `stop` | Return a snapshot with `type: "stopped"` and exit successfully. |

Advance counts must be in `1..10000000`. Each input line is bounded to 256
characters. Invalid commands return an error object without advancing or
terminating the simulation. Closing stdin exits cleanly. Each response is
flushed immediately, so a frontend can issue a command and await one complete
JSON line; it does not need to scrape a terminal display.

Snapshots expose the actual resource state, tile/panel completion counts,
queue occupancy, byte/MAC counters, utilization, and input preview. A completed
snapshot includes the output preview and deterministic output hash. Before
completion, `numerical_match` and `output_hash` are `null`, `validated` is
`false`, and the output preview is empty. Completion is reported as validated
only after the actual output tensors agree with the separate numerical
reference. A live session's optional progress file receives the same state
snapshots; command errors remain on the response stream.

For an unattended run with streaming progress and the usual final JSON output:

```sh
./build/npu-sim --progress-jsonl progress.jsonl --slice-cycles 64 \
  --vcd batch.vcd --dump-output > result.json
```

`--slice-cycles` defaults to 1024 and accepts the same bounds as `advance`.
The progress file starts at zero and ends at the exact architecture completion
cycle, even when the last host slice extends beyond completion. Model timing,
final tensors, and the existing result JSON are unchanged by progress or VCD
instrumentation. The slice size controls observation cadence rather than
architecture timing detail.

## Embedding and live snapshots

The public C++ interface supports an application that advances the kernel in
bounded slices and displays progress:

```cpp
sc_core::sc_set_time_resolution(1, sc_core::SC_NS);
npu::Accelerator accelerator("npu", config, problem);
auto* waveform = sc_core::sc_create_vcd_trace_file("npu");
waveform->set_time_unit(1, sc_core::SC_PS);
sc_core::sc_trace_delta_cycles(waveform, true);
accelerator.trace_signals(waveform);
while (!accelerator.completed()) {
    sc_core::sc_start(sc_core::sc_time::from_value(100));
    const npu::LiveSnapshot current = accelerator.snapshot();
    // Read current.cycles, current.inflight_tiles, current.statistics, etc.
    // An embedding application must also detect starvation/deadlock.
}
sc_core::sc_close_vcd_trace_file(waveform);
```

Call `trace_signals` before the first `sc_start`. The caller owns the trace file.
Call `snapshot` from the kernel's host thread while `sc_start` is paused, or
from a SystemC process. Do not access it from a concurrent web-server or GUI
thread: SystemC kernel objects and these counters are not synchronized for
cross-thread reads. A frontend can receive copied snapshots through its own
host-side channel. Snapshot busy counters include elapsed portions of active
operations; final `statistics()` values retain their existing completion-only
contract. Once completed, the snapshot timestamp stays at the actual completion
cycle even if a bounded host `sc_start` call ran to a later deadline. Snapshot
FIFO depths are the channel's currently available entries,
so their state settles through normal SystemC delta cycles.

## Passive invariants and verification

At each observed state change, the monitor checks that tile and interstage
occupancy stay within the configured slot count, cached weight storage stays
within its configured capacity, busy resources have an owning tile, and a
completed model has no outstanding resource ownership or queued DMA requests.
An actual violation triggers `SC_REPORT_FATAL("NPU/invariant", ...)`; the
monitor does not repair, clamp, or hide invalid state.

`tests/systemc_observability_tests.py` launches the real executable and parses
the generated VCD independently. It verifies signal declarations, resource
activity, occupancy limits, final byte/MAC/burst totals, and completion. It also
compares traced versus untraced output and event versus cycle timing. Existing
independent arithmetic and architecture suites remain the stronger checks of
the numerical and scheduling contracts.

`tests/live_protocol_tests.py` drives a real child process with bounded response
waits. It verifies pause and step behavior, recoverable command errors, clean
EOF, live and automatic progress, and completed output equality with batch
execution. It also checks that a stopped or completed model does not falsely
report further computation.

## Scope of the SystemC model

`npu-sim` covers quantized dense kernels and their overlapping pipeline
scheduling. The graph runtime lowers supported convolution and graph
operators around these runs. The separate [TLM platform](TLM-PLATFORM.md),
`npu-tlm-demo`, provides actual timed memory transactions, an MMIO register
target, a clock, reset, interrupts and repeated jobs within one kernel.
Its serial platform timing contract differs from the overlapped pipeline.
Both models retain explicit limits: no AXI pin-level implementation, cache
coherence, CPU instruction-set execution, physical memory-controller model,
or synthesizable MAC-array datapath. Such interfaces require their own
architectural contracts and conformance tests.

Primary references:

- [Accellera SystemC standards and downloads](https://www.accellera.org/downloads/standards/systemc)
- [SystemC 3.0.2 reference implementation](https://github.com/accellera-official/systemc/tree/3.0.2)
- [SystemC tracing implementation](https://github.com/accellera-official/systemc/tree/3.0.2/src/sysc/tracing)

