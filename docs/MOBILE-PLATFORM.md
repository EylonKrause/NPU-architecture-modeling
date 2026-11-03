# Mobile NPU system model

`npu-mobile-demo` executes independent quantized dense inference jobs on concurrent SystemC cores. It adds system-level behavior that matters when a phone runs several inference workloads: dispatch priorities, shared-memory contention, per-core SRAM residency, operating points, idle gating, wake-up overhead and deadline observations.

This is a synthetic, configurable architecture. It does not reproduce Apple's Neural Engine or Qualcomm's Hexagon hardware, infer undocumented Apple internals, or predict a particular device's battery life. The arithmetic is executed and checked; scheduling and time are modeled under the documented assumptions.

## Run

```sh
cmake --build build --target npu-mobile-demo -j4
build/npu-mobile-demo --dump-output --vcd results/mobile
```

Standard output is one JSON object. SystemC diagnostics go to standard error. The VCD basename above produces `results/mobile.vcd`; create the parent folder before running. No real-time sleeps or hardware access are involved.

Compare one core against four cores under a compute-bound workload:

```sh
build/npu-mobile-demo --cores 1 --jobs 8 --m 8 --n 8 --k 64 --macs-per-cycle 1
build/npu-mobile-demo --cores 4 --jobs 8 --m 8 --n 8 --k 64 --macs-per-cycle 1
```

Observe frequent arrivals, explicit priorities and a deadline-driven OPP policy:

```sh
build/npu-mobile-demo --jobs 12 --batch-size 3 --arrival-ns 1000 \
  --priorities 0,3,1 --deadline-ns 1000 --governor deadline \
  --opps 100:700,200:800,400:900 --opp 1 --vcd results/mobile-qos
```

The example OPP table is an invented architectural experiment, not measured phone data. Frequencies are in MHz and voltages in mV. Supply a measured OPP table and measured coefficients before interpreting energy comparisons against any physical device.

## What executes

Each core is a separate persistent `SC_THREAD`. A separate release process enqueues jobs at their requested timestamps, a dispatch process assigns them, and one shared bus process serializes memory transfers. Genuine SystemC timed waits permit compute on several cores to overlap while every DRAM transfer remains mutually exclusive.

Jobs contain actual signed INT8 activations and weights and INT32 bias. Each job changes its activation values; a model ID selects shared weights and bias. Workers copy activations and cache-miss weights into their local working tensors after the modeled read completes, execute `npu::compute_tile`, and publish quantized INT8 plus INT32 output after a modeled write. Every output element is compared against the separate scalar `npu::reference` path. Black-box tests also compute full outputs with an independent Python oracle.

Quantization uses input zero point `-3`, weight zero point `2`, output zero point `0`, a power-of-two shift of `4`, and ties-to-even rounding. Arithmetic uses the existing core's checked accumulation and saturation rules. This platform uses dense INT8 jobs; it does not implicitly add new graph operators or represent attention, floating-point arithmetic, or packed INT4 timing.

## Scheduler, QoS and memory contract

- Cores execute one inference at a time, without preemption. Higher numeric priority dispatches first; ties use earliest absolute deadline, then job ID. All releases and completions at a timestamp are gathered before selection.
- Jobs are released in batches. Job ID divided by `--batch-size`, rounded down, selects the batch. Its timestamp is that batch number multiplied by `--arrival-ns`.
- A positive `--deadline-ns` is relative to release. Zero means no deadline. `deadline_misses` counts actual completions after their deadlines; this policy does not promise deadlines or conceal failures.
- DRAM uses FIFO arbitration. Dispatch priority does not preempt a transaction or bypass this FIFO. No guarantee is made about starvation, fairness under an infinite workload, or hard real-time admission.
- A transfer takes the configured bus latency plus the byte count divided by bytes per nanosecond, rounded up. Reads carry activations plus uncached weights and bias. Writes carry one INT8 and one INT32 per output element.
- Each core has a finite, least-recently-used weight-and-bias cache. The model key is explicit; repeated jobs of the same model reuse identical values. An object larger than the cache is fetched on every job. Independent core caches do not provide coherence or a shared last-level cache.
- SRAM validation reserves the activation tensor, both output tensors, and the larger of configured weight-cache capacity and transient weight/bias size. The simulator rejects a configuration that cannot fit. The host's C++ allocation size is distinct from modeled SRAM residency.

`bus_transactions` reports every requested, granted and completed time, byte count, owning core, job ID and read/write direction. These records permit independent checks of serialization, arbitration stalls and accounting. `jobs` reports release, dispatch, compute start/end, completion, response time, OPP and cache hit state. `dispatch_order` records assignment order, which is distinct from completion order.

## OPP and idle states

`--opps` accepts up to 16 `MHz:mV` pairs. Frequencies must increase and voltages must not decrease. `--opp` selects the initial operating point and the fixed governor's point.

The fixed governor uses that point for every job. The deadline governor chooses the lowest point whose estimate fits the remaining deadline, or the highest point if none does. The estimate includes compute cycles, isolated read/write service, wake-up cost when required and OPP-transition cost. It intentionally does not predict future DRAM queue contention or other arrivals; actual misses remain visible.

Compute cycles are the MAC count divided by `--macs-per-cycle`, rounded up, plus `--pipeline-cycles`. The requested compute duration is those cycles multiplied by the selected clock period. The kernel resolves time to one picosecond, so nonintegral durations are quantized to that resolution. This is an event-level model, with no claim of RTL pipeline accuracy or an individual clock pulse for each MAC.

At startup, the compute domain is gated. A dispatched job pays `--wakeup-ns`. Once a job finishes, a core retains an idle state for `--gate-after-ns`; after that interval, its compute domain gates and the next job pays another wake-up. The SRAM cache is explicitly retained across this compute-domain gate. This does not model complete SRAM power loss. Switching OPP adds `--switch-ns` before read/compute service.

VCD state values are:

| Value | Core state |
|---|---|
| 0 | Awake and idle |
| 1 | Waking |
| 2 | OPP transition |
| 3 | Read service or read queue wait |
| 4 | Compute |
| 5 | Write service or write queue wait |
| 6 | Compute domain gated, SRAM retained |

Waveforms also include per-core job, MHz and mV, ready/completed job counts and DRAM ownership. Job IDs and core ownership are offset by one in waveform fields so zero can represent idle. JSON IDs remain zero-based. VCD uses a one-picosecond scale and delta-cycle tracing; consult the JSON timestamps for architectural timing rather than treating VCD's delta pseudo-timesteps as additional hardware cycles.

## Optional energy estimates

Without an explicit energy coefficient flag, `energy` is `null`; the model does not invent silicon energy values. All coefficient defaults are zero. Passing any coefficient enables the energy breakdown, with every omitted term still zero and `silicon_calibrated` remaining false. Coefficients must be finite and nonnegative. For example, the following numbers are only a unit-check experiment:

```sh
build/npu-mobile-demo --jobs 3 --cores 1 --batch-size 1 --arrival-ns 1000 \
  --mac-pj 2 --dram-byte-pj 3 --leakage-mw 5 --retention-mw 7 \
  --gated-mw 1 --wakeup-pj 13 --switch-pj 17
```

For MAC count $N$, selected voltage $V$, reference voltage $V_{\mathrm{ref}}$ and supplied per-MAC coefficient $e_{\mathrm{MAC}}$, the dynamic compute estimate is:

$$E_{\mathrm{compute}}=N e_{\mathrm{MAC}}\left(\frac{V}{V_{\mathrm{ref}}}\right)^2.$$

`--reference-mv` sets the reference voltage; its default is 800 mV. Frequency changes duration, while this user-supplied compute coefficient describes energy per MAC at the reference voltage. DRAM energy is transferred bytes multiplied by `--dram-byte-pj`. Active, retained-idle and gated intervals multiply their separate mW coefficients. Numerically, mW times ns yields pJ. Wake-up and OPP-change counts multiply their respective pJ coefficients. The JSON output includes all supplied/defaulted coefficients and every component of their sum.

Active time includes wake-up, OPP transition, compute, bus service and bus queue waits while the core is awake. Idle accounting covers every core through the common makespan, including a core that receives no job. There is no thermal feedback, temperature-dependent leakage, regulator loss, device calibration, display/RF/CPU/GPU energy or battery model.

## CLI bounds

Run `build/npu-mobile-demo --help` for flags. Jobs are limited to 128 and cores to 8. Dense shapes are bounded to rows/columns 1–64 and reduction dimension 1–256. Per-core SRAM is at most 16 MiB. Arrival, service and power transition inputs also have explicit limits. Total simulated time is bounded independently; malformed configurations fail before launching the kernel. These limits keep examples and tests tractable rather than implying phone hardware capacities.

## Public references and scope

[Qualcomm's Hexagon overview](https://www.qualcomm.com/processors/hexagon) documents low-power inference, combined scalar/vector/tensor resources and dedicated shared memory. [Apple's Core ML deployment session](https://developer.apple.com/videos/play/wwdc2024/10161/) describes dispatch across CPU, GPU and Neural Engine and discusses state and memory efficiency. These motivate studying concurrency, movement and power policy. Neither source supplies the OPP table, dispatch algorithm, memory sizes or timing constants used by this synthetic platform; those are explicit modeling choices. This executable does not claim to implement Core ML, Apple hardware internals or a Hexagon ISA.

The existing pipeline simulator and TLM coprocessor remain separate abstractions. This model adds multicore system experiments rather than silently changing their timing contracts.
