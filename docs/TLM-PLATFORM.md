# SystemC TLM 2.0 coprocessor platform

`npu-tlm-demo` is a persistent, working SystemC virtual coprocessor. A host
`SC_THREAD` sends actual activation, weight, and bias bytes through TLM memory
transactions, programs MMIO, and waits for a real SystemC interrupt signal.
An independent engine `SC_THREAD` DMA-reads those bytes, executes tiled INT8
dense arithmetic, DMA-writes INT32 accumulators and requantized INT8 outputs,
and raises completion. The host reads **all output bytes** back and checks them
against the scalar reference. It acknowledges the interrupt and submits another
job without reconstructing or restarting the SystemC kernel.

This is an executable architecture model. It does not create a physical NPU,
implement a commercial instruction set, or establish RTL cycle accuracy.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/npu-tlm-demo --jobs 2 --dump-output --vcd tlm-run
python3 tests/tlm_platform_tests.py --binary build/npu-tlm-demo
```

The JSON report includes complete-job numerical validation, interrupt count,
host/DMA transaction counts, actual transferred bytes, compute cycles, protocol
checks, and optional full output arrays. `--seed UINT32` selects deterministic
input values; `--jobs 1..100` submits repeated workloads. `--vcd BASENAME` writes
`BASENAME.vcd`, containing the clock, reset, IRQ, BUSY, DONE, and completion count.
The default clock is 200 MHz. The demo stops after the requested jobs; SystemC
simulation speed is independent of wall-clock time.

## Interfaces

The host and engine use `tlm_utils::simple_initiator_socket`; MMIO and both DRAM
ports use `tlm_utils::simple_target_socket`. Transactions use
`tlm::tlm_generic_payload` and synchronous `b_transport`. Every response is
checked. Initiators consume annotated timing with `wait()` inside their
`SC_THREAD`; targets do not call `wait()`.

| Register offset | Access | Meaning |
| --- | --- | --- |
| `0x00`, `0x04`, `0x08` | Read | Fixed M=3, N=5, K=7 dimensions |
| `0x0c` | Write | START=1; rejected while BUSY or before prior DONE is acknowledged |
| `0x10` | Read | STATUS: bit 0 BUSY, bit 1 DONE |
| `0x14` | Read/write | IRQ enable, values 0 or 1 |
| `0x18` | Write | IRQ_ACK=1 clears DONE; rejected while BUSY |
| `0x1c` | Read | Completed-job count since reset |

Registers require aligned, four-byte, little-endian accesses. Writing a
read-only register, reading a write-only register, invalid command/value,
misaligned/unmapped address, byte enables, and malformed burst/streaming width
return explicit TLM error responses. No invalid operation changes register
state. The host self-test deliberately issues protocol errors, counts expected
rejections separately, and fails if any unexpected error occurs.

| DRAM address | Data |
| --- | --- |
| `0x0000` | Row-major A, 21 signed INT8 elements |
| `0x1000` | Row-major W, 35 signed INT8 elements |
| `0x2000` | Five signed INT32 biases, little endian |
| `0x3000` | Fifteen signed INT8 outputs |
| `0x4000` | Fifteen signed INT32 accumulators, little endian |

DRAM is 64 KiB and supports byte-granular reads/writes in bursts of 1–64 bytes.
Out-of-range accesses, byte enables, wrapping streaming bursts, and unsupported
commands are rejected. DMI is explicitly denied on every socket. Read-only
`transport_dbg` permits inspecting storage without advancing simulation time
or regular-traffic counters; it is not a timing shortcut used by inference.

## Reset, interrupt, and numerical contract

The host holds active-low reset for three clock edges and releases it before
programming MMIO. Reset clears BUSY, DONE, IRQ enable, and completed-job count.
The engine thread uses SystemC asynchronous reset; an IRQ/busy/done observation
method is the sole writer of trace/output signals. DRAM contents survive reset,
as external memory commonly does. The demo tests startup reset, not an
inference interrupted by reset.

The first job enables IRQ before START. Later jobs mask the IRQ, poll DONE,
verify the IRQ remains low, then enable it and observe its assertion. Completion
is set only after both output DMA writes complete. ACK clears DONE and IRQ;
the next START is accepted by the same engine thread. Changing the host input
bytes between jobs produces a different verified result.

The fixed arithmetic uses input zero point -3, weight zero point 2, output zero
point 0, multiplier 1, right shift 4 with ties-to-even rounding, and INT8
saturation. The full INT32 sum includes bias. Output tiles are 2 by 3, including
edge tiles. The scalar host oracle and an independent Python integer test check
every accumulator and output, rather than relying on a hash alone.

## Timing contract and limits

This platform is **loosely timed and serial**. It loads a complete problem,
computes its tiles sequentially, then stores outputs. MMIO costs one clock
period. Each DRAM burst reserves a shared bus timeline for three latency cycles
plus payload cycles at 16 bytes/cycle. Each compute tile costs its integer MAC
work at four MACs/cycle plus two pipeline cycles, as specified by the core.
The host performs an additional read from a separate scratch region during
each inference, forcing host/engine contention; the JSON reports accumulated
memory-arbitration wait cycles. The read does not alter the active tensors.
These compute delays are analytical; they do not model individual MAC RTL
transitions. Both host and engine initiators consume memory delay. Memory
effects occur in call order, as allowed by this restricted loosely timed model;
the supported host protocol does not modify tensors during an active job.

These elapsed cycles intentionally differ from `npu-sim`'s overlapped DMA and
compute pipeline. Neither executable claims bit-/cycle-equivalence to a
production device without calibration. This platform does not model a CPU ISA,
cache coherency, IOMMU, AXI beat signals, an operating system, power rails, clock
domain crossings, arbitrary tensor descriptors, or an instruction compiler.
Those require additional architectural contracts and validation evidence.

Official background:
[Accellera SystemC and TLM standards](https://www.accellera.org/downloads/standards/systemc),
[SystemC reference implementation](https://github.com/accellera-official/systemc).
