#pragma once

#include "npu/core.hpp"
#include <systemc>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace npu {

// External memory behind one interface. The accelerator issues a MemRequest per
// panel load or tile store and blocks until it completes; the address runs
// follow the map in core.hpp.
//
// serial: one non-preemptive DMA server for reads and writes. Each request is
//   latency + ceil(burst / bytes_per_cycle) per burst, bursts back to back;
//   when both directions wait, the server alternates.
// axi: per core one AXI4 port with a read channel (AR/R) and a write channel
//   (AW/W/B), each moving one data_bytes beat per cycle. Runs split into INCR
//   bursts (next_burst in core.hpp). For burst i of a channel, with N its
//   outstanding limit:
//     issue_i      = max(arrival, issue_{i-1} + 1, completion_{i-N})
//     start_i      = max(issue_i + read_latency (reads only), data_free, dram_free_i)
//     data_free    = start_i + beats_i, stretched to the DRAM end of the burst
//                    when DRAM is slower than the bus
//     completion_i = data_free (+ write_latency for writes, the B response)
//   A request completes with its last burst. Every beat moves data_bytes over
//   DRAM, reserved on a timeline in integer byte units shared by all channels:
//   cycle c begins at byte c * dram_bytes_per_cycle, and a burst takes the
//   earliest gap at or after its data phase that holds all its bytes. One
//   arbiter grants the bursts of a cycle once every process has run in it, in
//   core order and read before write, so SystemC evaluation order never
//   decides a grant.

enum class MemDirection : std::uint8_t { Read, Write };

// One DMA transfer. The issuing thread owns it, as a member of its module,
// for as long as it is in flight: a coroutine stack can be released without
// unwinding when the simulation stops paused. The port keeps a pointer only
// until completion.
struct MemRequest {
    MemDirection direction = MemDirection::Read;
    std::vector<AddressRun> runs;
    std::uint64_t tile_id = 0, block_id = 0;
    // Issuing core. The axi arbiter grants same-cycle bursts in core order.
    std::uint64_t requester = 0;
    // Set by the port: payload bytes; the cycles of arrival, service start and
    // completion; and the bursts, beats and data-channel cycles it used.
    std::uint64_t bytes = 0, arrival = 0, start = 0, completion = 0;
    std::uint64_t bursts = 0, beats = 0, data_cycles = 0;
    sc_core::sc_event done;
};

// Counters commit when a request completes.
struct ChannelStatistics {
    std::uint64_t requests = 0, bursts = 0, beats = 0, payload_bytes = 0;
    // Cycles the data channel moves beats.
    std::uint64_t busy_cycles = 0;
    // Request latency, arrival to completion: the sum and the largest.
    std::uint64_t latency_cycles = 0, max_latency_cycles = 0;
    std::uint64_t peak_outstanding = 0;
};

struct ChannelView {
    // A request is in flight.
    bool active = false;
    // Bursts issued and not complete, and requests waiting to start.
    std::uint64_t outstanding = 0, queued = 0;
    ChannelStatistics statistics;
};

struct MemoryView {
    ChannelView read, write;
    // The request in service, or the last one started, and its direction;
    // with both channels active, the read.
    std::uint64_t tile_id = 0, block_id = 0;
    bool reading = false;
    // Cycles with a request in flight on either channel, including the
    // elapsed part of requests in flight.
    std::uint64_t busy_cycles = 0;
    // axi: bytes moved over DRAM, beats * data_bytes.
    std::uint64_t dram_bytes = 0;
};

class MemoryPort {
public:
    virtual ~MemoryPort() = default;
    // Blocks the calling SC_THREAD until `request` completes.
    virtual void transport(MemRequest& request) = 0;
    // The port of one requester. The serial model has requester 0 only;
    // dram_bytes counts that port's beats.
    virtual MemoryView view(std::uint64_t requester) const = 0;
};

// The ports of a cluster as one: counters and live counts add, the latency and
// outstanding peaks take the largest port, and the request shown is that of the
// lowest-numbered port with a channel active, else port 0's. One port is
// returned unchanged.
MemoryView combine(const std::vector<MemoryView>& ports);

// Called from the process that changed a port's view, at request boundaries.
using MemoryListener = std::function<void(std::uint64_t requester)>;

// The model config.arch.memory.model selects, as a child of the module under
// construction: the serial server, or an axi port per core of arch.cores.
std::unique_ptr<MemoryPort> make_memory(const Config& config, MemoryListener changed);

}  // namespace npu
