#pragma once

#include "npu/core.hpp"
#include <systemc>
#include <cstdint>
#include <deque>
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

enum class MemDirection : std::uint8_t { Read, Write };

// One DMA transfer. The issuing thread owns it, as a member of its module,
// for as long as it is in flight: a coroutine stack can be released without
// unwinding when the simulation stops paused. The port keeps a pointer only
// until completion.
struct MemRequest {
    MemDirection direction = MemDirection::Read;
    std::vector<AddressRun> runs;
    std::uint64_t tile_id = 0, block_id = 0;
    // Issuing core.
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
};

class MemoryPort {
public:
    virtual ~MemoryPort() = default;
    // Blocks the calling SC_THREAD until `request` completes.
    virtual void transport(MemRequest& request) = 0;
    virtual MemoryView view() const = 0;
};

// The model config.arch.memory.model selects, as a child of the module under
// construction. `changed` gets a delta notification whenever the view changes
// at a request boundary.
std::unique_ptr<MemoryPort> make_memory(const Config& config, sc_core::sc_event& changed);

}  // namespace npu
