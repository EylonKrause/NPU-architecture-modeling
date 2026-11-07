#include "npu/memory.hpp"

#include <algorithm>
#include <stdexcept>

namespace npu {
namespace {

std::uint64_t now() { return sc_core::sc_time_stamp().value(); }

void advance(TimingMode mode, std::uint64_t cycles) {
    if (mode == TimingMode::Cycle) {
        for (std::uint64_t i = 0; i < cycles; ++i) sc_core::wait(sc_core::sc_time::from_value(1));
    } else {
        sc_core::wait(sc_core::sc_time::from_value(cycles));
    }
}

void begin(MemRequest& request) {
    request.bytes = 0;
    for (const auto& run : request.runs) request.bytes = checked_add(request.bytes, run.length);
    request.arrival = request.start = request.completion = now();
    request.bursts = request.beats = request.data_cycles = 0;
}

void commit(ChannelStatistics& stats, const MemRequest& request) {
    ++stats.requests;
    stats.bursts = checked_add(stats.bursts, request.bursts);
    stats.beats = checked_add(stats.beats, request.beats);
    stats.payload_bytes = checked_add(stats.payload_bytes, request.bytes);
    stats.busy_cycles = checked_add(stats.busy_cycles, request.data_cycles);
    const auto latency = request.completion - request.arrival;
    stats.latency_cycles = checked_add(stats.latency_cycles, latency);
    stats.max_latency_cycles = std::max(stats.max_latency_cycles, latency);
}

// One non-preemptive server for both directions. A transfer is split into
// burst_bytes bursts, each paying the latency before ceil(burst /
// bytes_per_cycle) data cycles (its beats).
class SerialDma final : public sc_core::sc_module, public MemoryPort {
public:
    SerialDma(sc_core::sc_module_name name, const Config& config, sc_core::sc_event& changed)
        : sc_module(name), config_(config), changed_(changed) {
        SC_THREAD(serve);
    }

    void transport(MemRequest& request) override {
        begin(request);
        (request.direction == MemDirection::Read ? reads_ : writes_).push_back(&request);
        changed_.notify(sc_core::SC_ZERO_TIME);
        ready_.notify(sc_core::SC_ZERO_TIME);
        sc_core::wait(request.done);
    }

    MemoryView view() const override {
        MemoryView view;
        view.read.active = busy_ && read_;
        view.write.active = busy_ && !read_;
        view.read.outstanding = view.read.active ? 1 : 0;
        view.write.outstanding = view.write.active ? 1 : 0;
        view.read.queued = static_cast<std::uint64_t>(reads_.size());
        view.write.queued = static_cast<std::uint64_t>(writes_.size());
        view.read.statistics = read_stats_;
        view.write.statistics = write_stats_;
        view.tile_id = tile_id_;
        view.block_id = block_id_;
        view.reading = read_;
        view.busy_cycles = busy_ ? checked_add(busy_cycles_, std::min(now() - start_, duration_)) :
                                   busy_cycles_;
        return view;
    }

private:
    const Config& config_;
    sc_core::sc_event& changed_;
    sc_core::sc_event ready_;
    std::deque<MemRequest*> reads_, writes_;
    ChannelStatistics read_stats_, write_stats_;
    bool prefer_read_ = true, busy_ = false, read_ = false;
    std::uint64_t start_ = 0, duration_ = 0, busy_cycles_ = 0;
    std::uint64_t tile_id_ = 0, block_id_ = 0;

    std::uint64_t data_cycles(std::uint64_t bytes) const {
        const auto& memory = config_.arch.memory;
        const auto burst = memory.burst_bytes;
        if (!burst) return ceil_div(bytes, memory.bytes_per_cycle);
        const auto tail = bytes % burst;
        return checked_add(checked_mul(bytes / burst, ceil_div(burst, memory.bytes_per_cycle)),
                           tail ? ceil_div(tail, memory.bytes_per_cycle) : 0);
    }

    void serve() {
        const auto burst = config_.arch.memory.burst_bytes;
        while (true) {
            if (reads_.empty() && writes_.empty()) sc_core::wait(ready_);
            // Load and store each queue within two delta cycles of the cycle
            // they wake in; waiting two deltas sees both before choosing.
            // Delta cycles cost no architecture time.
            sc_core::wait(sc_core::SC_ZERO_TIME);
            sc_core::wait(sc_core::SC_ZERO_TIME);
            if (reads_.empty() && writes_.empty()) continue;
            const bool read = !reads_.empty() && (writes_.empty() || prefer_read_);
            auto& queue = read ? reads_ : writes_;
            MemRequest* request = queue.front();
            queue.pop_front();
            auto& stats = read ? read_stats_ : write_stats_;
            stats.peak_outstanding = 1;
            start_ = request->start = now();
            duration_ = dma_transfer_cycles(config_, request->bytes);
            tile_id_ = request->tile_id;
            block_id_ = request->block_id;
            read_ = read;
            busy_ = true;
            changed_.notify(sc_core::SC_ZERO_TIME);
            advance(config_.sim.timing, duration_);
            request->completion = now();
            request->bursts = burst ? ceil_div(request->bytes, burst) : 1;
            request->beats = request->data_cycles = data_cycles(request->bytes);
            commit(stats, *request);
            busy_cycles_ = checked_add(busy_cycles_, duration_);
            busy_ = false;
            changed_.notify(sc_core::SC_ZERO_TIME);
            prefer_read_ = !read;
            request->done.notify(sc_core::SC_ZERO_TIME);
            // The requester owns the request; it may reuse it once resumed.
        }
    }
};

}  // namespace

std::unique_ptr<MemoryPort> make_memory(const Config& config, sc_core::sc_event& changed) {
    return std::make_unique<SerialDma>("memory", config, changed);
}

}  // namespace npu
