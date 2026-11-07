#include "npu/memory.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

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
    SerialDma(sc_core::sc_module_name name, const Config& config, MemoryListener changed)
        : sc_module(name), config_(config), changed_(std::move(changed)) {
        SC_THREAD(serve);
    }

    void transport(MemRequest& request) override {
        if (request.requester != 0) throw std::invalid_argument("the serial DMA has one port");
        begin(request);
        (request.direction == MemDirection::Read ? reads_ : writes_).push_back(&request);
        changed_(0);
        ready_.notify(sc_core::SC_ZERO_TIME);
        sc_core::wait(request.done);
    }

    MemoryView view(std::uint64_t requester) const override {
        if (requester != 0) throw std::invalid_argument("the serial DMA has one port");
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
    MemoryListener changed_;
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
            changed_(0);
            advance(config_.sim.timing, duration_);
            request->completion = now();
            request->bursts = burst ? ceil_div(request->bytes, burst) : 1;
            request->beats = request->data_cycles = data_cycles(request->bytes);
            commit(stats, *request);
            busy_cycles_ = checked_add(busy_cycles_, duration_);
            busy_ = false;
            changed_(0);
            prefer_read_ = !read;
            request->done.notify(sc_core::SC_ZERO_TIME);
            // The requester owns the request; it may reuse it once resumed.
        }
    }
};

// Per core one AXI port of two channels; one arbiter thread grants every
// channel's bursts and owns the DRAM timeline.
class AxiMemory final : public sc_core::sc_module, public MemoryPort {
public:
    AxiMemory(sc_core::sc_module_name name, const Config& config, MemoryListener changed,
              std::uint64_t ports)
        : sc_module(name), config_(config), changed_(std::move(changed)),
          ports_(static_cast<std::size_t>(ports)), issue_at_(2 * ports_.size(), none) {
        const auto& axi = config.arch.memory.axi;
        for (auto& port : ports_) {
            port.read.limit = axi.read_outstanding;
            port.read.lead = axi.read_latency;
            port.write.limit = axi.write_outstanding;
            port.write.tail = axi.write_latency;
        }
        SC_THREAD(arbitrate);
    }

    void transport(MemRequest& request) override {
        begin(request);
        if (request.requester >= ports_.size())
            throw std::invalid_argument("memory requester has no AXI port");
        const auto index = slot(request.requester, request.direction);
        auto& port = ports_[static_cast<std::size_t>(request.requester)];
        auto& channel = channel_at(index);
        if (port.active++ == 0) port.busy_since = request.arrival;
        ++channel.active;
        channel.tile_id = request.tile_id;
        channel.block_id = request.block_id;
        port.last_read = request.direction == MemDirection::Read;
        if (request.bytes == 0) {
            request.done.notify(sc_core::SC_ZERO_TIME);
        } else {
            channel.queue.push_back(&request);
            if (channel.queue.size() == 1) front(channel);
            refresh(index);
            wake_.notify(sc_core::SC_ZERO_TIME);
        }
        changed_(request.requester);
        sc_core::wait(request.done);
        commit(channel.stats, request);
        --channel.active;
        if (--port.active == 0)
            port.busy_cycles = checked_add(port.busy_cycles, now() - port.busy_since);
        changed_(request.requester);
    }

    MemoryView view(std::uint64_t requester) const override {
        if (requester >= ports_.size())
            throw std::invalid_argument("memory requester has no AXI port");
        const auto time = now();
        const auto& port = ports_[static_cast<std::size_t>(requester)];
        MemoryView view;
        const auto channel_view = [time](const Channel& channel) {
            ChannelView result;
            result.active = channel.active != 0;
            result.outstanding = static_cast<std::uint64_t>(std::count_if(
                channel.recent.begin(), channel.recent.end(),
                [time](std::uint64_t completion) { return completion > time; }));
            result.queued = static_cast<std::uint64_t>(channel.queue.size()) -
                            static_cast<std::uint64_t>(channel.started);
            result.statistics = channel.stats;
            return result;
        };
        view.read = channel_view(port.read);
        view.write = channel_view(port.write);
        view.reading = view.read.active || (!view.write.active && port.last_read);
        const auto& shown = view.reading ? port.read : port.write;
        view.tile_id = shown.tile_id;
        view.block_id = shown.block_id;
        view.busy_cycles = port.active ? checked_add(port.busy_cycles, time - port.busy_since) :
                                         port.busy_cycles;
        view.dram_bytes = checked_mul(checked_add(port.read.stats.beats, port.write.stats.beats),
                                      config_.arch.memory.axi.data_bytes);
        return view;
    }

private:
    struct Channel {
        // Outstanding limit; cycles before the data phase (read latency) and
        // after it (the write response).
        std::uint64_t limit = 1, lead = 0, tail = 0;
        // Requests not fully issued; the front one is issuing.
        std::deque<MemRequest*> queue;
        bool started = false;
        std::size_t run = 0;
        std::uint64_t address = 0;
        bool issued = false;
        std::uint64_t last_issue = 0, data_free = 0;
        // Completion cycles of the last `limit` bursts issued.
        std::deque<std::uint64_t> recent;
        std::uint64_t active = 0, tile_id = 0, block_id = 0;
        ChannelStatistics stats;

        std::uint64_t next_issue() const {
            auto issue = queue.front()->arrival;
            if (issued) issue = std::max(issue, last_issue + 1);
            if (recent.size() == limit) issue = std::max(issue, recent.front());
            return issue;
        }
    };
    struct Port {
        Channel read, write;
        std::uint64_t active = 0, busy_since = 0, busy_cycles = 0;
        bool last_read = false;
    };

    static constexpr auto none = std::numeric_limits<std::uint64_t>::max();
    const Config& config_;
    MemoryListener changed_;
    sc_core::sc_event wake_;
    std::vector<Port> ports_;
    // Next issue cycle of every channel, none while its queue is empty, in
    // grant order: port, then read before write. A channel's issue cycle moves
    // only when a request reaches the front of its queue or a burst issues, so
    // the arbiter scans these instead of every channel's state.
    std::vector<std::uint64_t> issue_at_;
    DramTimeline dram_;

    static std::size_t slot(std::uint64_t requester, MemDirection direction) {
        return static_cast<std::size_t>(2 * requester) +
               static_cast<std::size_t>(direction == MemDirection::Write);
    }

    Channel& channel_at(std::size_t index) {
        auto& port = ports_[index / 2];
        return index % 2 ? port.write : port.read;
    }

    void refresh(std::size_t index) {
        const auto& channel = channel_at(index);
        issue_at_[index] = channel.queue.empty() ? none : channel.next_issue();
    }

    // Points the channel at the first nonempty run of its front request.
    static void front(Channel& channel) {
        const auto& runs = channel.queue.front()->runs;
        channel.run = 0;
        while (runs[channel.run].length == 0) ++channel.run;
        channel.address = runs[channel.run].base;
        channel.started = false;
    }

    void arbitrate() {
        const bool step = config_.sim.timing == TimingMode::Cycle;
        while (true) {
            const auto next = *std::min_element(issue_at_.begin(), issue_at_.end());
            if (next == none) {
                wait(wake_);
                continue;
            }
            const auto time = now();
            if (next < time) SC_REPORT_FATAL("NPU/invariant", "AXI burst issue time passed");
            if (next > time) {
                wait(sc_core::sc_time::from_value(step ? 1 : next - time), wake_);
                continue;
            }
            // Grant only after every other process has run in this cycle, so
            // every request arriving now is queued and the fixed order below
            // (core, then read before write) decides, not process evaluation
            // order.
            if (sc_core::sc_pending_activity_at_current_time()) {
                wait(sc_core::SC_ZERO_TIME);
                continue;
            }
            for (std::size_t index = 0; index < issue_at_.size(); ++index) {
                if (issue_at_[index] != time) continue;
                issue(channel_at(index), time);
                refresh(index);
            }
        }
    }

    void issue(Channel& channel, std::uint64_t time) {
        const auto& axi = config_.arch.memory.axi;
        const auto rate = config_.arch.memory.dram_bytes_per_cycle;
        auto& request = *channel.queue.front();
        const auto& run = request.runs[channel.run];
        const auto end = run.base + run.length;
        const auto burst = next_burst(axi, channel.address, end);
        if (channel.recent.size() == channel.limit) channel.recent.pop_front();
        const auto earliest = std::max(checked_add(time, channel.lead), channel.data_free);
        auto start = earliest;
        auto data_end = checked_add(earliest, burst.beats);
        if (rate) {
            // DRAM carries every beat at full width. A burst waits for a gap
            // holding all its bytes; with DRAM slower than the bus its data
            // phase ends when the last byte leaves DRAM.
            const auto bytes = checked_mul(burst.beats, axi.data_bytes);
            dram_.release(checked_mul(time, rate));
            const auto first = dram_.reserve(checked_mul(earliest, rate), bytes);
            start = ceil_div(first, rate);
            data_end = std::max(checked_add(start, burst.beats),
                                ceil_div(checked_add(first, bytes), rate));
        }
        channel.data_free = data_end;
        const auto completion = checked_add(data_end, channel.tail);
        channel.recent.push_back(completion);
        channel.last_issue = time;
        channel.issued = channel.started = true;
        const auto outstanding = static_cast<std::uint64_t>(std::count_if(
            channel.recent.begin(), channel.recent.end(),
            [time](std::uint64_t done) { return done > time; }));
        channel.stats.peak_outstanding = std::max(channel.stats.peak_outstanding, outstanding);
        ++request.bursts;
        request.beats = checked_add(request.beats, burst.beats);
        request.data_cycles = checked_add(request.data_cycles, data_end - start);
        request.completion = completion;
        channel.address = burst.address + burst.bytes;
        if (channel.address != end) return;
        while (++channel.run < request.runs.size() && request.runs[channel.run].length == 0) {}
        if (channel.run < request.runs.size()) {
            channel.address = request.runs[channel.run].base;
            return;
        }
        request.done.notify(sc_core::sc_time::from_value(completion - time));
        channel.queue.pop_front();
        channel.started = false;
        if (!channel.queue.empty()) front(channel);
    }
};

}  // namespace

MemoryView combine(const std::vector<MemoryView>& ports) {
    if (ports.empty()) throw std::invalid_argument("no memory ports to combine");
    if (ports.size() == 1) return ports.front();
    MemoryView total;
    const auto add = [](ChannelView& sum, const ChannelView& port) {
        sum.active = sum.active || port.active;
        sum.outstanding = checked_add(sum.outstanding, port.outstanding);
        sum.queued = checked_add(sum.queued, port.queued);
        auto& s = sum.statistics;
        const auto& p = port.statistics;
        s.requests = checked_add(s.requests, p.requests);
        s.bursts = checked_add(s.bursts, p.bursts);
        s.beats = checked_add(s.beats, p.beats);
        s.payload_bytes = checked_add(s.payload_bytes, p.payload_bytes);
        s.busy_cycles = checked_add(s.busy_cycles, p.busy_cycles);
        s.latency_cycles = checked_add(s.latency_cycles, p.latency_cycles);
        s.max_latency_cycles = std::max(s.max_latency_cycles, p.max_latency_cycles);
        s.peak_outstanding = std::max(s.peak_outstanding, p.peak_outstanding);
    };
    const auto* shown = &ports.front();
    for (const auto& port : ports) {
        add(total.read, port.read);
        add(total.write, port.write);
        total.busy_cycles = checked_add(total.busy_cycles, port.busy_cycles);
        total.dram_bytes = checked_add(total.dram_bytes, port.dram_bytes);
        if (shown == &ports.front() && !(shown->read.active || shown->write.active) &&
            (port.read.active || port.write.active))
            shown = &port;
    }
    total.tile_id = shown->tile_id;
    total.block_id = shown->block_id;
    total.reading = shown->reading;
    return total;
}

std::unique_ptr<MemoryPort> make_memory(const Config& config, MemoryListener changed) {
    if (config.arch.memory.model == MemoryModel::Axi)
        return std::make_unique<AxiMemory>("memory", config, std::move(changed),
                                           config.arch.cores);
    if (config.arch.cores != 1) throw std::invalid_argument("the serial DMA has one port");
    return std::make_unique<SerialDma>("memory", config, std::move(changed));
}

}  // namespace npu
