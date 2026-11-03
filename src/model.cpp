#include "npu/model.hpp"
#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace npu {

Accelerator::Accelerator(sc_core::sc_module_name name, const Config& config,
                         const Problem& problem)
    : sc_module(name), config_(config), problem_(problem), tiles_(make_tiles(config)),
      free_slots_("free_slots", static_cast<int>(config.buffer_slots)),
      loaded_("loaded_tiles", static_cast<int>(config.buffer_slots)),
      computed_("computed_tiles", static_cast<int>(config.buffer_slots)) {
    validate(config);
    const auto size = static_cast<std::size_t>(checked_mul(config.m, config.n));
    output_.accumulators.resize(size);
    output_.quantized.resize(size);
    trace_.reserve(static_cast<std::size_t>(checked_mul(3, tiles_.size())));
    for (std::uint64_t i = 0; i < config.buffer_slots; ++i) {
        if (!free_slots_.nb_write(i)) throw std::logic_error("cannot initialize slots");
    }
    SC_THREAD(load);
    SC_THREAD(compute);
    SC_THREAD(store);
    SC_THREAD(dma);
}

std::uint64_t Accelerator::now() { return sc_core::sc_time_stamp().value(); }

void Accelerator::advance(std::uint64_t cycles) {
    if (config_.timing == "cycle") {
        for (std::uint64_t i = 0; i < cycles; ++i)
            wait(sc_core::sc_time::from_value(1));
    } else {
        wait(sc_core::sc_time::from_value(cycles));
    }
}

void Accelerator::transfer(const Tile& tile, bool read) {
    Request request(tile, read ? tile_read_bytes(config_, tile) : tile_write_bytes(tile));
    (read ? reads_ : writes_).push_back(&request);
    dma_ready_.notify(sc_core::SC_ZERO_TIME);
    wait(request.done);
}

void Accelerator::load() {
    for (const auto& tile : tiles_) {
        const auto start = now();
        std::uint64_t token = 0;
        free_slots_.read(token);
        (void)token;
        stats_.slot_wait_cycles = checked_add(stats_.slot_wait_cycles, now() - start);
        ++inflight_;
        stats_.peak_inflight_tiles = std::max(stats_.peak_inflight_tiles, inflight_);
        transfer(tile, true);
        loaded_.write(tile.id);
    }
}

void Accelerator::compute() {
    for (std::size_t count = 0; count < tiles_.size(); ++count) {
        std::uint64_t id = 0;
        loaded_.read(id);
        const auto& tile = tiles_.at(static_cast<std::size_t>(id));
        const auto duration = tile_compute_cycles(config_, tile);
        const auto start = now();
        compute_tile(problem_, config_.quant, tile, output_);
        advance(duration);
        stats_.compute_busy_cycles = checked_add(stats_.compute_busy_cycles, duration);
        stats_.macs = checked_add(stats_.macs, checked_mul(checked_mul(tile.rows, tile.cols), config_.k));
        trace_.push_back({"compute", tile.id, start, now(), 0});
        computed_.write(tile.id);
    }
}

void Accelerator::store() {
    for (std::size_t count = 0; count < tiles_.size(); ++count) {
        std::uint64_t id = 0;
        computed_.read(id);
        const auto& tile = tiles_.at(static_cast<std::size_t>(id));
        transfer(tile, false);
        --inflight_;
        free_slots_.write(id);
    }
    stats_.cycles = now();
    completed_ = true;
    // No free-running clock and no sc_stop: sc_start returns when every process
    // finishes or blocks with no pending event. Final JSON stays on stdout.
}

void Accelerator::dma() {
    while (true) {
        if (reads_.empty() && writes_.empty()) wait(dma_ready_);
        // Coalesce simultaneous requests. Delta cycles cost no architecture time.
        wait(sc_core::SC_ZERO_TIME);
        wait(sc_core::SC_ZERO_TIME);
        if (reads_.empty() && writes_.empty()) continue;
        const bool read = !reads_.empty() && (writes_.empty() || prefer_read_);
        auto& queue = read ? reads_ : writes_;
        Request* request = queue.front();
        queue.pop_front();
        const auto start = now();
        const auto duration = checked_add(config_.memory_latency,
            ceil_div(request->bytes, config_.memory_bytes_per_cycle));
        advance(duration);
        trace_.push_back({read ? "load" : "store", request->tile.id, start, now(), request->bytes});
        stats_.dma_busy_cycles = checked_add(stats_.dma_busy_cycles, duration);
        auto& traffic = read ? stats_.read_bytes : stats_.write_bytes;
        traffic = checked_add(traffic, request->bytes);
        prefer_read_ = !read;
        request->done.notify(sc_core::SC_ZERO_TIME);
        // Do not inspect request again: requester owns its lifetime.
    }
}

void write_trace(std::ostream& stream, const std::vector<Span>& spans) {
    std::vector<const Span*> ordered;
    ordered.reserve(spans.size());
    for (const auto& span : spans) ordered.push_back(&span);
    std::sort(ordered.begin(), ordered.end(), [](const Span* a, const Span* b) {
        return std::tie(a->start, a->stage, a->tile_id) < std::tie(b->start, b->stage, b->tile_id);
    });
    stream << "stage,tile_id,start_cycle,end_cycle,bytes\n";
    for (const auto* s : ordered)
        stream << s->stage << ',' << s->tile_id << ',' << s->start << ',' << s->end << ',' << s->bytes << '\n';
    if (!stream) throw std::runtime_error("cannot write trace");
}

}  // namespace npu
