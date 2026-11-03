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
    trace_.reserve(static_cast<std::size_t>(checked_mul(checked_mul(3, tiles_.size()), reduction_blocks(config))));
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

bool Accelerator::weight_cached(const Tile& tile, std::uint64_t block, std::uint64_t bytes) {
    if (!config_.weight_cache_bytes) return false;
    const auto key = checked_add(checked_mul(tile.col, reduction_blocks(config_)), block);
    const auto found = weight_lines_.find(key);
    if (found != weight_lines_.end()) {
        weight_lru_.splice(weight_lru_.begin(), weight_lru_, found->second);
        ++stats_.weight_cache_hits;
        stats_.weight_bytes_saved = checked_add(stats_.weight_bytes_saved, bytes);
        return true;
    }
    ++stats_.weight_cache_misses;
    if (bytes > config_.weight_cache_bytes) return false;
    while (cached_bytes_ > config_.weight_cache_bytes - bytes) {
        const auto& last = weight_lru_.back();
        cached_bytes_ -= last.bytes;
        weight_lines_.erase(last.key);
        weight_lru_.pop_back();
    }
    weight_lru_.push_front({key, bytes});
    weight_lines_[key] = weight_lru_.begin();
    cached_bytes_ += bytes;
    stats_.peak_cached_weight_bytes = std::max(stats_.peak_cached_weight_bytes, cached_bytes_);
    return false;
}

void Accelerator::transfer(const Tile& tile, bool read, std::uint64_t block) {
    auto bytes = read ? panel_read_bytes(config_, tile, block) : tile_write_bytes(tile);
    if (read) {
        const auto weights = ceil_div(checked_mul(checked_mul(panel_k(config_, block), tile.cols), config_.weight_bits), 8);
        if (weight_cached(tile, block, weights)) bytes -= weights;
    }
    Request request(tile, bytes, block);
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
        for (std::uint64_t block = 0; block < reduction_blocks(config_); ++block) {
            transfer(tile, true, block);
            loaded_.write(checked_add(checked_mul(tile.id, reduction_blocks(config_)), block));
            // A slot holds one reduction panel plus resident partial sums.
            // Wait until compute consumed it before reusing that panel buffer.
            if (block + 1 != reduction_blocks(config_)) wait(panel_consumed_);
        }
    }
}

void Accelerator::compute() {
    const auto blocks = reduction_blocks(config_);
    for (std::uint64_t count = 0; count < checked_mul(tiles_.size(), blocks); ++count) {
        std::uint64_t job = 0;
        loaded_.read(job);
        const auto id = job / blocks;
        const auto block = job % blocks;
        const auto& tile = tiles_.at(static_cast<std::size_t>(id));
        const auto duration = panel_compute_cycles(config_, tile, block);
        const auto start = now();
        // Integer arithmetic is evaluated once at the final reduction panel;
        // the timing model retains the partial-sum allocation across panels.
        if (block + 1 == blocks) compute_tile(problem_, config_.quant, tile, output_);
        advance(duration);
        stats_.compute_busy_cycles = checked_add(stats_.compute_busy_cycles, duration);
        stats_.macs = checked_add(stats_.macs, checked_mul(checked_mul(tile.rows, tile.cols), panel_k(config_, block)));
        trace_.push_back({"compute", tile.id, start, now(), 0, block});
        if (block + 1 == blocks) computed_.write(tile.id);
        else panel_consumed_.notify(sc_core::SC_ZERO_TIME);
    }
}

void Accelerator::store() {
    for (std::size_t count = 0; count < tiles_.size(); ++count) {
        std::uint64_t id = 0;
        computed_.read(id);
        const auto& tile = tiles_.at(static_cast<std::size_t>(id));
        transfer(tile, false, reduction_blocks(config_) - 1);
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
        const auto duration = dma_transfer_cycles(config_, request->bytes);
        advance(duration);
        trace_.push_back({read ? "load" : "store", request->tile.id, start, now(), request->bytes, request->block});
        ++stats_.dma_requests;
        stats_.dma_bursts = checked_add(stats_.dma_bursts,
            config_.dma_burst_bytes ? ceil_div(request->bytes, config_.dma_burst_bytes) : 1);
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
        return std::tie(a->start, a->stage, a->tile_id, a->block_id) < std::tie(b->start, b->stage, b->tile_id, b->block_id);
    });
    stream << "stage,tile_id,start_cycle,end_cycle,bytes,block_id\n";
    for (const auto* s : ordered)
        stream << s->stage << ',' << s->tile_id << ',' << s->start << ',' << s->end << ',' << s->bytes << ',' << s->block_id << '\n';
    if (!stream) throw std::runtime_error("cannot write trace");
}

}  // namespace npu
