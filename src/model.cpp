#include "npu/model.hpp"
#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace npu {
namespace {

std::uint64_t packed_weight_bytes(const Config& config, std::uint64_t k, std::uint64_t cols) {
    return ceil_div(checked_mul(checked_mul(k, cols), config.weights.bits), 8);
}

}  // namespace

Accelerator::Accelerator(sc_core::sc_module_name name, const Config& config,
                         const Problem& problem)
    : sc_module(name), config_(config), problem_(problem),
      encoded_weights_(config.sim.timing_only ? std::vector<EncodedWeightPanel>{} :
                       encode_weights(config, problem)),
      tiles_(make_tiles(config)), record_spans_(!config.sim.trace_path.empty()),
      free_slots_("free_slots", static_cast<int>(config.mapping.buffer_slots)),
      loaded_("loaded_tiles", static_cast<int>(config.mapping.buffer_slots)),
      computed_("computed_tiles", static_cast<int>(config.mapping.buffer_slots)) {
    validate(config);
    for (const auto& panel : encoded_weights_) {
        stats_.encoded_weight_storage_bytes =
            checked_add(stats_.encoded_weight_storage_bytes, panel.buffer.size());
        stats_.dense_weight_storage_bytes = checked_add(stats_.dense_weight_storage_bytes,
            packed_weight_bytes(config, panel.k, panel.cols));
    }
    if (config.weights.encoding == WeightEncoding::Dense) {
        for (const auto& tile : tiles_) {
            if (tile.row != 0) continue;
            for (std::uint64_t block = 0; block < reduction_blocks(config); ++block)
                stats_.dense_weight_storage_bytes = checked_add(stats_.dense_weight_storage_bytes,
                    packed_weight_bytes(config, panel_k(config, block), tile.cols));
        }
        stats_.encoded_weight_storage_bytes = stats_.dense_weight_storage_bytes;
    }
    if (!config.sim.timing_only) {
        const auto size = static_cast<std::size_t>(checked_mul(config.shape.m,
            checked_mul(config.shape.groups, config.shape.n)));
        output_.accumulators.resize(size);
        output_.quantized.resize(size);
    }
    if (record_spans_)
        spans_.reserve(static_cast<std::size_t>(
            checked_mul(checked_mul(3, tiles_.size()), reduction_blocks(config))));
    for (std::uint64_t i = 0; i < config.mapping.buffer_slots; ++i) {
        if (!free_slots_.nb_write(i)) throw std::logic_error("cannot initialize slots");
    }
    SC_THREAD(load);
    SC_THREAD(compute);
    SC_THREAD(store);
    SC_THREAD(dma);
    SC_METHOD(observe);
    sensitive << state_changed_;
}

std::uint64_t Accelerator::now() { return sc_core::sc_time_stamp().value(); }

LiveSnapshot Accelerator::snapshot() const {
    LiveSnapshot view;
    view.cycles = completed_ ? stats_.cycles : now();
    view.completed = completed_;
    view.dma_busy = dma_busy_;
    view.compute_busy = compute_busy_;
    view.dma_read = dma_read_;
    view.inflight_tiles = inflight_;
    view.loaded_tiles = static_cast<std::uint64_t>(loaded_.num_available());
    view.computed_tiles = static_cast<std::uint64_t>(computed_.num_available());
    view.read_queue_depth = static_cast<std::uint64_t>(reads_.size());
    view.write_queue_depth = static_cast<std::uint64_t>(writes_.size());
    view.dma_tile_id = dma_tile_id_;
    view.dma_block_id = dma_block_id_;
    view.compute_tile_id = compute_tile_id_;
    view.compute_block_id = compute_block_id_;
    view.statistics = stats_;
    view.statistics.cycles = view.cycles;
    if (dma_busy_) view.statistics.dma_busy_cycles = checked_add(
        stats_.dma_busy_cycles, std::min(now() - dma_start_, dma_duration_));
    if (compute_busy_) view.statistics.compute_busy_cycles = checked_add(
        stats_.compute_busy_cycles, std::min(now() - compute_start_, compute_duration_));
    return view;
}

void Accelerator::changed() { state_changed_.notify(sc_core::SC_ZERO_TIME); }

void Accelerator::observe() {
    const auto view = snapshot();
    if (view.inflight_tiles > config_.mapping.buffer_slots ||
        view.loaded_tiles > config_.mapping.buffer_slots ||
        view.computed_tiles > config_.mapping.buffer_slots ||
        cached_bytes_ > config_.arch.weight_cache_bytes ||
        ((view.dma_busy || view.compute_busy) && view.inflight_tiles == 0) ||
        (view.completed && (view.inflight_tiles != 0 || view.dma_busy ||
                            view.compute_busy || !reads_.empty() || !writes_.empty()))) {
        SC_REPORT_FATAL("NPU/invariant", "accelerator resource ownership invariant violated");
    }
    cycles_signal_.write(view.cycles);
    done_signal_.write(view.completed);
    dma_busy_signal_.write(view.dma_busy);
    compute_busy_signal_.write(view.compute_busy);
    dma_read_signal_.write(view.dma_read);
    inflight_signal_.write(view.inflight_tiles);
    loaded_signal_.write(view.loaded_tiles);
    computed_signal_.write(view.computed_tiles);
    read_queue_signal_.write(view.read_queue_depth);
    write_queue_signal_.write(view.write_queue_depth);
    read_bytes_signal_.write(view.statistics.read_bytes);
    write_bytes_signal_.write(view.statistics.write_bytes);
    macs_signal_.write(view.statistics.macs);
    dma_tile_signal_.write(view.dma_tile_id);
    dma_block_signal_.write(view.dma_block_id);
    compute_tile_signal_.write(view.compute_tile_id);
    compute_block_signal_.write(view.compute_block_id);
    dma_cycles_signal_.write(view.statistics.dma_busy_cycles);
    compute_cycles_signal_.write(view.statistics.compute_busy_cycles);
    dma_requests_signal_.write(view.statistics.dma_requests);
    dma_bursts_signal_.write(view.statistics.dma_bursts);
    executed_macs_signal_.write(view.statistics.executed_macs);
    skipped_macs_signal_.write(view.statistics.skipped_macs);
    decode_cycles_signal_.write(view.statistics.weight_decode_cycles);
}

void Accelerator::trace_signals(sc_core::sc_trace_file* file) const {
    if (!file) throw std::invalid_argument("trace file must not be null");
    const std::string prefix = std::string(name()) + ".";
#define NPU_TRACE(member, label) sc_core::sc_trace(file, member, prefix + label)
    NPU_TRACE(done_signal_, "done");
    NPU_TRACE(dma_busy_signal_, "dma_busy");
    NPU_TRACE(compute_busy_signal_, "compute_busy");
    NPU_TRACE(dma_read_signal_, "dma_read");
    NPU_TRACE(cycles_signal_, "cycles");
    NPU_TRACE(inflight_signal_, "inflight_tiles");
    NPU_TRACE(loaded_signal_, "loaded_tiles");
    NPU_TRACE(computed_signal_, "computed_tiles");
    NPU_TRACE(read_queue_signal_, "read_queue_depth");
    NPU_TRACE(write_queue_signal_, "write_queue_depth");
    NPU_TRACE(read_bytes_signal_, "read_bytes");
    NPU_TRACE(write_bytes_signal_, "write_bytes");
    NPU_TRACE(macs_signal_, "macs");
    NPU_TRACE(dma_tile_signal_, "dma_tile_id");
    NPU_TRACE(dma_block_signal_, "dma_block_id");
    NPU_TRACE(compute_tile_signal_, "compute_tile_id");
    NPU_TRACE(compute_block_signal_, "compute_block_id");
    NPU_TRACE(dma_cycles_signal_, "dma_busy_cycles");
    NPU_TRACE(compute_cycles_signal_, "compute_busy_cycles");
    NPU_TRACE(dma_requests_signal_, "dma_requests");
    NPU_TRACE(dma_bursts_signal_, "dma_bursts");
    NPU_TRACE(executed_macs_signal_, "executed_macs");
    NPU_TRACE(skipped_macs_signal_, "skipped_macs");
    NPU_TRACE(decode_cycles_signal_, "weight_decode_cycles");
#undef NPU_TRACE
}

void Accelerator::advance(std::uint64_t cycles) {
    if (config_.sim.timing == TimingMode::Cycle) {
        for (std::uint64_t i = 0; i < cycles; ++i)
            wait(sc_core::sc_time::from_value(1));
    } else {
        wait(sc_core::sc_time::from_value(cycles));
    }
}

bool Accelerator::weight_cached(const Tile& tile, std::uint64_t block, std::uint64_t bytes) {
    if (!config_.arch.weight_cache_bytes) return false;
    // One line per (output channel panel, reduction block); channels are
    // numbered across groups.
    const auto channel = checked_add(checked_mul(tile.group, config_.shape.n), tile.col);
    const auto key = checked_add(checked_mul(channel, reduction_blocks(config_)), block);
    const auto found = weight_lines_.find(key);
    if (found != weight_lines_.end()) {
        weight_lru_.splice(weight_lru_.begin(), weight_lru_, found->second);
        ++stats_.weight_cache_hits;
        stats_.weight_bytes_saved = checked_add(stats_.weight_bytes_saved, bytes);
        return true;
    }
    ++stats_.weight_cache_misses;
    if (bytes > config_.arch.weight_cache_bytes) return false;
    while (cached_bytes_ > config_.arch.weight_cache_bytes - bytes) {
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
    std::uint64_t weight_bytes = 0;
    if (read) {
        const auto& format = config_.weights;
        const auto weights = packed_weight_bytes(config_, panel_k(config_, block), tile.cols);
        const auto encoded_bytes = format.encoding == WeightEncoding::Dense ? weights :
            static_cast<std::uint64_t>(
                weight_panel(config_, encoded_weights_, tile, block).buffer.size());
        if (format.encoding != WeightEncoding::Dense) {
            // panel_read_bytes uses a conservative encoded SRAM/traffic bound.
            // Replace that bound with the serialized panel byte count.
            const auto elements = checked_mul(panel_k(config_, block), tile.cols);
            const auto bound = format.encoding == WeightEncoding::Palette ?
                checked_add(checked_add(4, std::min(elements, UINT64_C(1) << format.palette_bits)),
                            ceil_div(checked_mul(elements, format.palette_bits), 8)) :
                checked_add(checked_add(4, ceil_div(ceil_div(elements, format.sparse_block), 8)),
                            weights);
            bytes = checked_add(bytes - bound, encoded_bytes);
        }
        if (weight_cached(tile, block, encoded_bytes)) bytes -= encoded_bytes;
        else weight_bytes = encoded_bytes;
    }
    Request request(tile, bytes, block);
    (read ? reads_ : writes_).push_back(&request);
    changed();
    dma_ready_.notify(sc_core::SC_ZERO_TIME);
    wait(request.done);
    stats_.weight_read_bytes = checked_add(stats_.weight_read_bytes, weight_bytes);
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
        changed();
        for (std::uint64_t block = 0; block < reduction_blocks(config_); ++block) {
            transfer(tile, true, block);
            ++stats_.loaded_panels;
            loaded_.write(checked_add(checked_mul(tile.id, reduction_blocks(config_)), block));
            changed();
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
        const auto* encoded = config_.weights.encoding == WeightEncoding::Dense ? nullptr :
            &weight_panel(config_, encoded_weights_, tile, block);
        const auto duration = encoded ? encoded_panel_compute_cycles(config_, tile, block, *encoded) :
            panel_compute_cycles(config_, tile, block);
        const auto start = now();
        compute_start_ = start;
        compute_duration_ = duration;
        compute_tile_id_ = tile.id;
        compute_block_id_ = block;
        compute_busy_ = true;
        changed();
        // Integer arithmetic is evaluated once at the final reduction panel;
        // the timing model retains the partial-sum allocation across panels.
        if (block + 1 == blocks && !config_.sim.timing_only) {
            if (encoded) compute_tile_encoded(problem_, config_, encoded_weights_, tile, output_);
            else compute_tile(problem_, config_.quant, tile, output_);
        }
        advance(duration);
        stats_.compute_busy_cycles = checked_add(stats_.compute_busy_cycles, duration);
        const auto logical_macs = checked_mul(checked_mul(tile.rows, tile.cols), panel_k(config_, block));
        const auto executed_macs = config_.weights.encoding == WeightEncoding::Sparse ?
            checked_mul(tile.rows, encoded->nonzero_values) : logical_macs;
        stats_.macs = checked_add(stats_.macs, logical_macs);
        stats_.executed_macs = checked_add(stats_.executed_macs, executed_macs);
        stats_.skipped_macs = checked_add(stats_.skipped_macs, logical_macs - executed_macs);
        if (encoded) {
            stats_.weight_decode_cycles = checked_add(stats_.weight_decode_cycles, encoded->decode_cycles);
            stats_.decoded_weight_values = checked_add(stats_.decoded_weight_values, encoded->retained_values);
        }
        ++stats_.computed_panels;
        compute_busy_ = false;
        changed();
        if (record_spans_) spans_.push_back({"compute", tile.id, start, now(), 0, block});
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
        ++stats_.completed_tiles;
        --inflight_;
        free_slots_.write(id);
        changed();
    }
    stats_.cycles = now();
    completed_ = true;
    changed();
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
        dma_start_ = start;
        dma_duration_ = duration;
        dma_tile_id_ = request->tile.id;
        dma_block_id_ = request->block;
        dma_read_ = read;
        dma_busy_ = true;
        changed();
        advance(duration);
        if (record_spans_)
            spans_.push_back({read ? "load" : "store", request->tile.id, start, now(),
                              request->bytes, request->block});
        ++stats_.dma_requests;
        const auto burst = config_.arch.memory.burst_bytes;
        stats_.dma_bursts = checked_add(stats_.dma_bursts,
                                        burst ? ceil_div(request->bytes, burst) : 1);
        stats_.dma_busy_cycles = checked_add(stats_.dma_busy_cycles, duration);
        auto& traffic = read ? stats_.read_bytes : stats_.write_bytes;
        traffic = checked_add(traffic, request->bytes);
        dma_busy_ = false;
        changed();
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
