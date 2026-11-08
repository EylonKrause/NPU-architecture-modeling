#include "npu/model.hpp"
#include <algorithm>
#include <list>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace npu {
namespace {

std::uint64_t packed_weight_bytes(const Config& config, std::uint64_t k, std::uint64_t cols) {
    return ceil_div(checked_mul(checked_mul(k, cols), config.weights.bits), 8);
}

std::uint64_t now() { return sc_core::sc_time_stamp().value(); }

void merge_memory(Statistics& stats, const MemoryView& memory) {
    stats.read_bytes = memory.read.statistics.payload_bytes;
    stats.write_bytes = memory.write.statistics.payload_bytes;
    stats.dma_requests =
        checked_add(memory.read.statistics.requests, memory.write.statistics.requests);
    stats.dma_bursts = checked_add(memory.read.statistics.bursts, memory.write.statistics.bursts);
    stats.dma_busy_cycles = memory.busy_cycles;
}

// Counters add; the in-flight and cache peaks are per core, so the cluster's
// is the largest core's; cycles is the latest core's.
Statistics sum_statistics(const std::vector<Statistics>& cores) {
    Statistics total;
    for (const auto& s : cores) {
        total.cycles = std::max(total.cycles, s.cycles);
        total.peak_inflight_tiles = std::max(total.peak_inflight_tiles, s.peak_inflight_tiles);
        total.peak_cached_weight_bytes =
            std::max(total.peak_cached_weight_bytes, s.peak_cached_weight_bytes);
        total.peak_cached_activation_bytes =
            std::max(total.peak_cached_activation_bytes, s.peak_cached_activation_bytes);
        for (auto [sum, value] :
             {std::pair{&total.macs, s.macs},
              std::pair{&total.read_bytes, s.read_bytes},
              std::pair{&total.write_bytes, s.write_bytes},
              std::pair{&total.compute_busy_cycles, s.compute_busy_cycles},
              std::pair{&total.dma_busy_cycles, s.dma_busy_cycles},
              std::pair{&total.slot_wait_cycles, s.slot_wait_cycles},
              std::pair{&total.panel_wait_cycles, s.panel_wait_cycles},
              std::pair{&total.dma_requests, s.dma_requests},
              std::pair{&total.dma_bursts, s.dma_bursts},
              std::pair{&total.weight_cache_hits, s.weight_cache_hits},
              std::pair{&total.weight_cache_misses, s.weight_cache_misses},
              std::pair{&total.weight_bytes_saved, s.weight_bytes_saved},
              std::pair{&total.loaded_panels, s.loaded_panels},
              std::pair{&total.computed_panels, s.computed_panels},
              std::pair{&total.completed_tiles, s.completed_tiles},
              std::pair{&total.executed_macs, s.executed_macs},
              std::pair{&total.skipped_macs, s.skipped_macs},
              std::pair{&total.weight_decode_cycles, s.weight_decode_cycles},
              std::pair{&total.decoded_weight_values, s.decoded_weight_values},
              std::pair{&total.weight_read_bytes, s.weight_read_bytes},
              std::pair{&total.encoded_weight_storage_bytes, s.encoded_weight_storage_bytes},
              std::pair{&total.dense_weight_storage_bytes, s.dense_weight_storage_bytes},
              std::pair{&total.activation_cache_hits, s.activation_cache_hits},
              std::pair{&total.activation_cache_misses, s.activation_cache_misses},
              std::pair{&total.activation_bytes_saved, s.activation_bytes_saved},
              std::pair{&total.activation_read_bytes, s.activation_read_bytes},
              std::pair{&total.depthwise_macs, s.depthwise_macs}})
            *sum = checked_add(*sum, value);
    }
    return total;
}

PipelineView combine_views(const std::vector<PipelineView>& cores) {
    if (cores.size() == 1) return cores.front();
    PipelineView total;
    std::vector<MemoryView> ports;
    std::vector<Statistics> statistics;
    const PipelineView* computing = &cores.front();
    total.completed = true;
    for (const auto& core : cores) {
        ports.push_back(core.memory);
        statistics.push_back(core.statistics);
        total.cycles = std::max(total.cycles, core.cycles);
        total.completed = total.completed && core.completed;
        total.compute_busy = total.compute_busy || core.compute_busy;
        total.inflight_tiles = checked_add(total.inflight_tiles, core.inflight_tiles);
        total.loaded_tiles = checked_add(total.loaded_tiles, core.loaded_tiles);
        total.computed_tiles = checked_add(total.computed_tiles, core.computed_tiles);
        total.read_queue_depth = checked_add(total.read_queue_depth, core.read_queue_depth);
        total.write_queue_depth = checked_add(total.write_queue_depth, core.write_queue_depth);
        if (!computing->compute_busy && core.compute_busy) computing = &core;
    }
    total.memory = npu::combine(ports);
    total.statistics = sum_statistics(statistics);
    total.dma_busy = total.memory.read.active || total.memory.write.active;
    total.dma_read = total.memory.reading;
    total.dma_write = total.memory.write.active;
    total.dma_tile_id = total.memory.tile_id;
    total.dma_block_id = total.memory.block_id;
    total.compute_tile_id = computing->compute_tile_id;
    total.compute_block_id = computing->compute_block_id;
    return total;
}

// An LRU of panels held in SRAM, keyed by cache line. A panel larger than the
// capacity is never kept; capacity 0 disables the cache and counts nothing.
class PanelCache {
public:
    explicit PanelCache(std::uint64_t capacity) : capacity_(capacity) {}

    // A hit moves the line to the front and saves the panel's bytes; a miss
    // inserts it, evicting least recently used lines until it fits.
    bool lookup(std::uint64_t key, std::uint64_t bytes) {
        if (!capacity_) return false;
        const auto found = lines_.find(key);
        if (found != lines_.end()) {
            lru_.splice(lru_.begin(), lru_, found->second);
            ++hits;
            saved = checked_add(saved, bytes);
            return true;
        }
        ++misses;
        if (bytes > capacity_) return false;
        while (cached_ > capacity_ - bytes) {
            const auto& last = lru_.back();
            cached_ -= last.bytes;
            lines_.erase(last.key);
            lru_.pop_back();
        }
        lru_.push_front({key, bytes});
        lines_[key] = lru_.begin();
        cached_ += bytes;
        peak = std::max(peak, cached_);
        return false;
    }
    bool overfull() const { return cached_ > capacity_; }

    std::uint64_t hits = 0, misses = 0, saved = 0, peak = 0;

private:
    struct Line {
        std::uint64_t key, bytes;
    };
    std::uint64_t capacity_, cached_ = 0;
    std::list<Line> lru_;
    std::unordered_map<std::uint64_t, std::list<Line>::iterator> lines_;
};

}  // namespace

// One core pipeline: load, compute and store threads over the core's own
// slots, panel buffers, weight cache and activation cache, issuing requests on
// its memory port. The cluster's observer drives the core's signals.
class Core final : public sc_core::sc_module {
public:
    Core(sc_core::sc_module_name name, std::uint64_t index, const Config& config,
         const Problem& problem, const std::vector<EncodedWeightPanel>& encoded_weights,
         const std::vector<std::uint64_t>& weight_offsets, std::vector<Tile> tiles, Output& output,
         sc_core::sc_event& state_changed);

    void attach(MemoryPort& memory) { memory_ = &memory; }
    std::uint64_t tiles() const { return static_cast<std::uint64_t>(tiles_.size()); }
    bool completed() const { return completed_; }
    const std::vector<Span>& spans() const { return spans_; }
    Statistics statistics(const MemoryView& memory) const;
    PipelineView view(const MemoryView& memory) const;
    // Marks the core's view as changed for the next observation.
    void mark() { dirty_ = true; }
    bool dirty() const { return dirty_; }
    // Checks the resource invariants and writes the signals.
    void observe(const MemoryView& memory);
    void trace_signals(sc_core::sc_trace_file* file, const std::string& prefix) const;

private:
    const std::uint64_t index_;
    const Config& config_;
    const Problem& problem_;
    const std::vector<EncodedWeightPanel>& encoded_weights_;
    const std::vector<std::uint64_t>& weight_offsets_;
    // This core's tiles in run order; ids are global.
    std::vector<Tile> tiles_;
    Output& output_;
    sc_core::sc_event& changed_;
    bool record_spans_;
    std::uint64_t panel_buffers_;
    Statistics stats_;
    std::vector<Span> spans_;
    // Tokens are indexes into tiles_.
    sc_core::sc_fifo<std::uint64_t> free_slots_;
    sc_core::sc_fifo<std::uint64_t> loaded_;
    sc_core::sc_fifo<std::uint64_t> computed_;
    sc_core::sc_event panel_consumed_;
    PanelCache weight_cache_, activation_cache_;
    std::uint64_t inflight_ = 0;
    bool completed_ = false, compute_busy_ = false, dirty_ = true;
    std::uint64_t compute_start_ = 0, compute_duration_ = 0;
    std::uint64_t compute_tile_id_ = 0, compute_block_id_ = 0;
    MemoryPort* memory_ = nullptr;
    MemRequest load_request_, store_request_;
    // Written only by the cluster's observer method, so simultaneous resource
    // updates in the scheduling threads are safe.
    sc_core::sc_signal<bool> done_signal_{"done"};
    sc_core::sc_signal<bool> dma_busy_signal_{"dma_busy"};
    sc_core::sc_signal<bool> compute_busy_signal_{"compute_busy"};
    sc_core::sc_signal<bool> dma_read_signal_{"dma_read"};
    sc_core::sc_signal<bool> dma_write_signal_{"dma_write"};
    using CounterSignal = sc_core::sc_signal<sc_dt::sc_uint<64>>;
    CounterSignal cycles_signal_{"cycles"};
    CounterSignal inflight_signal_{"inflight_tiles"};
    CounterSignal loaded_signal_{"loaded_tiles_count"};
    CounterSignal computed_signal_{"computed_tiles_count"};
    CounterSignal read_queue_signal_{"read_queue_depth"};
    CounterSignal write_queue_signal_{"write_queue_depth"};
    CounterSignal read_bytes_signal_{"read_bytes"};
    CounterSignal write_bytes_signal_{"write_bytes"};
    CounterSignal macs_signal_{"macs"};
    CounterSignal dma_tile_signal_{"dma_tile_id"};
    CounterSignal dma_block_signal_{"dma_block_id"};
    CounterSignal compute_tile_signal_{"compute_tile_id"};
    CounterSignal compute_block_signal_{"compute_block_id"};
    CounterSignal dma_cycles_signal_{"dma_busy_cycles"};
    CounterSignal compute_cycles_signal_{"compute_busy_cycles"};
    CounterSignal dma_requests_signal_{"dma_requests"};
    CounterSignal dma_bursts_signal_{"dma_bursts"};
    CounterSignal executed_macs_signal_{"executed_macs"};
    CounterSignal skipped_macs_signal_{"skipped_macs"};
    CounterSignal decode_cycles_signal_{"weight_decode_cycles"};

    void changed();
    void load();
    void compute();
    void store();
    void transfer(const Tile& tile, bool read, std::uint64_t block = 0);
    void advance(std::uint64_t cycles);
};

Core::Core(sc_core::sc_module_name name, std::uint64_t index, const Config& config,
           const Problem& problem, const std::vector<EncodedWeightPanel>& encoded_weights,
           const std::vector<std::uint64_t>& weight_offsets, std::vector<Tile> tiles,
           Output& output, sc_core::sc_event& state_changed)
    : sc_module(name),
      index_(index),
      config_(config),
      problem_(problem),
      encoded_weights_(encoded_weights),
      weight_offsets_(weight_offsets),
      tiles_(std::move(tiles)),
      output_(output),
      changed_(state_changed),
      record_spans_(!config.sim.trace_path.empty()),
      panel_buffers_(slot_panel_buffers(config)),
      free_slots_("free_slots", static_cast<int>(config.mapping.buffer_slots)),
      // Every loaded panel waiting for compute occupies one panel buffer.
      loaded_("loaded_tiles", static_cast<int>(config.mapping.buffer_slots * panel_buffers_)),
      computed_("computed_tiles", static_cast<int>(config.mapping.buffer_slots)),
      weight_cache_(config.arch.weight_cache_bytes),
      activation_cache_(config.arch.activation_cache_bytes) {
    if (record_spans_)
        spans_.reserve(static_cast<std::size_t>(
            checked_mul(checked_mul(3, tiles_.size()), reduction_blocks(config))));
    for (std::uint64_t i = 0; i < config.mapping.buffer_slots; ++i) {
        if (!free_slots_.nb_write(i)) throw std::logic_error("cannot initialize slots");
    }
    SC_THREAD(load);
    SC_THREAD(compute);
    SC_THREAD(store);
    load_request_.direction = MemDirection::Read;
    store_request_.direction = MemDirection::Write;
    load_request_.requester = store_request_.requester = index_;
}

Statistics Core::statistics(const MemoryView& memory) const {
    auto stats = stats_;
    merge_memory(stats, memory);
    stats.weight_cache_hits = weight_cache_.hits;
    stats.weight_cache_misses = weight_cache_.misses;
    stats.weight_bytes_saved = weight_cache_.saved;
    stats.peak_cached_weight_bytes = weight_cache_.peak;
    stats.activation_cache_hits = activation_cache_.hits;
    stats.activation_cache_misses = activation_cache_.misses;
    stats.activation_bytes_saved = activation_cache_.saved;
    stats.peak_cached_activation_bytes = activation_cache_.peak;
    return stats;
}

PipelineView Core::view(const MemoryView& memory) const {
    PipelineView view;
    view.memory = memory;
    view.cycles = completed_ ? stats_.cycles : now();
    view.completed = completed_;
    view.dma_busy = memory.read.active || memory.write.active;
    view.compute_busy = compute_busy_;
    view.dma_read = memory.reading;
    view.dma_write = memory.write.active;
    view.inflight_tiles = inflight_;
    view.loaded_tiles = static_cast<std::uint64_t>(loaded_.num_available());
    view.computed_tiles = static_cast<std::uint64_t>(computed_.num_available());
    view.read_queue_depth = memory.read.queued;
    view.write_queue_depth = memory.write.queued;
    view.dma_tile_id = memory.tile_id;
    view.dma_block_id = memory.block_id;
    view.compute_tile_id = compute_tile_id_;
    view.compute_block_id = compute_block_id_;
    view.statistics = statistics(memory);
    view.statistics.cycles = view.cycles;
    if (compute_busy_)
        view.statistics.compute_busy_cycles = checked_add(
            stats_.compute_busy_cycles, std::min(now() - compute_start_, compute_duration_));
    return view;
}

void Core::changed() {
    dirty_ = true;
    changed_.notify(sc_core::SC_ZERO_TIME);
}

void Core::observe(const MemoryView& memory) {
    dirty_ = false;
    const auto view = this->view(memory);
    if (view.inflight_tiles > config_.mapping.buffer_slots ||
        view.loaded_tiles > config_.mapping.buffer_slots * panel_buffers_ ||
        view.computed_tiles > config_.mapping.buffer_slots || weight_cache_.overfull() ||
        activation_cache_.overfull() ||
        ((view.dma_busy || view.compute_busy) && view.inflight_tiles == 0) ||
        (view.completed && (view.inflight_tiles != 0 || view.dma_busy || view.compute_busy ||
                            view.read_queue_depth != 0 || view.write_queue_depth != 0))) {
        SC_REPORT_FATAL("NPU/invariant", "accelerator resource ownership invariant violated");
    }
    cycles_signal_.write(view.cycles);
    done_signal_.write(view.completed);
    dma_busy_signal_.write(view.dma_busy);
    compute_busy_signal_.write(view.compute_busy);
    dma_read_signal_.write(view.dma_read);
    dma_write_signal_.write(view.dma_write);
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

void Core::trace_signals(sc_core::sc_trace_file* file, const std::string& prefix) const {
#define NPU_TRACE(member, label) sc_core::sc_trace(file, member, prefix + label)
    NPU_TRACE(done_signal_, "done");
    NPU_TRACE(dma_busy_signal_, "dma_busy");
    NPU_TRACE(compute_busy_signal_, "compute_busy");
    NPU_TRACE(dma_read_signal_, "dma_read");
    NPU_TRACE(dma_write_signal_, "dma_write");
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

void Core::advance(std::uint64_t cycles) {
    if (config_.sim.timing == TimingMode::Cycle) {
        for (std::uint64_t i = 0; i < cycles; ++i) wait(sc_core::sc_time::from_value(1));
    } else {
        wait(sc_core::sc_time::from_value(cycles));
    }
}

void Core::transfer(const Tile& tile, bool read, std::uint64_t block) {
    auto& request = read ? load_request_ : store_request_;
    std::uint64_t weight_bytes = 0, activation_bytes = 0;
    if (read) {
        std::uint64_t encoded_bytes = 0, offset = 0;
        if (config_.weights.encoding == WeightEncoding::Dense) {
            encoded_bytes = packed_weight_bytes(config_, panel_k(config_, block), tile.cols);
            offset = dense_weight_offset(config_, tile, block);
        } else {
            const auto& panel = weight_panel(config_, encoded_weights_, tile, block);
            encoded_bytes = static_cast<std::uint64_t>(panel.buffer.size());
            offset = weight_offsets_.at(static_cast<std::size_t>(&panel - encoded_weights_.data()));
        }
        // One weight line per (output channel panel, reduction block);
        // channels are numbered across groups.
        const auto channel = checked_add(checked_mul(tile.group, config_.shape.n), tile.col);
        const auto weight_key = checked_add(checked_mul(channel, reduction_blocks(config_)), block);
        if (!weight_cache_.lookup(weight_key, encoded_bytes)) weight_bytes = encoded_bytes;
        const auto activations = activation_panel_bytes(config_, tile, block);
        const bool reused =
            config_.arch.activation_cache_bytes &&
            activation_cache_.lookup(activation_cache_key(config_, tile, block), activations);
        if (!reused) activation_bytes = activations;
        panel_read_runs(config_, tile, block, offset, weight_bytes, request.runs, !reused);
    } else {
        tile_store_runs(config_, tile, request.runs);
    }
    request.tile_id = tile.id;
    request.block_id = block;
    memory_->transport(request);
    if (record_spans_)
        spans_.push_back({read ? "load" : "store", tile.id, request.start, request.completion,
                          request.bytes, block, index_});
    stats_.weight_read_bytes = checked_add(stats_.weight_read_bytes, weight_bytes);
    stats_.activation_read_bytes = checked_add(stats_.activation_read_bytes, activation_bytes);
}

void Core::load() {
    const auto blocks = reduction_blocks(config_);
    for (std::size_t local = 0; local < tiles_.size(); ++local) {
        const auto& tile = tiles_[local];
        const auto start = now();
        std::uint64_t token = 0;
        free_slots_.read(token);
        (void)token;
        stats_.slot_wait_cycles = checked_add(stats_.slot_wait_cycles, now() - start);
        ++inflight_;
        stats_.peak_inflight_tiles = std::max(stats_.peak_inflight_tiles, inflight_);
        changed();
        const auto first_panel = checked_mul(local, blocks);
        for (std::uint64_t block = 0; block < blocks; ++block) {
            // Panel b reuses the buffer of panel b - panel_buffers_. Compute
            // takes panels in load order, so that buffer is free once the
            // computed-panel count passes it.
            if (block >= panel_buffers_) {
                const auto released = checked_add(first_panel, block - panel_buffers_ + 1);
                const auto wait_start = now();
                while (stats_.computed_panels < released) wait(panel_consumed_);
                stats_.panel_wait_cycles =
                    checked_add(stats_.panel_wait_cycles, now() - wait_start);
            }
            transfer(tile, true, block);
            ++stats_.loaded_panels;
            loaded_.write(checked_add(first_panel, block));
            changed();
        }
    }
}

void Core::compute() {
    const auto blocks = reduction_blocks(config_);
    const bool depthwise = channel_wise(config_);
    for (std::uint64_t count = 0; count < checked_mul(tiles_.size(), blocks); ++count) {
        std::uint64_t job = 0;
        loaded_.read(job);
        const auto block = job % blocks;
        const auto& tile = tiles_.at(static_cast<std::size_t>(job / blocks));
        const auto* encoded = config_.weights.encoding == WeightEncoding::Dense ?
                                  nullptr :
                                  &weight_panel(config_, encoded_weights_, tile, block);
        const auto duration = encoded ?
                                  encoded_panel_compute_cycles(config_, tile, block, *encoded) :
                                  panel_compute_cycles(config_, tile, block);
        const auto start = now();
        compute_start_ = start;
        compute_duration_ = duration;
        compute_tile_id_ = tile.id;
        compute_block_id_ = block;
        compute_busy_ = true;
        changed();
        // The tile's integer result is computed at its final reduction panel;
        // earlier panels take time only, with the partial sums held in the
        // slot. Cores write disjoint tiles of the shared output.
        if (block + 1 == blocks && !config_.sim.timing_only) {
            if (encoded) compute_tile_encoded(problem_, config_, encoded_weights_, tile, output_);
            else if (depthwise) compute_channel_tile(problem_, config_.quant, tile, output_);
            else compute_tile(problem_, config_.quant, tile, output_);
        }
        advance(duration);
        stats_.compute_busy_cycles = checked_add(stats_.compute_busy_cycles, duration);
        const auto logical_macs =
            checked_mul(checked_mul(tile.rows, tile.cols), panel_k(config_, block));
        const auto executed_macs = config_.weights.encoding == WeightEncoding::Sparse ?
                                       checked_mul(tile.rows, encoded->nonzero_values) :
                                       logical_macs;
        stats_.macs = checked_add(stats_.macs, logical_macs);
        stats_.executed_macs = checked_add(stats_.executed_macs, executed_macs);
        stats_.skipped_macs = checked_add(stats_.skipped_macs, logical_macs - executed_macs);
        if (depthwise) stats_.depthwise_macs = checked_add(stats_.depthwise_macs, logical_macs);
        if (encoded) {
            stats_.weight_decode_cycles =
                checked_add(stats_.weight_decode_cycles, encoded->decode_cycles);
            stats_.decoded_weight_values =
                checked_add(stats_.decoded_weight_values, encoded->retained_values);
        }
        ++stats_.computed_panels;
        compute_busy_ = false;
        changed();
        if (record_spans_) spans_.push_back({"compute", tile.id, start, now(), 0, block, index_});
        // A final panel's buffer is released with the slot after the store.
        if (block + 1 == blocks) computed_.write(job / blocks);
        else panel_consumed_.notify(sc_core::SC_ZERO_TIME);
    }
}

void Core::store() {
    for (std::size_t count = 0; count < tiles_.size(); ++count) {
        std::uint64_t local = 0;
        computed_.read(local);
        const auto& tile = tiles_.at(static_cast<std::size_t>(local));
        transfer(tile, false, reduction_blocks(config_) - 1);
        ++stats_.completed_tiles;
        --inflight_;
        free_slots_.write(local);
        changed();
    }
    stats_.cycles = now();
    completed_ = true;
    changed();
    // There is no free-running clock and no sc_stop: sc_start returns once every
    // process has finished or blocks with no pending event.
}

Accelerator::Accelerator(sc_core::sc_module_name name, const Config& config, const Problem& problem)
    : sc_module(name),
      encoded_weights_(config.sim.timing_only ? std::vector<EncodedWeightPanel>{} :
                                                encode_weights(config, problem)) {
    validate(config);
    weight_offsets_.reserve(encoded_weights_.size());
    for (const auto& panel : encoded_weights_) {
        // Packed back to back in load order, as encode_weights lists them.
        weight_offsets_.push_back(encoded_storage_bytes_);
        encoded_storage_bytes_ = checked_add(encoded_storage_bytes_, panel.buffer.size());
        dense_storage_bytes_ =
            checked_add(dense_storage_bytes_, packed_weight_bytes(config, panel.k, panel.cols));
    }
    const auto column_panels = ceil_div(grid_columns(config), config.mapping.tile_n);
    const bool dense = config.weights.encoding == WeightEncoding::Dense;
    // Both splits are predicted from the panels the loader reads and the
    // array computes.
    const auto column_costs =
        dense ? dense_column_costs(config) : encoded_column_costs(config, encoded_weights_);
    if (dense) {
        // Every group of the grid stores the same column panels.
        for (std::uint64_t col = 0; col < column_panels; ++col) {
            const auto column = column_costs(0, col);
            dense_storage_bytes_ =
                checked_add(dense_storage_bytes_,
                            checked_mul(grid_groups(config),
                                        checked_add(column.cacheable, column.uncacheable)));
        }
        encoded_storage_bytes_ = dense_storage_bytes_;
    }
    if (!config.sim.timing_only) {
        const auto size = static_cast<std::size_t>(
            checked_mul(config.shape.m, checked_mul(config.shape.groups, config.shape.n)));
        output_.accumulators.resize(size);
        output_.quantized.resize(size);
    }

    const auto by_rows = cluster_plan(config, Partition::M);
    const auto by_columns = cluster_plan(config, Partition::N);
    estimates_[1] = estimate_cluster(config, by_columns, column_costs);
    // With whole groups per core both splits are the same.
    estimates_[0] = by_rows.split == ClusterSplit::Groups ?
                        estimates_[1] :
                        estimate_cluster(config, by_rows, column_costs);
    auto partition = config.mapping.partition;
    if (partition == Partition::Auto) partition = auto_partition(estimates_[0], estimates_[1]);
    plan_ = partition == Partition::M ? by_rows : by_columns;

    auto assigned = make_core_tiles(config, plan_);
    // Process creation order: every core's load, compute and store threads,
    // then the memory module's, then the observer.
    for (std::uint64_t core = 0; core < plan_.cores; ++core)
        cores_.push_back(std::make_unique<Core>(("core" + std::to_string(core)).c_str(), core,
                                                config, problem, encoded_weights_, weight_offsets_,
                                                std::move(assigned[static_cast<std::size_t>(core)]),
                                                output_, state_changed_));
    memory_ = make_memory(config, [this](std::uint64_t requester) {
        cores_.at(static_cast<std::size_t>(requester))->mark();
        state_changed_.notify(sc_core::SC_ZERO_TIME);
    });
    for (auto& core : cores_) core->attach(*memory_);
    SC_METHOD(observe);
    sensitive << state_changed_;
}

Accelerator::~Accelerator() = default;

const ClusterEstimate& Accelerator::estimate(Partition partition) const {
    if (partition == Partition::Auto)
        throw std::invalid_argument("estimates exist for partitions m and n");
    return estimates_[partition == Partition::M ? 0 : 1];
}

std::uint64_t Accelerator::core_tiles(std::uint64_t core) const {
    return cores_.at(static_cast<std::size_t>(core))->tiles();
}

Statistics Accelerator::statistics(std::uint64_t core) const {
    return cores_.at(static_cast<std::size_t>(core))->statistics(memory_->view(core));
}

Statistics Accelerator::statistics() const {
    std::vector<Statistics> per_core;
    for (std::uint64_t core = 0; core < cores(); ++core) per_core.push_back(statistics(core));
    auto total = sum_statistics(per_core);
    total.encoded_weight_storage_bytes = encoded_storage_bytes_;
    total.dense_weight_storage_bytes = dense_storage_bytes_;
    return total;
}

MemoryView Accelerator::memory(std::uint64_t core) const { return memory_->view(core); }

MemoryView Accelerator::memory() const {
    std::vector<MemoryView> ports;
    for (std::uint64_t core = 0; core < cores(); ++core) ports.push_back(memory_->view(core));
    return combine(ports);
}

std::vector<Span> Accelerator::spans() const {
    std::vector<Span> all;
    for (const auto& core : cores_)
        all.insert(all.end(), core->spans().begin(), core->spans().end());
    return all;
}

bool Accelerator::completed() const {
    return std::all_of(cores_.begin(), cores_.end(),
                       [](const auto& core) { return core->completed(); });
}

LiveSnapshot Accelerator::snapshot() const {
    LiveSnapshot snapshot;
    for (std::uint64_t core = 0; core < cores(); ++core)
        snapshot.cores.push_back(cores_[static_cast<std::size_t>(core)]->view(memory_->view(core)));
    static_cast<PipelineView&>(snapshot) = combine_views(snapshot.cores);
    snapshot.statistics.encoded_weight_storage_bytes = encoded_storage_bytes_;
    snapshot.statistics.dense_weight_storage_bytes = dense_storage_bytes_;
    return snapshot;
}

void Accelerator::observe() {
    // Only cores whose threads or memory port changed since the last run
    // rebuild their view and signals.
    for (std::uint64_t core = 0; core < cores(); ++core) {
        auto& each = *cores_[static_cast<std::size_t>(core)];
        if (each.dirty()) each.observe(memory_->view(core));
    }
}

void Accelerator::trace_signals(sc_core::sc_trace_file* file) const {
    if (!file) throw std::invalid_argument("trace file must not be null");
    for (const auto& core : cores_)
        core->trace_signals(file, std::string(cores_.size() == 1 ? name() : core->name()) + ".");
}

void write_trace(std::ostream& stream, const std::vector<Span>& spans) {
    std::vector<const Span*> ordered;
    ordered.reserve(spans.size());
    for (const auto& span : spans) ordered.push_back(&span);
    std::sort(ordered.begin(), ordered.end(), [](const Span* a, const Span* b) {
        return std::tie(a->start, a->stage, a->tile_id, a->block_id) <
               std::tie(b->start, b->stage, b->tile_id, b->block_id);
    });
    stream << "stage,tile_id,start_cycle,end_cycle,bytes,block_id,core\n";
    for (const auto* s : ordered)
        stream << s->stage << ',' << s->tile_id << ',' << s->start << ',' << s->end << ','
               << s->bytes << ',' << s->block_id << ',' << s->core << '\n';
    if (!stream) throw std::runtime_error("cannot write trace");
}

}  // namespace npu
