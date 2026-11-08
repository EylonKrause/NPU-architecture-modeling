#pragma once

#include "npu/core.hpp"
#include "npu/memory.hpp"
#include <array>
#include <memory>
#include <ostream>
#include <string>
#include <systemc>
#include <vector>

namespace npu {

struct Span {
    std::string stage;
    std::uint64_t tile_id = 0, start = 0, end = 0, bytes = 0;
    std::uint64_t block_id = 0, core = 0;
};

struct Statistics {
    std::uint64_t cycles = 0, macs = 0, read_bytes = 0, write_bytes = 0;
    std::uint64_t compute_busy_cycles = 0, dma_busy_cycles = 0;
    // slot_wait_cycles: the loader waiting for a free slot; panel_wait_cycles:
    // waiting for a panel buffer of the slot it holds.
    std::uint64_t peak_inflight_tiles = 0, slot_wait_cycles = 0, panel_wait_cycles = 0;
    std::uint64_t dma_requests = 0, dma_bursts = 0;
    std::uint64_t weight_cache_hits = 0, weight_cache_misses = 0;
    std::uint64_t weight_bytes_saved = 0, peak_cached_weight_bytes = 0;
    std::uint64_t loaded_panels = 0, computed_panels = 0, completed_tiles = 0;
    std::uint64_t executed_macs = 0, skipped_macs = 0, weight_decode_cycles = 0;
    std::uint64_t decoded_weight_values = 0, weight_read_bytes = 0;
    std::uint64_t encoded_weight_storage_bytes = 0, dense_weight_storage_bytes = 0;
    std::uint64_t activation_cache_hits = 0, activation_cache_misses = 0;
    std::uint64_t activation_bytes_saved = 0, peak_cached_activation_bytes = 0;
    std::uint64_t activation_read_bytes = 0;
    // MACs the depthwise datapath ran.
    std::uint64_t depthwise_macs = 0;
};

// A point-in-time view of one core's pipeline, or of the cluster. Resource
// identifiers are meaningful only while their busy flag is set. Traffic and MAC
// counters commit at operation completion; busy-cycle counters include the
// elapsed portion of an operation in flight. dma_busy is set while either
// memory channel has a request in flight and dma_write while the write channel
// has; dma_read gives the direction of dma_tile_id, the read when both are busy.
// The serial model runs one request at a time.
struct PipelineView {
    std::uint64_t cycles = 0;
    bool completed = false, dma_busy = false, compute_busy = false, dma_read = false;
    bool dma_write = false;
    std::uint64_t inflight_tiles = 0, loaded_tiles = 0, computed_tiles = 0;
    std::uint64_t read_queue_depth = 0, write_queue_depth = 0;
    std::uint64_t dma_tile_id = 0, dma_block_id = 0;
    std::uint64_t compute_tile_id = 0, compute_block_id = 0;
    Statistics statistics;
    MemoryView memory;
};

// The cluster, then each core. Cluster counters add the cores' and its memory
// combines their ports; cycles is the makespan once every core completes;
// peak_inflight_tiles and the peak cached weight and activation bytes are the
// largest core's, as slots and caches are per core; resource identifiers come
// from the lowest-numbered busy core, else core 0. Tile ids are global.
struct LiveSnapshot : PipelineView {
    std::vector<PipelineView> cores;
};

class Core;

// A cluster of arch.cores core pipelines splitting the output tiles by
// plan() and sharing one memory module. Each core owns its slots, panel
// buffers, weight cache and memory request objects; the cores write disjoint
// tiles of one output tensor. All state is owned by this module or immutable
// references whose owners outlive sc_start(), so a simulation stopped while a
// transfer is in flight releases it with the module.
class Accelerator final : public sc_core::sc_module {
public:
    Accelerator(sc_core::sc_module_name name, const Config& config, const Problem& problem);
    ~Accelerator() override;
    const Output& output() const { return output_; }
    const ClusterPlan& plan() const { return plan_; }
    // Prediction of the plan of partition M or N.
    const ClusterEstimate& estimate(Partition partition) const;
    std::uint64_t cores() const { return static_cast<std::uint64_t>(cores_.size()); }
    std::uint64_t core_tiles(std::uint64_t core) const;
    // Memory traffic and DMA counters come from the memory module.
    Statistics statistics() const;
    Statistics statistics(std::uint64_t core) const;
    MemoryView memory() const;
    MemoryView memory(std::uint64_t core) const;
    // Completed load, compute and store intervals of every core, recorded only
    // when sim.trace_path is set. Not named trace(), which would hide
    // sc_object::trace(sc_trace_file*).
    std::vector<Span> spans() const;
    bool completed() const;
    LiveSnapshot snapshot() const;
    // Register before the first sc_start(). The caller owns and closes file.
    // One core traces as <name>.<signal>; a cluster as <name>.core<i>.<signal>.
    void trace_signals(sc_core::sc_trace_file* file) const;

private:
    std::vector<EncodedWeightPanel> encoded_weights_;
    // Offset of each encoded panel from weight_base, in encoded_weights_ order.
    std::vector<std::uint64_t> weight_offsets_;
    std::uint64_t encoded_storage_bytes_ = 0, dense_storage_bytes_ = 0;
    ClusterPlan plan_;
    std::array<ClusterEstimate, 2> estimates_;
    // Empty in a --timing-only run.
    Output output_;
    sc_core::sc_event state_changed_;
    // Declared before the cores so that the cores, and the requests they own,
    // are destroyed first.
    std::unique_ptr<MemoryPort> memory_;
    std::vector<std::unique_ptr<Core>> cores_;

    // A single passive SC_METHOD drives every signal of every core.
    void observe();
};

void write_trace(std::ostream& stream, const std::vector<Span>& spans);

}  // namespace npu
