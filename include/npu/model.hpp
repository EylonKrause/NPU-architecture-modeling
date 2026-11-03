#pragma once

#include "npu/core.hpp"
#include <systemc>
#include <deque>
#include <list>
#include <ostream>
#include <unordered_map>

namespace npu {

struct Span {
    std::string stage;
    std::uint64_t tile_id = 0, start = 0, end = 0, bytes = 0;
    std::uint64_t block_id = 0;
};

struct Statistics {
    std::uint64_t cycles = 0, macs = 0, read_bytes = 0, write_bytes = 0;
    std::uint64_t compute_busy_cycles = 0, dma_busy_cycles = 0;
    std::uint64_t peak_inflight_tiles = 0, slot_wait_cycles = 0;
    std::uint64_t dma_requests = 0, dma_bursts = 0;
    std::uint64_t weight_cache_hits = 0, weight_cache_misses = 0;
    std::uint64_t weight_bytes_saved = 0, peak_cached_weight_bytes = 0;
    std::uint64_t loaded_panels = 0, computed_panels = 0, completed_tiles = 0;
};

// A point-in-time view of the running SystemC model. Resource identifiers are
// meaningful only while their corresponding busy flag is set. Traffic/MAC
// counters commit at operation completion; busy-cycle counters include the
// elapsed portion of an operation currently in flight.
struct LiveSnapshot {
    std::uint64_t cycles = 0;
    bool completed = false, dma_busy = false, compute_busy = false, dma_read = false;
    std::uint64_t inflight_tiles = 0, loaded_tiles = 0, computed_tiles = 0;
    std::uint64_t read_queue_depth = 0, write_queue_depth = 0;
    std::uint64_t dma_tile_id = 0, dma_block_id = 0;
    std::uint64_t compute_tile_id = 0, compute_block_id = 0;
    Statistics statistics;
};

// All state is owned by this module or immutable references whose owners outlive
// sc_start(). Request pointers never outlive their blocking caller's stack frame.
class Accelerator final : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(Accelerator);
    Accelerator(sc_core::sc_module_name name, const Config& config,
                const Problem& problem);
    const Output& output() const { return output_; }
    const Statistics& statistics() const { return stats_; }
    const std::vector<Span>& trace() const { return trace_; }
    bool completed() const { return completed_; }
    LiveSnapshot snapshot() const;
    // Register before the first sc_start(). The caller owns and closes file.
    void trace_signals(sc_core::sc_trace_file* file) const;

private:
    struct Request {
        Tile tile;
        std::uint64_t bytes;
        std::uint64_t block;
        sc_core::sc_event done;
        Request(const Tile& t, std::uint64_t b, std::uint64_t panel)
            : tile(t), bytes(b), block(panel) {}
    };

    const Config& config_;
    const Problem& problem_;
    std::vector<Tile> tiles_;
    Output output_;
    Statistics stats_;
    std::vector<Span> trace_;
    sc_core::sc_fifo<std::uint64_t> free_slots_;
    sc_core::sc_fifo<std::uint64_t> loaded_;
    sc_core::sc_fifo<std::uint64_t> computed_;
    std::deque<Request*> reads_, writes_;
    sc_core::sc_event dma_ready_;
    sc_core::sc_event panel_consumed_;
    struct WeightLine { std::uint64_t key, bytes; };
    std::list<WeightLine> weight_lru_;
    std::unordered_map<std::uint64_t, std::list<WeightLine>::iterator> weight_lines_;
    std::uint64_t cached_bytes_ = 0;
    std::uint64_t inflight_ = 0;
    bool prefer_read_ = true, completed_ = false;
    bool dma_busy_ = false, compute_busy_ = false, dma_read_ = false;
    std::uint64_t dma_start_ = 0, dma_duration_ = 0;
    std::uint64_t compute_start_ = 0, compute_duration_ = 0;
    std::uint64_t dma_tile_id_ = 0, dma_block_id_ = 0;
    std::uint64_t compute_tile_id_ = 0, compute_block_id_ = 0;
    sc_core::sc_event state_changed_;
    // A single passive SC_METHOD owns every signal write. The scheduling
    // threads retain plain state, so simultaneous resource updates are safe.
    sc_core::sc_signal<bool> done_signal_{"done"};
    sc_core::sc_signal<bool> dma_busy_signal_{"dma_busy"};
    sc_core::sc_signal<bool> compute_busy_signal_{"compute_busy"};
    sc_core::sc_signal<bool> dma_read_signal_{"dma_read"};
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

    void observe();
    void changed();
    void load();
    void compute();
    void store();
    void dma();
    void transfer(const Tile& tile, bool read, std::uint64_t block = 0);
    bool weight_cached(const Tile& tile, std::uint64_t block, std::uint64_t bytes);
    void advance(std::uint64_t cycles);
    static std::uint64_t now();
};

void write_trace(std::ostream& stream, const std::vector<Span>& spans);

}  // namespace npu
