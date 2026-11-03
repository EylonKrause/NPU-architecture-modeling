#pragma once

#include "npu/core.hpp"
#include <systemc>
#include <deque>
#include <ostream>

namespace npu {

struct Span {
    std::string stage;
    std::uint64_t tile_id = 0, start = 0, end = 0, bytes = 0;
};

struct Statistics {
    std::uint64_t cycles = 0, macs = 0, read_bytes = 0, write_bytes = 0;
    std::uint64_t compute_busy_cycles = 0, dma_busy_cycles = 0;
    std::uint64_t peak_inflight_tiles = 0, slot_wait_cycles = 0;
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

private:
    struct Request {
        Tile tile;
        std::uint64_t bytes;
        sc_core::sc_event done;
        Request(const Tile& t, std::uint64_t b) : tile(t), bytes(b) {}
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
    std::uint64_t inflight_ = 0;
    bool prefer_read_ = true, completed_ = false;

    void load();
    void compute();
    void store();
    void dma();
    void transfer(const Tile& tile, bool read);
    void advance(std::uint64_t cycles);
    static std::uint64_t now();
};

void write_trace(std::ostream& stream, const std::vector<Span>& spans);

}  // namespace npu
