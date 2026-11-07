#pragma once

#include "npu/config.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

namespace npu {

// Layout as ShapeConfig: activations M x (G*K), weights G blocks of K x N,
// bias G*N.
struct Problem {
    std::uint64_t m = 0, n = 0, k = 0, groups = 1;
    std::vector<std::int8_t> activations;
    std::vector<std::int8_t> weights;
    std::vector<std::int32_t> bias;
};

// M x (G*N) row-major.
struct Output {
    std::vector<std::int32_t> accumulators;
    std::vector<std::int8_t> quantized;
};

// An output tile of one group; col is the column within the group's N.
struct Tile {
    std::uint64_t id = 0, row = 0, col = 0, rows = 0, cols = 0, group = 0;
};

// Compressed panels are shared by all output-row tiles of a group. Each buffer
// holds its four-byte codec header, LUT or bitmap, and the packed payload.
// The expanded Problem::weights remain exclusively the independent reference.
struct EncodedWeightPanel {
    std::uint64_t group = 0, col = 0, cols = 0, k_start = 0, k = 0;
    std::vector<std::uint8_t> buffer;
    std::uint64_t nonzero_values = 0, retained_values = 0, decode_cycles = 0;
};

std::vector<EncodedWeightPanel> encode_weights(const Config& config, const Problem& problem);
const EncodedWeightPanel& weight_panel(const Config& config,
    const std::vector<EncodedWeightPanel>& panels, const Tile& tile, std::uint64_t block);
void compute_tile_encoded(const Problem& problem, const Config& config,
    const std::vector<EncodedWeightPanel>& panels, const Tile& tile, Output& output);
std::uint64_t encoded_panel_compute_cycles(const Config& config, const Tile& tile,
    std::uint64_t block, const EncodedWeightPanel& panel);

// Shape and work limits. A bit-exact run materializes tensors and the
// reference under max_host_bytes; --timing-only skips both.
constexpr std::uint64_t max_dimension = 1048576;
constexpr std::uint64_t max_functional_macs = 4000000000;
constexpr std::uint64_t max_timing_only_macs = 10000000000000;
constexpr std::uint64_t max_host_bytes = 256ULL * 1024 * 1024;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b);
std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b);
std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b);
void validate(const Config& config);
// G*M*N*K.
std::uint64_t total_macs(const ShapeConfig& shape);
// G * ceil(M / tile_m) * ceil(N / tile_n).
std::uint64_t tile_count(const Config& config);
// SRAM reserved for the slots and the weight cache.
std::uint64_t scratchpad_bytes(const Config& config);
// Panel buffers in use per slot: mapping.panel_buffers, at most one per
// reduction panel.
std::uint64_t slot_panel_buffers(const Config& config);
Problem make_problem(const Config& config);
std::vector<Tile> make_tiles(const Config& config);
Output reference(const Problem& problem, const Requant& quant);
void compute_tile(const Problem& problem, const Requant& quant,
                  const Tile& tile, Output& output);
// TOSA 1.0.2 apply_scale_32 (pseudocode/library/scale_helpers.tosac): returns
// (value * multiplier + round) >> shift, an arithmetic shift, with
// round = 2^(shift-1) moved 2^30 away from zero for double_round when
// shift > 31. Throws std::invalid_argument when multiplier < 0 or shift is
// outside 2..62, and std::overflow_error when value is outside
// [-2^(shift-1), 2^(shift-1)), the spec's REQUIRE on the input.
std::int32_t apply_scale_32(std::int32_t value, std::int32_t multiplier, std::int32_t shift,
                            bool double_round);
// Channel indexes the G*N output channels.
std::int8_t requantize(std::int32_t accumulator, const Requant& quant);
std::int8_t requantize(std::int32_t accumulator, const Requant& quant,
                      std::uint64_t channel);
std::uint64_t reduction_blocks(const Config& config);
std::uint64_t panel_k(const Config& config, std::uint64_t block);
std::uint64_t panel_read_bytes(const Config& config, const Tile& tile, std::uint64_t block);
std::uint64_t panel_compute_cycles(const Config& config, const Tile& tile, std::uint64_t block);
std::uint64_t peak_macs_per_cycle(const Config& config);
// Serial model: one DMA server for reads and writes.
std::uint64_t dma_transfer_cycles(const Config& config, std::uint64_t bytes);
// Serial: bytes_per_cycle. Axi: both channels of every core's port,
// cores * 2 * data_bytes, capped by a nonzero DRAM rate.
std::uint64_t peak_memory_bytes_per_cycle(const Config& config);

// Work split of a cluster of P = arch.cores cores. With G >= P the units are
// whole groups. Otherwise a unit is one row tile of a group's output rows
// (Partition::M: every core with work reads all of that group's weights) or
// one column tile of its output columns (Partition::N: every core reads every
// activation row of its groups), numbered group-major. Core c takes a
// contiguous run of U / P units, one more for each of the first U % P cores,
// and runs its tiles in the single-core order: group, row, column.
enum class ClusterSplit : std::uint8_t { Groups, Rows, Columns };
template <> struct EnumNames<ClusterSplit> {
    static constexpr std::array<std::string_view, 3> names{"groups", "rows", "columns"};
};

struct ClusterPlan {
    Partition partition = Partition::N;  // M or N
    ClusterSplit split = ClusterSplit::Groups;
    std::uint64_t cores = 1, units = 1;
    std::uint64_t first_unit(std::uint64_t core) const;
    std::uint64_t unit_count(std::uint64_t core) const;
    std::uint64_t unit_core(std::uint64_t unit) const;
};

// The plan of partition M or N.
ClusterPlan cluster_plan(const Config& config, Partition partition);
std::uint64_t tile_unit(const Config& config, const ClusterPlan& plan, const Tile& tile);
// Each core's tiles under `plan`, in make_tiles order.
std::vector<std::vector<Tile>> make_core_tiles(const Config& config, const ClusterPlan& plan);

// Weight bytes of one column panel of a group over all its reduction panels,
// split by whether each panel fits the weight cache on its own. With no cache
// every panel is uncacheable.
struct ColumnWeights {
    std::uint64_t cacheable = 0, uncacheable = 0;
};
using ColumnWeightBytes =
    std::function<ColumnWeights(std::uint64_t group, std::uint64_t column_panel)>;
ColumnWeightBytes dense_column_weights(const Config& config);

// Closed-form prediction of a plan. A core's LRU weight cache sees, per group,
// the core's column panels as one cycle repeated once per row tile it runs, so
// after the first pass the cacheable panels all hit when their bytes fit the
// cache together and all miss otherwise; read bytes are exact for that reason.
// Compute cycles time palette and sparse panels at the decode bound.
struct CoreEstimate {
    std::uint64_t tiles = 0, macs = 0, compute_cycles = 0;
    std::uint64_t read_bytes = 0, weight_bytes = 0, write_bytes = 0;
};
struct ClusterEstimate {
    std::vector<CoreEstimate> cores;
    std::uint64_t read_bytes = 0, write_bytes = 0;
    // Makespan lower bound: the slowest core's compute or port transfer
    // cycles (an axi channel moves data_bytes per cycle, the serial server
    // bytes_per_cycle for both directions), or the DRAM time of every byte at
    // dram_bytes_per_cycle.
    std::uint64_t bound_cycles = 0;
};
ClusterEstimate estimate_cluster(const Config& config, const ClusterPlan& plan,
                                 const ColumnWeightBytes& weights);
// Partition::Auto between the m and n estimates: the smaller bound, then
// fewer read plus write bytes, then N.
Partition auto_partition(const ClusterEstimate& m, const ClusterEstimate& n);

// External memory map, laid out as an offline compiler places the tensors:
//   0x8000_0000 activations, row-major, row stride G*K bytes;
//   0x9000_0000 weights, each (group, column panel, reduction panel) packed
//               contiguously in the order the loader first reads it;
//   0xA000_0000 bias and parameters, one record per output channel: INT32
//               bias, then each per-channel vector present (weight zero
//               point, multiplier, shift) as INT32;
//   0xB000_0000 INT8 outputs, row-major, row stride G*N bytes.
// A panel load is the tile's M_t activation runs of K_p bytes, its weight
// panel unless the weight cache holds it, and on the first panel the tile's
// N_t parameter records; a tile store is M_t runs of N_t bytes. Addresses are
// 64-bit and hold no data; they place burst splits. A --timing-only tensor
// larger than its 256 MiB window runs past it, and since every base is 4 KiB
// aligned its bursts split as they would in a disjoint window.
constexpr std::uint64_t activation_base = 0x80000000;
constexpr std::uint64_t weight_base = 0x90000000;
constexpr std::uint64_t param_base = 0xA0000000;
constexpr std::uint64_t output_base = 0xB0000000;

struct AddressRun {
    std::uint64_t base = 0, length = 0;
};

// Offset of a dense weight panel from weight_base.
std::uint64_t dense_weight_offset(const Config& config, const Tile& tile, std::uint64_t block);
// Bytes of one output channel's parameter record.
std::uint64_t param_record_bytes(const Config& config);
// Replaces `runs` with the runs of a panel load. weight_bytes 0 omits the
// weight panel (a weight-cache hit).
void panel_read_runs(const Config& config, const Tile& tile, std::uint64_t block,
                     std::uint64_t weight_offset, std::uint64_t weight_bytes,
                     std::vector<AddressRun>& runs);
void tile_store_runs(const Config& config, const Tile& tile, std::vector<AddressRun>& runs);

// One AXI INCR burst. Beats count the data_bytes-aligned slots it touches, so
// a misaligned first or last beat is a partial beat.
struct Burst {
    std::uint64_t address = 0, bytes = 0, beats = 0;
};
constexpr std::uint64_t axi_boundary_bytes = 4096;
// The burst starting at `address` of a run ending at `end`: as long as the
// run allows, ending at the next 4 KiB boundary or after max_burst_beats
// beats, whichever comes first.
Burst next_burst(const AxiConfig& axi, std::uint64_t address, std::uint64_t end);

// Reserved DRAM intervals in byte-time units; the axi model maps cycle c to
// byte c * dram_bytes_per_cycle, so fractional cycles stay exact integers.
class DramTimeline {
public:
    // Reserves `bytes` in the earliest gap starting at or after `earliest`
    // and returns where the reservation begins.
    std::uint64_t reserve(std::uint64_t earliest, std::uint64_t bytes);
    // Forgets intervals ending at or before `time`.
    void release(std::uint64_t time);
    // Touching reservations merge into one interval.
    std::size_t intervals() const { return busy_.size(); }

private:
    struct Interval {
        std::uint64_t begin, end;
    };
    std::vector<Interval> busy_;  // Sorted, disjoint and never touching.
};
std::uint64_t tile_read_bytes(const Config& config, const Tile& tile);
std::uint64_t tile_write_bytes(const Tile& tile);
std::uint64_t tile_compute_cycles(const Config& config, const Tile& tile);
std::uint64_t output_hash(const Output& output);

}  // namespace npu
