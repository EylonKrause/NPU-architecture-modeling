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
    // With shape.conv, the NHWC input tensor whose im2col view `activations`
    // holds; empty for a plain GEMM.
    std::vector<std::int8_t> input;
};

// M x (G*N) row-major.
struct Output {
    std::vector<std::int32_t> accumulators;
    std::vector<std::int8_t> quantized;
};

// An output tile of one group; col is the column within the group's N. On the
// depthwise datapath (channel_wise) group is 0 and col, cols select output
// channels, one per group; see grid_groups.
struct Tile {
    std::uint64_t id = 0, row = 0, col = 0, rows = 0, cols = 0, group = 0;
};

// The depthwise datapath runs the problem: depthwise_macs_per_cycle > 0,
// G > 1 and N = 1.
bool channel_wise(const Config& config);
// The output tile grid: grid_groups groups of grid_columns columns each, every
// group cut into ceil(M / tile_m) x ceil(columns / tile_n) tiles taken
// group-major, then row, then column. A GEMM's grid is G groups of N; a
// channel-wise problem's is one group of G columns, so its tiles span tile_n
// consecutive groups and run row tile outer, channel tile inner.
std::uint64_t grid_groups(const Config& config);
std::uint64_t grid_columns(const Config& config);

// Geometry of shape.conv, validated against M, K and G.
struct ConvGeometry {
    std::uint64_t output_h = 0, output_w = 0;
    // Input channels per group, and K of one kernel row (kernel_w * C_g).
    std::uint64_t group_channels = 0, kernel_row_k = 0;
};
bool implicit_conv(const ShapeConfig& shape);
ConvGeometry conv_geometry(const ShapeConfig& shape);
// Kernel rows [first, first + count) of reduction block `block`; tile_k is 0
// or a whole number of kernel rows.
struct KernelRows {
    std::uint64_t first = 0, count = 0;
};
KernelRows conv_kernel_rows(const Config& config, std::uint64_t block);
// One input row an activation panel reads: columns first..last inclusive.
struct InputSpan {
    std::uint64_t row = 0, first = 0, last = 0;
};
// The input rows that output pixels [first_output, first_output + outputs)
// read over the kernel rows of `block`, ascending: per row the hull of the
// needed columns, clipped to the image. A stride wider than the kernel
// footprint leaves skipped columns inside the hull; padding reads nothing.
void conv_input_spans(const Config& config, std::uint64_t first_output, std::uint64_t outputs,
                      std::uint64_t block, std::vector<InputSpan>& spans);
// The M x (G*K) operand of an NHWC input tensor; padding is the input zero point.
std::vector<std::int8_t> im2col(const Config& config, const std::vector<std::int8_t>& input);

// Compressed panels are shared by all output-row tiles of a group. Each buffer
// holds its four-byte codec header, LUT or bitmap, and the packed payload.
// With an encoded format, the expanded Problem::weights feed the reference only.
struct EncodedWeightPanel {
    std::uint64_t group = 0, col = 0, cols = 0, k_start = 0, k = 0;
    std::vector<std::uint8_t> buffer;
    std::uint64_t nonzero_values = 0, retained_values = 0, decode_cycles = 0;
};

std::vector<EncodedWeightPanel> encode_weights(const Config& config, const Problem& problem);
const EncodedWeightPanel& weight_panel(const Config& config,
                                       const std::vector<EncodedWeightPanel>& panels,
                                       const Tile& tile, std::uint64_t block);
void compute_tile_encoded(const Problem& problem, const Config& config,
                          const std::vector<EncodedWeightPanel>& panels, const Tile& tile,
                          Output& output);
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
// SRAM reserved for the slots, the weight cache and the activation cache.
std::uint64_t scratchpad_bytes(const Config& config);
// Panel buffers in use per slot: mapping.panel_buffers, at most one per
// reduction panel.
std::uint64_t slot_panel_buffers(const Config& config);
Problem make_problem(const Config& config);
std::vector<Tile> make_tiles(const Config& config);
Output reference(const Problem& problem, const Requant& quant);
void compute_tile(const Problem& problem, const Requant& quant, const Tile& tile, Output& output);
// A depthwise-datapath tile: output channels col..col + cols - 1, each its own
// group's single column.
void compute_channel_tile(const Problem& problem, const Requant& quant, const Tile& tile,
                          Output& output);
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
std::int8_t requantize(std::int32_t accumulator, const Requant& quant, std::uint64_t channel);
std::uint64_t reduction_blocks(const Config& config);
std::uint64_t panel_k(const Config& config, std::uint64_t block);
std::uint64_t panel_read_bytes(const Config& config, const Tile& tile, std::uint64_t block);
// Activation bytes of a panel load: M_t x K_p for a GEMM tile, M_t x C_t x K_p
// on the depthwise datapath, or with shape.conv the input footprint of
// conv_input_spans times the panel's input channels.
std::uint64_t activation_panel_bytes(const Config& config, const Tile& tile, std::uint64_t block);
// Activation-cache line of a panel: (group, or channel tile on the depthwise
// datapath; row tile; reduction block).
std::uint64_t activation_cache_key(const Config& config, const Tile& tile, std::uint64_t block);
std::uint64_t panel_compute_cycles(const Config& config, const Tile& tile, std::uint64_t block);
// MACs per cycle of the datapath that runs the problem: the array or vector
// datapath, or depthwise_macs_per_cycle when channel_wise.
std::uint64_t peak_macs_per_cycle(const Config& config);
// Serial model: one DMA server for reads and writes.
std::uint64_t dma_transfer_cycles(const Config& config, std::uint64_t bytes);
// Serial: bytes_per_cycle. Axi: both channels of every core's port,
// cores * 2 * data_bytes, capped by a nonzero DRAM rate.
std::uint64_t peak_memory_bytes_per_cycle(const Config& config);

// Work split of a cluster of P = arch.cores cores over the output grid
// (grid_groups). With grid groups >= P the units are whole groups. Otherwise a
// unit is one row tile of a group's output rows (Partition::M: every core with
// work reads all of that group's weights) or one column tile of its output
// columns (Partition::N: every core reads every activation row of its groups),
// numbered group-major; on the depthwise datapath a column tile is a channel
// tile. Core c takes a contiguous run of U / P units, one more for each of the
// first U % P cores, and runs its tiles in the single-core order: group, row,
// column.
enum class ClusterSplit : std::uint8_t { Groups, Rows, Columns };
template <>
struct EnumNames<ClusterSplit> {
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

// One column panel of a group over all its reduction panels: its weight bytes,
// split by whether each panel fits the weight cache on its own (with no cache
// every panel is uncacheable), and the compute cycles of one of its tiles with
// tile_m rows and with the M % tile_m edge rows. A row extent the shape lacks
// costs 0.
struct ColumnCost {
    std::uint64_t cacheable = 0, uncacheable = 0;
    std::uint64_t full_rows_cycles = 0, edge_rows_cycles = 0;
};
using ColumnCosts = std::function<ColumnCost(std::uint64_t group, std::uint64_t column_panel)>;
// Dense panels, in closed form.
ColumnCosts dense_column_costs(const Config& config);
// The encoded panels' own buffer sizes and compute cycles, as the loader reads
// and the MAC array runs them; `panels` must outlive the result.
ColumnCosts encoded_column_costs(const Config& config,
                                 const std::vector<EncodedWeightPanel>& panels);

// Closed-form prediction of a plan. A core's LRU weight cache sees, per group,
// the core's column panels as one cycle repeated once per row tile it runs, so
// after the first pass the cacheable panels all hit when their bytes fit the
// cache together and all miss otherwise. Its activation cache sees one row
// tile's panels repeated once per column tile it runs, with the same outcome,
// so read bytes are exact. Compute cycles add the column costs,
// so they equal each core's compute busy cycles.
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
                                 const ColumnCosts& costs);
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
// A panel load is the tile's activation runs unless the activation cache holds
// them, its weight panel unless the weight cache holds it, and on the first
// panel the tile's N_t parameter records; a tile store is M_t runs of N_t
// bytes. Activation runs are M_t runs of K_p bytes for a GEMM tile; on the
// depthwise datapath per row the C_t groups' K_p slices, one run when they
// cover all of K. With shape.conv the activation window holds the NHWC input
// instead, and the runs are the panel's input rows (conv_input_spans): one run
// per row over its column hull when the panel's channels are all of C, else
// one run per pixel of its channel range. Addresses are 64-bit and hold no
// data; they place burst splits. A --timing-only tensor larger than its
// 256 MiB window runs past it, and since every base is 4 KiB aligned its
// bursts split as they would in a disjoint window.
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
// weight panel (a weight-cache hit) and activations false the activation runs
// (an activation-cache hit).
void panel_read_runs(const Config& config, const Tile& tile, std::uint64_t block,
                     std::uint64_t weight_offset, std::uint64_t weight_bytes,
                     std::vector<AddressRun>& runs, bool activations = true);
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
