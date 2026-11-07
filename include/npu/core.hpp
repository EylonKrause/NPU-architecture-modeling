#pragma once

#include "npu/config.hpp"

#include <cstddef>
#include <cstdint>
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
std::uint64_t dma_transfer_cycles(const Config& config, std::uint64_t bytes);
std::uint64_t tile_read_bytes(const Config& config, const Tile& tile);
std::uint64_t tile_write_bytes(const Tile& tile);
std::uint64_t tile_compute_cycles(const Config& config, const Tile& tile);
std::uint64_t output_hash(const Output& output);

}  // namespace npu
