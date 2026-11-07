#pragma once

#include "npu/config.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace npu {

struct Problem {
    std::uint64_t m = 0, n = 0, k = 0;
    std::vector<std::int8_t> activations;
    std::vector<std::int8_t> weights;
    std::vector<std::int32_t> bias;
};

struct Output {
    std::vector<std::int32_t> accumulators;
    std::vector<std::int8_t> quantized;
};

struct Tile {
    std::uint64_t id = 0, row = 0, col = 0, rows = 0, cols = 0;
};

// Compressed panels are shared by all output-row tiles. Each buffer contains
// its four-byte codec header, LUT or bitmap, and genuinely packed payload.
// The expanded Problem::weights remain exclusively the independent reference.
struct EncodedWeightPanel {
    std::uint64_t col = 0, cols = 0, k_start = 0, k = 0;
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

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b);
std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b);
std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b);
void validate(const Config& config);
std::uint64_t scratchpad_bytes(const Config& config);
Problem make_problem(const Config& config);
std::vector<Tile> make_tiles(const Config& config);
Output reference(const Problem& problem, const Requant& quant);
void compute_tile(const Problem& problem, const Requant& quant,
                  const Tile& tile, Output& output);
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
