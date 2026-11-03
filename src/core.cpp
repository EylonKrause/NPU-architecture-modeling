#include "npu/core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace npu {
namespace {

constexpr std::uint64_t max_dimension = 4096;
constexpr std::uint64_t max_macs = 100000000;
constexpr std::uint64_t max_host_bytes = 256ULL * 1024 * 1024;
constexpr std::uint64_t max_timing_value = 1000000000;
constexpr std::uint64_t exact_cycle_limit = 1ULL << 53;

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

void validate_quant(const Quantization& quant) {
    require(quant.input_zero >= -128 && quant.input_zero <= 127,
            "input zero point must be signed INT8");
    require(quant.weight_zero >= -128 && quant.weight_zero <= 127,
            "weight zero point must be signed INT8");
    require(quant.output_zero >= -128 && quant.output_zero <= 127,
            "output zero point must be signed INT8");
    require(quant.multiplier <= static_cast<std::uint32_t>(
                std::numeric_limits<std::int32_t>::max()),
            "requantization multiplier must be at most INT32_MAX");
    require(quant.shift <= 31, "requantization shift must be in 0..31");
    require(quant.rounding == "away" || quant.rounding == "even",
            "requantization rounding must be away or even");
    for (const auto value : quant.weight_zero_points)
        require(value >= -128 && value <= 127,
                "per-channel weight zero points must be signed INT8");
    for (const auto value : quant.multipliers)
        require(value <= static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max()),
                "per-channel multipliers must be at most INT32_MAX");
    for (const auto value : quant.shifts)
        require(value <= 31, "per-channel shifts must be in 0..31");
}

void validate_quant_channels(const Quantization& quant, std::uint64_t channels) {
    validate_quant(quant);
    require(quant.weight_zero_points.empty() || quant.weight_zero_points.size() == channels,
            "per-channel weight zero point count must equal N");
    require(quant.multipliers.empty() || quant.multipliers.size() == channels,
            "per-channel multiplier count must equal N");
    require(quant.shifts.empty() || quant.shifts.size() == channels,
            "per-channel shift count must equal N");
}

std::int32_t weight_zero(const Quantization& quant, std::uint64_t channel) {
    return quant.weight_zero_points.empty() ? quant.weight_zero :
        quant.weight_zero_points.at(static_cast<std::size_t>(channel));
}

void validate_shape(std::uint64_t m, std::uint64_t n, std::uint64_t k) {
    require(m >= 1 && m <= max_dimension && n >= 1 && n <= max_dimension &&
                k >= 1 && k <= max_dimension,
            "matrix dimensions must be in 1..4096");
    require(checked_mul(checked_mul(m, n), k) <= max_macs,
            "workload exceeds 100000000 MACs");
}

void validate_problem(const Problem& problem) {
    validate_shape(problem.m, problem.n, problem.k);
    require(problem.activations.size() == checked_mul(problem.m, problem.k),
            "activation vector size does not match M*K");
    require(problem.weights.size() == checked_mul(problem.k, problem.n),
            "weight vector size does not match K*N");
    require(problem.bias.size() == problem.n,
            "bias vector size does not match N");
}

void validate_tile(const Tile& tile, std::uint64_t m, std::uint64_t n) {
    require(tile.rows > 0 && tile.cols > 0, "tile must have positive extents");
    require(tile.row < m && tile.col < n && tile.rows <= m - tile.row &&
                tile.cols <= n - tile.col,
            "tile is outside the output matrix");
}

std::int32_t checked_accumulator(std::int64_t value) {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max())
        throw std::overflow_error("accumulator is outside INT32");
    return static_cast<std::int32_t>(value);
}

void validate_mapping(const Config& config) {
    require(config.weight_bits == 4 || config.weight_bits == 8,
            "weight bits must be 4 or 8");
    require(config.tile_k <= max_dimension,
            "reduction tile dimension must be zero (full K) or in 1..4096");
    require((config.array_rows == 0 && config.array_cols == 0) ||
                (config.array_rows >= 1 && config.array_rows <= 256 &&
                 config.array_cols >= 1 && config.array_cols <= 256),
            "systolic array dimensions must both be zero or in 1..256");
}

std::uint64_t channel_metadata_bytes(const Config& config, std::uint64_t cols) {
    const auto vectors = static_cast<std::uint64_t>(!config.quant.weight_zero_points.empty()) +
        static_cast<std::uint64_t>(!config.quant.multipliers.empty()) +
        static_cast<std::uint64_t>(!config.quant.shifts.empty());
    return checked_mul(checked_mul(4, cols), vectors);
}

std::uint64_t panel_bytes(const Config& config, std::uint64_t k,
                          std::uint64_t rows, std::uint64_t cols, bool first) {
    const auto activations = checked_mul(rows, k);
    const auto weights = ceil_div(checked_mul(checked_mul(k, cols), config.weight_bits), 8);
    const auto metadata = first ? checked_add(checked_mul(4, cols),
                                               channel_metadata_bytes(config, cols)) : 0;
    return checked_add(checked_add(activations, weights), metadata);
}

std::uint64_t compute_panel_cycles(const Config& config, std::uint64_t rows,
                                   std::uint64_t cols, std::uint64_t k) {
    std::uint64_t work = 0;
    if (config.array_rows != 0) {
        const auto waves = checked_mul(ceil_div(rows, config.array_rows),
                                        ceil_div(cols, config.array_cols));
        const auto fill_drain = checked_add(config.array_rows, config.array_cols) - 2;
        work = checked_mul(waves, checked_add(k, fill_drain));
    } else {
        work = ceil_div(checked_mul(checked_mul(rows, cols), k), config.macs_per_cycle);
    }
    return checked_add(work, config.pipeline_cycles);
}

std::uint64_t all_read_bytes(const Config& config, std::uint64_t rows,
                             std::uint64_t cols) {
    std::uint64_t total = 0;
    for (std::uint64_t block = 0; block < reduction_blocks(config); ++block)
        total = checked_add(total, panel_bytes(config, panel_k(config, block), rows, cols, block == 0));
    return total;
}

std::uint64_t compute_cycles(const Config& config, std::uint64_t rows,
                             std::uint64_t cols) {
    std::uint64_t total = 0;
    for (std::uint64_t block = 0; block < reduction_blocks(config); ++block)
        total = checked_add(total, compute_panel_cycles(config, rows, cols, panel_k(config, block)));
    return total;
}

std::int8_t requantize_unchecked(std::int32_t accumulator, const Quantization& quant,
                       std::uint64_t channel) {
    if ((!quant.multipliers.empty() && channel >= quant.multipliers.size()) ||
        (!quant.shifts.empty() && channel >= quant.shifts.size()))
        throw std::out_of_range("requantization channel is outside parameter vectors");
    const auto index = static_cast<std::size_t>(channel);
    const auto multiplier = quant.multipliers.empty() ? quant.multiplier :
        quant.multipliers.at(index);
    const auto shift = quant.shifts.empty() ? quant.shift : quant.shifts.at(index);
    // INT32 * at most INT32_MAX fits in INT64, including INT32_MIN.
    const auto product = static_cast<std::int64_t>(accumulator) * multiplier;
    const auto magnitude = static_cast<std::uint64_t>(product < 0 ? -product : product);
    auto rounded_magnitude = magnitude;
    if (shift != 0) {
        const auto half = UINT64_C(1) << (shift - 1);
        if (quant.rounding == "even") {
            const auto quotient = magnitude >> shift;
            const auto remainder = magnitude & ((UINT64_C(1) << shift) - 1);
            const bool increment = remainder > half ||
                (remainder == half && (quotient & UINT64_C(1)) != 0);
            rounded_magnitude = quotient + static_cast<std::uint64_t>(increment);
        } else {
            rounded_magnitude = (magnitude + half) >> shift;
        }
    }
    auto scaled = product < 0 ? -static_cast<std::int64_t>(rounded_magnitude) :
                               static_cast<std::int64_t>(rounded_magnitude);
    scaled += quant.output_zero;
    if (quant.relu) scaled = std::max(scaled, static_cast<std::int64_t>(quant.output_zero));
    return static_cast<std::int8_t>(std::clamp<std::int64_t>(scaled, -128, 127));
}

}  // namespace

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("unsigned addition overflow");
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::overflow_error("unsigned multiplication overflow");
    return a * b;
}

std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b) {
    if (!b) throw std::invalid_argument("division by zero");
    return a / b + static_cast<std::uint64_t>(a % b != 0);
}

std::uint64_t dma_transfer_cycles(const Config& config, std::uint64_t bytes) {
    if (!bytes) return 0;
    if (!config.dma_burst_bytes)
        return checked_add(config.memory_latency, ceil_div(bytes, config.memory_bytes_per_cycle));
    const auto full = bytes / config.dma_burst_bytes;
    const auto tail = bytes % config.dma_burst_bytes;
    auto cycles = checked_mul(full, checked_add(config.memory_latency,
        ceil_div(config.dma_burst_bytes, config.memory_bytes_per_cycle)));
    if (tail) cycles = checked_add(cycles, checked_add(config.memory_latency,
        ceil_div(tail, config.memory_bytes_per_cycle)));
    return cycles;
}

std::uint64_t scratchpad_bytes(const Config& config) {
    validate_shape(config.m, config.n, config.k);
    validate_mapping(config);
    require(config.tile_m >= 1 && config.tile_m <= max_dimension &&
                config.tile_n >= 1 && config.tile_n <= max_dimension &&
                config.buffer_slots >= 1 && config.buffer_slots <= 64,
            "scratchpad tile dimensions must be in 1..4096 and slots in 1..64");
    const auto rows = std::min(config.tile_m, config.m);
    const auto cols = std::min(config.tile_n, config.n);
    const auto per_slot = checked_add(panel_bytes(config, std::min(config.k, config.tile_k ? config.tile_k : config.k), rows, cols, true),
                                      checked_mul(5, checked_mul(rows, cols)));
    return checked_add(checked_mul(per_slot, config.buffer_slots), config.weight_cache_bytes);
}

void validate(const Config& config) {
    validate_shape(config.m, config.n, config.k);
    validate_quant_channels(config.quant, config.n);
    validate_mapping(config);
    if (config.weight_bits == 4) {
        require(config.quant.weight_zero >= -8 && config.quant.weight_zero <= 7,
                "INT4 weight zero point must be in -8..7");
        for (const auto value : config.quant.weight_zero_points)
            require(value >= -8 && value <= 7,
                    "INT4 per-channel weight zero points must be in -8..7");
    }
    require(config.tile_m >= 1 && config.tile_m <= max_dimension &&
                config.tile_n >= 1 && config.tile_n <= max_dimension,
            "tile dimensions must be in 1..4096");
    require(config.buffer_slots >= 1 && config.buffer_slots <= 64,
            "buffer slots must be in 1..64");
    require(config.macs_per_cycle >= 1 &&
                config.macs_per_cycle <= max_timing_value &&
                config.memory_bytes_per_cycle >= 1 &&
                config.memory_bytes_per_cycle <= max_timing_value,
            "compute and memory rates must be in 1..1000000000");
    require(config.memory_latency <= max_timing_value &&
                config.pipeline_cycles <= max_timing_value,
            "latency and pipeline cycles must be at most 1000000000");
    require(config.dma_burst_bytes <= max_timing_value && config.weight_cache_bytes <= max_timing_value,
            "DMA burst and weight cache sizes must be at most 1000000000 bytes");
    require(std::isfinite(config.frequency_mhz) && config.frequency_mhz >= 1e-6 &&
                config.frequency_mhz <= static_cast<double>(max_timing_value),
            "frequency MHz must be finite and in 0.000001..1000000000 (at least 1 Hz)");
    require(config.timing == "event" || config.timing == "cycle",
            "timing mode must be event or cycle");
    require(config.sram_bytes >= scratchpad_bytes(config),
            "tile buffers exceed configured SRAM capacity");

    // Account for both reference and simulated outputs, INT64 tile scratch,
    // inputs, tile metadata and trace storage before any allocation. The
    // conservative per reduction-panel trace and sort budget is 512 bytes, inclusive of
    // Tile, three Span records (including their strings), and three pointers
    // for chronological trace sorting. Keeping this bound here avoids a
    // dependency from the arithmetic library to the SystemC model headers.
    // Accelerator SRAM and simulator host storage are separate constraints.
    const auto outputs = checked_mul(config.m, config.n);
    const auto tile_count = checked_mul(ceil_div(config.m, config.tile_m),
                                        ceil_div(config.n, config.tile_n));
    std::uint64_t host_bytes = checked_add(checked_mul(config.m, config.k),
                                          checked_mul(config.k, config.n));
    host_bytes = checked_add(host_bytes, checked_mul(4, config.n));
    host_bytes = checked_add(host_bytes, checked_mul(10, outputs));
    host_bytes = checked_add(host_bytes, checked_mul(8, checked_mul(
        std::min(config.m, config.tile_m), std::min(config.n, config.tile_n))));
    host_bytes = checked_add(host_bytes, checked_mul(512, checked_mul(tile_count, reduction_blocks(config))));
    host_bytes = checked_add(host_bytes, channel_metadata_bytes(config, config.n));
    require(host_bytes <= max_host_bytes,
            "estimated simulator working storage exceeds 256 MiB");

    // The serial schedule bounds every overlapped schedule. Four tile shape
    // classes give the exact serial sum without iterating over huge tile lists.
    std::uint64_t serial_cycles = 0;
    const std::uint64_t row_sizes[] = {config.tile_m, config.m % config.tile_m};
    const std::uint64_t row_counts[] = {config.m / config.tile_m,
                                       config.m % config.tile_m ? 1ULL : 0ULL};
    const std::uint64_t col_sizes[] = {config.tile_n, config.n % config.tile_n};
    const std::uint64_t col_counts[] = {config.n / config.tile_n,
                                       config.n % config.tile_n ? 1ULL : 0ULL};
    for (unsigned r = 0; r < 2; ++r) {
        for (unsigned c = 0; c < 2; ++c) {
            const auto count = checked_mul(row_counts[r], col_counts[c]);
            if (!count) continue;
            std::uint64_t cycles = 0;
            for (std::uint64_t block = 0; block < reduction_blocks(config); ++block) {
                const auto bytes = panel_bytes(config, panel_k(config, block),
                                                row_sizes[r], col_sizes[c], block == 0);
                cycles = checked_add(cycles, dma_transfer_cycles(config, bytes));
            }
            cycles = checked_add(cycles,
                compute_cycles(config, row_sizes[r], col_sizes[c]));
            cycles = checked_add(cycles, dma_transfer_cycles(config,
                checked_mul(row_sizes[r], col_sizes[c])));
            serial_cycles = checked_add(serial_cycles, checked_mul(count, cycles));
        }
    }
    require(serial_cycles < exact_cycle_limit,
            "simulation cycle bound must be less than 2^53");
    // Cycle mode performs one host wait per stage cycle. Bound the complete
    // serial work even when overlap would make the reported makespan shorter.
    require(config.timing != "cycle" || serial_cycles <= 10000000,
            "cycle mode serial work must be at most 10000000 cycles; use event mode for larger runs");
}

Problem make_problem(const Config& config) {
    validate(config);
    Problem problem;
    problem.m = config.m;
    problem.n = config.n;
    problem.k = config.k;
    problem.activations.resize(static_cast<std::size_t>(config.m * config.k));
    problem.weights.resize(static_cast<std::size_t>(config.k * config.n));
    problem.bias.resize(static_cast<std::size_t>(config.n));
    auto state = static_cast<std::uint32_t>(config.seed);
    const auto next = [&state]() {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        return state;
    };
    for (auto& value : problem.activations)
        value = static_cast<std::int8_t>(static_cast<int>(next() % 31) - 15);
    for (auto& value : problem.weights)
        value = config.weight_bits == 4 ?
            static_cast<std::int8_t>(static_cast<int>(next() % 16) - 8) :
            static_cast<std::int8_t>(static_cast<int>(next() % 31) - 15);
    for (auto& value : problem.bias)
        value = static_cast<std::int32_t>(next() % 127) - 63;
    return problem;
}

std::vector<Tile> make_tiles(const Config& config) {
    validate(config);
    std::vector<Tile> tiles;
    tiles.reserve(static_cast<std::size_t>(checked_mul(
        ceil_div(config.m, config.tile_m), ceil_div(config.n, config.tile_n))));
    for (std::uint64_t row = 0; row < config.m; row += config.tile_m) {
        for (std::uint64_t col = 0; col < config.n; col += config.tile_n) {
            tiles.push_back({static_cast<std::uint64_t>(tiles.size()), row, col,
                             std::min(config.tile_m, config.m - row),
                             std::min(config.tile_n, config.n - col)});
        }
    }
    return tiles;
}

std::int8_t requantize(std::int32_t accumulator, const Quantization& quant) {
    return requantize(accumulator, quant, 0);
}

std::int8_t requantize(std::int32_t accumulator, const Quantization& quant,
                       std::uint64_t channel) {
    validate_quant(quant);
    return requantize_unchecked(accumulator, quant, channel);
}

Output reference(const Problem& problem, const Quantization& quant) {
    validate_problem(problem);
    validate_quant_channels(quant, problem.n);
    Output result;
    const auto size = static_cast<std::size_t>(problem.m * problem.n);
    result.accumulators.resize(size);
    result.quantized.resize(size);
    // Scalar reference: each complete dot product is evaluated independently.
    for (std::uint64_t row = 0; row < problem.m; ++row) {
        for (std::uint64_t col = 0; col < problem.n; ++col) {
            std::int64_t accumulator = problem.bias[static_cast<std::size_t>(col)];
            for (std::uint64_t k = 0; k < problem.k; ++k) {
                const auto a = static_cast<std::int32_t>(
                    problem.activations[static_cast<std::size_t>(row * problem.k + k)]) - quant.input_zero;
                const auto b = static_cast<std::int32_t>(
                    problem.weights[static_cast<std::size_t>(k * problem.n + col)]) - weight_zero(quant, col);
                accumulator += static_cast<std::int64_t>(a) * b;
            }
            const auto index = static_cast<std::size_t>(row * problem.n + col);
            result.accumulators[index] = checked_accumulator(accumulator);
            result.quantized[index] = requantize_unchecked(result.accumulators[index], quant, col);
        }
    }
    return result;
}

void compute_tile(const Problem& problem, const Quantization& quant,
                  const Tile& tile, Output& output) {
    validate_problem(problem);
    validate_quant_channels(quant, problem.n);
    validate_tile(tile, problem.m, problem.n);
    const auto output_size = static_cast<std::size_t>(problem.m * problem.n);
    if (output.accumulators.empty() && output.quantized.empty()) {
        output.accumulators.resize(output_size);
        output.quantized.resize(output_size);
    }
    require(output.accumulators.size() == output_size &&
                output.quantized.size() == output_size,
            "output vectors do not match M*N");
    std::vector<std::int64_t> sums(static_cast<std::size_t>(tile.rows * tile.cols));
    for (std::uint64_t row = 0; row < tile.rows; ++row)
        for (std::uint64_t col = 0; col < tile.cols; ++col)
            sums[static_cast<std::size_t>(row * tile.cols + col)] =
                problem.bias[static_cast<std::size_t>(tile.col + col)];
    // Tiled datapath uses K-major outer products, independent of the reference
    // dot-product loop ordering, and only commits final INT32-safe values.
    for (std::uint64_t k = 0; k < problem.k; ++k) {
        for (std::uint64_t row = 0; row < tile.rows; ++row) {
            const auto a = static_cast<std::int32_t>(problem.activations[
                static_cast<std::size_t>((tile.row + row) * problem.k + k)]) - quant.input_zero;
            for (std::uint64_t col = 0; col < tile.cols; ++col) {
                const auto b = static_cast<std::int32_t>(problem.weights[
                    static_cast<std::size_t>(k * problem.n + tile.col + col)]) - weight_zero(quant, tile.col + col);
                sums[static_cast<std::size_t>(row * tile.cols + col)] +=
                    static_cast<std::int64_t>(a) * b;
            }
        }
    }
    for (std::uint64_t row = 0; row < tile.rows; ++row) {
        for (std::uint64_t col = 0; col < tile.cols; ++col) {
            const auto accumulator = checked_accumulator(
                sums[static_cast<std::size_t>(row * tile.cols + col)]);
            const auto index = static_cast<std::size_t>((tile.row + row) * problem.n + tile.col + col);
            output.accumulators[index] = accumulator;
            output.quantized[index] = requantize_unchecked(accumulator, quant, tile.col + col);
        }
    }
}

std::uint64_t reduction_blocks(const Config& config) {
    validate_shape(config.m, config.n, config.k);
    validate_mapping(config);
    return ceil_div(config.k, config.tile_k ? config.tile_k : config.k);
}

std::uint64_t panel_k(const Config& config, std::uint64_t block) {
    const auto blocks = reduction_blocks(config);
    require(block < blocks, "reduction block is outside K");
    const auto size = config.tile_k ? config.tile_k : config.k;
    return std::min(size, config.k - checked_mul(block, size));
}

std::uint64_t panel_read_bytes(const Config& config, const Tile& tile,
                               std::uint64_t block) {
    validate_shape(config.m, config.n, config.k);
    validate_tile(tile, config.m, config.n);
    validate_mapping(config);
    return panel_bytes(config, panel_k(config, block), tile.rows, tile.cols, block == 0);
}

std::uint64_t panel_compute_cycles(const Config& config, const Tile& tile,
                                   std::uint64_t block) {
    validate_shape(config.m, config.n, config.k);
    validate_tile(tile, config.m, config.n);
    validate_mapping(config);
    require(config.macs_per_cycle >= 1 && config.macs_per_cycle <= max_timing_value &&
                config.pipeline_cycles <= max_timing_value,
            "compute rate or pipeline timing is outside supported limits");
    return compute_panel_cycles(config, tile.rows, tile.cols, panel_k(config, block));
}

std::uint64_t peak_macs_per_cycle(const Config& config) {
    validate_mapping(config);
    return config.array_rows ? checked_mul(config.array_rows, config.array_cols) : config.macs_per_cycle;
}

std::uint64_t tile_read_bytes(const Config& config, const Tile& tile) {
    validate_shape(config.m, config.n, config.k);
    validate_tile(tile, config.m, config.n);
    validate_mapping(config);
    return all_read_bytes(config, tile.rows, tile.cols);
}

std::uint64_t tile_write_bytes(const Tile& tile) {
    require(tile.rows >= 1 && tile.rows <= max_dimension &&
                tile.cols >= 1 && tile.cols <= max_dimension,
            "tile extents must be in 1..4096");
    return checked_mul(tile.rows, tile.cols);
}

std::uint64_t tile_compute_cycles(const Config& config, const Tile& tile) {
    validate_shape(config.m, config.n, config.k);
    require(config.macs_per_cycle >= 1 && config.macs_per_cycle <= max_timing_value &&
                config.pipeline_cycles <= max_timing_value,
            "compute rate or pipeline timing is outside supported limits");
    validate_tile(tile, config.m, config.n);
    validate_mapping(config);
    return compute_cycles(config, tile.rows, tile.cols);
}

std::uint64_t output_hash(const Output& output) {
    require(output.accumulators.size() == output.quantized.size(),
            "hash requires equal accumulator and quantized vector sizes");
    // FNV-1a 64: offset=14695981039346656037, prime=1099511628211.
    // Serialize every row-major INT32 accumulator as four little-endian bytes,
    // then every row-major INT8 output as its unsigned byte representation.
    // There are no length prefixes, separators, native padding or host-endian
    // dependencies. Unsigned multiplication wraps modulo 2^64 by definition.
    std::uint64_t hash = UINT64_C(14695981039346656037);
    const auto append = [&hash](std::uint8_t byte) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    };
    for (const auto accumulator : output.accumulators) {
        const auto bits = static_cast<std::uint32_t>(accumulator);
        for (unsigned shift = 0; shift < 32; shift += 8)
            append(static_cast<std::uint8_t>(bits >> shift));
    }
    for (const auto value : output.quantized) append(static_cast<std::uint8_t>(value));
    return hash;
}

}  // namespace npu
