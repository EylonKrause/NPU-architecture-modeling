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

std::uint64_t read_bytes(std::uint64_t k, std::uint64_t rows,
                         std::uint64_t cols) {
    return checked_add(checked_add(checked_mul(rows, k), checked_mul(k, cols)),
                       checked_mul(4, cols));
}

std::uint64_t compute_cycles(const Config& config, std::uint64_t rows,
                             std::uint64_t cols) {
    return checked_add(ceil_div(checked_mul(checked_mul(rows, cols), config.k),
                                 config.macs_per_cycle),
                       config.pipeline_cycles);
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

std::uint64_t scratchpad_bytes(const Config& config) {
    validate_shape(config.m, config.n, config.k);
    require(config.tile_m >= 1 && config.tile_m <= max_dimension &&
                config.tile_n >= 1 && config.tile_n <= max_dimension &&
                config.buffer_slots >= 1 && config.buffer_slots <= 64,
            "scratchpad tile dimensions must be in 1..4096 and slots in 1..64");
    const auto rows = std::min(config.tile_m, config.m);
    const auto cols = std::min(config.tile_n, config.n);
    const auto per_slot = checked_add(read_bytes(config.k, rows, cols),
                                      checked_mul(5, checked_mul(rows, cols)));
    return checked_mul(per_slot, config.buffer_slots);
}

void validate(const Config& config) {
    validate_shape(config.m, config.n, config.k);
    validate_quant(config.quant);
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
    require(std::isfinite(config.frequency_mhz) && config.frequency_mhz >= 1e-6 &&
                config.frequency_mhz <= static_cast<double>(max_timing_value),
            "frequency MHz must be finite and in 0.000001..1000000000 (at least 1 Hz)");
    require(config.timing == "event" || config.timing == "cycle",
            "timing mode must be event or cycle");
    require(config.sram_bytes >= scratchpad_bytes(config),
            "tile buffers exceed configured SRAM capacity");

    // Account for both reference and simulated outputs, INT64 tile scratch,
    // inputs, tile metadata and trace storage before any allocation. The
    // conservative per tile trace and sort budget is 512 bytes, inclusive of
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
    host_bytes = checked_add(host_bytes, checked_mul(512, tile_count));
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
            auto cycles = checked_add(config.memory_latency,
                ceil_div(read_bytes(config.k, row_sizes[r], col_sizes[c]),
                         config.memory_bytes_per_cycle));
            cycles = checked_add(cycles,
                compute_cycles(config, row_sizes[r], col_sizes[c]));
            cycles = checked_add(cycles, checked_add(config.memory_latency,
                ceil_div(checked_mul(row_sizes[r], col_sizes[c]),
                         config.memory_bytes_per_cycle)));
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
        value = static_cast<std::int8_t>(static_cast<int>(next() % 31) - 15);
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
    validate_quant(quant);
    // INT32 * at most INT32_MAX fits in INT64, including INT32_MIN.
    const auto product = static_cast<std::int64_t>(accumulator) * quant.multiplier;
    const auto magnitude = static_cast<std::uint64_t>(product < 0 ? -product : product);
    const auto rounded_magnitude = quant.shift == 0 ? magnitude :
        (magnitude + (UINT64_C(1) << (quant.shift - 1))) >> quant.shift;
    auto scaled = product < 0 ? -static_cast<std::int64_t>(rounded_magnitude) :
                               static_cast<std::int64_t>(rounded_magnitude);
    scaled += quant.output_zero;
    if (quant.relu) scaled = std::max(scaled, static_cast<std::int64_t>(quant.output_zero));
    return static_cast<std::int8_t>(std::clamp<std::int64_t>(scaled, -128, 127));
}

Output reference(const Problem& problem, const Quantization& quant) {
    validate_problem(problem);
    validate_quant(quant);
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
                    problem.weights[static_cast<std::size_t>(k * problem.n + col)]) - quant.weight_zero;
                accumulator += static_cast<std::int64_t>(a) * b;
            }
            const auto index = static_cast<std::size_t>(row * problem.n + col);
            result.accumulators[index] = checked_accumulator(accumulator);
            result.quantized[index] = requantize(result.accumulators[index], quant);
        }
    }
    return result;
}

void compute_tile(const Problem& problem, const Quantization& quant,
                  const Tile& tile, Output& output) {
    validate_problem(problem);
    validate_quant(quant);
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
                    static_cast<std::size_t>(k * problem.n + tile.col + col)]) - quant.weight_zero;
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
            output.quantized[index] = requantize(accumulator, quant);
        }
    }
}

std::uint64_t tile_read_bytes(const Config& config, const Tile& tile) {
    validate_shape(config.m, config.n, config.k);
    validate_tile(tile, config.m, config.n);
    return read_bytes(config.k, tile.rows, tile.cols);
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
