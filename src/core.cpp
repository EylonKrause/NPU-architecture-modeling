#include "npu/core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace npu {
namespace {

constexpr std::uint64_t max_timing_value = 1000000000;
constexpr std::uint64_t exact_cycle_limit = 1ULL << 53;
// Host bytes held per output tile (Tile record and FIFO token), and per
// reduction panel when a stage trace is kept (three Span records, their
// strings and the pointers used to sort them).
constexpr std::uint64_t tile_record_bytes = 64;
constexpr std::uint64_t trace_panel_bytes = 512;

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

std::string range_text(std::int64_t low, std::int64_t high) {
    return std::to_string(low) + ".." + std::to_string(high);
}

const std::string& dimension_range() {
    static const std::string text = range_text(1, static_cast<std::int64_t>(max_dimension));
    return text;
}

// Shift range of a rescale mode: native keeps the original 0..31; the TOSA
// modes take the apply_scale_32 range 2..62.
struct ShiftRange { std::int32_t low, high; };
ShiftRange shift_range(RescaleMode mode) {
    return mode == RescaleMode::Native ? ShiftRange{0, 31} : ShiftRange{2, 62};
}

std::string shift_rule(RescaleMode mode) {
    const auto range = shift_range(mode);
    return range_text(range.low, range.high) + " for " + std::string(to_string(mode)) + " rescale";
}

bool int8_value(std::int32_t value) { return value >= -128 && value <= 127; }

// O(1): every scalar field and the mode rules. Per-channel entries are
// checked where they are used.
void validate_quant_scalars(const Requant& quant) {
    require(int8_value(quant.input_zero), "input zero point must be signed INT8");
    require(int8_value(quant.weight_zero), "weight zero point must be signed INT8");
    require(int8_value(quant.output_zero), "output zero point must be signed INT8");
    require(quant.multiplier >= 0, "requantization multiplier must be in 0..2147483647");
    const auto range = shift_range(quant.rescale);
    if (quant.shift < range.low || quant.shift > range.high)
        fail("requantization shift must be in " + shift_rule(quant.rescale));
    require(quant.rescale == RescaleMode::Native || quant.rounding == Rounding::HalfAwayFromZero,
            "rounding even applies to native rescale only; tosa-single and tosa-double fix "
            "their own rounding");
}

void validate_channel_entry(const Requant& quant, std::size_t index) {
    if (index < quant.weight_zero_points.size() && !int8_value(quant.weight_zero_points[index]))
        fail("per-channel weight zero points must be signed INT8");
    if (index < quant.multipliers.size() && quant.multipliers[index] < 0)
        fail("per-channel multipliers must be in 0..2147483647");
    if (index < quant.shifts.size()) {
        const auto range = shift_range(quant.rescale);
        if (quant.shifts[index] < range.low || quant.shifts[index] > range.high)
            fail("per-channel shifts must be in " + shift_rule(quant.rescale));
    }
}

void validate_channel_counts(const Requant& quant, std::uint64_t channels) {
    require(quant.weight_zero_points.empty() || quant.weight_zero_points.size() == channels,
            "per-channel weight zero point count must equal G*N");
    require(quant.multipliers.empty() || quant.multipliers.size() == channels,
            "per-channel multiplier count must equal G*N");
    require(quant.shifts.empty() || quant.shifts.size() == channels,
            "per-channel shift count must equal G*N");
}

void validate_quant_channels(const Requant& quant, std::uint64_t channels) {
    validate_quant_scalars(quant);
    validate_channel_counts(quant, channels);
    for (std::size_t index = 0; index < static_cast<std::size_t>(channels); ++index)
        validate_channel_entry(quant, index);
}

// Checks the entries of the channels [first, first + count) only, so a tile
// costs O(tile width) rather than O(G*N).
void validate_quant_range(const Requant& quant, std::uint64_t channels,
                          std::uint64_t first, std::uint64_t count) {
    validate_quant_scalars(quant);
    validate_channel_counts(quant, channels);
    for (auto index = first; index < first + count; ++index)
        validate_channel_entry(quant, static_cast<std::size_t>(index));
}

std::int32_t weight_zero(const Requant& quant, std::uint64_t channel) {
    return quant.weight_zero_points.empty() ? quant.weight_zero :
        quant.weight_zero_points.at(static_cast<std::size_t>(channel));
}

bool dimension(std::uint64_t value) { return value >= 1 && value <= max_dimension; }

void validate_shape(std::uint64_t m, std::uint64_t n, std::uint64_t k, std::uint64_t groups,
                    bool timing_only) {
    if (!dimension(m) || !dimension(n) || !dimension(k))
        fail("matrix dimensions must be in " + dimension_range());
    if (!dimension(groups)) fail("groups must be in " + dimension_range());
    // G*N and G*K are the widths of the stored output and activation matrices.
    if (groups * n > max_dimension || groups * k > max_dimension)
        fail("groups*N and groups*K must be at most " + std::to_string(max_dimension));
    // Each factor is at most 2^20, so the product stays below 2^60.
    const auto macs = m * (groups * n) * k;
    if (timing_only && macs > max_timing_only_macs)
        fail("workload exceeds " + std::to_string(max_timing_only_macs) +
             " MACs, the --timing-only limit");
    if (!timing_only && macs > max_functional_macs)
        fail("workload exceeds " + std::to_string(max_functional_macs) +
             " MACs for a bit-exact run; --timing-only allows up to " +
             std::to_string(max_timing_only_macs));
}

void validate_shape(const Config& config) {
    const auto& s = config.shape;
    validate_shape(s.m, s.n, s.k, s.groups, config.sim.timing_only);
}

std::uint64_t channels(const Problem& problem) { return problem.groups * problem.n; }

void validate_problem(const Problem& problem) {
    validate_shape(problem.m, problem.n, problem.k, problem.groups, false);
    require(problem.activations.size() ==
                checked_mul(problem.m, checked_mul(problem.groups, problem.k)),
            "activation vector size does not match M*G*K");
    require(problem.weights.size() ==
                checked_mul(problem.groups, checked_mul(problem.k, problem.n)),
            "weight vector size does not match G*K*N");
    require(problem.bias.size() == channels(problem), "bias vector size does not match G*N");
}

void validate_tile(const Tile& tile, std::uint64_t m, std::uint64_t n, std::uint64_t groups) {
    require(tile.rows > 0 && tile.cols > 0, "tile must have positive extents");
    require(tile.group < groups && tile.row < m && tile.col < n && tile.rows <= m - tile.row &&
                tile.cols <= n - tile.col,
            "tile is outside the output matrix");
}

void validate_tile(const Tile& tile, const ShapeConfig& shape) {
    validate_tile(tile, shape.m, shape.n, shape.groups);
}

std::int32_t checked_accumulator(std::int64_t value) {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max())
        throw std::overflow_error("accumulator is outside INT32");
    return static_cast<std::int32_t>(value);
}

void validate_mapping(const Config& config) {
    const auto& weights = config.weights;
    const auto& compute = config.arch.compute;
    require(weights.bits == 4 || weights.bits == 8, "weight bits must be 4 or 8");
    require(weights.palette_bits == 1 || weights.palette_bits == 2 || weights.palette_bits == 4 ||
                weights.palette_bits == 8, "palette bits must be 1, 2, 4 or 8");
    require(weights.sparse_block >= 1 && weights.sparse_block <= 4096,
            "sparse block must be in 1..4096");
    require(compute.weight_decode_values_per_cycle >= 1 &&
                compute.weight_decode_values_per_cycle <= max_timing_value,
            "weight decode rate must be in 1..1000000000");
    require(weights.encoding != WeightEncoding::Sparse ||
                (compute.array_rows == 0 && compute.array_cols == 0),
            "sparse zero-skipping is supported only by the throughput datapath, not systolic timing");
    if (config.mapping.tile_k > max_dimension)
        fail("reduction tile dimension must be zero (full K) or in " + dimension_range());
    require(config.mapping.panel_buffers == 1 || config.mapping.panel_buffers == 2,
            "panel buffers must be 1 or 2");
    require((compute.array_rows == 0 && compute.array_cols == 0) ||
                (compute.array_rows >= 1 && compute.array_rows <= 256 &&
                 compute.array_cols >= 1 && compute.array_cols <= 256),
            "systolic array dimensions must both be zero or in 1..256");
}

std::uint64_t panel_size(const Config& config) {
    return config.mapping.tile_k ? config.mapping.tile_k : config.shape.k;
}

void validate_compute_rates(const Config& config) {
    const auto& compute = config.arch.compute;
    require(compute.macs_per_cycle >= 1 && compute.macs_per_cycle <= max_timing_value &&
                compute.pipeline_cycles <= max_timing_value,
            "compute rate or pipeline timing is outside supported limits");
}

std::uint64_t encoded_weight_bound(const Config& config, std::uint64_t count) {
    const auto& weights = config.weights;
    if (weights.encoding == WeightEncoding::Palette)
        return checked_add(checked_add(4, std::min(count, UINT64_C(1) << weights.palette_bits)),
                           ceil_div(checked_mul(count, weights.palette_bits), 8));
    const auto dense = ceil_div(checked_mul(count, weights.bits), 8);
    if (weights.encoding == WeightEncoding::Sparse)
        return checked_add(checked_add(4, ceil_div(ceil_div(count, weights.sparse_block), 8)),
                           dense);
    return dense;
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
    const auto weights = encoded_weight_bound(config, checked_mul(k, cols));
    const auto metadata = first ? checked_add(checked_mul(4, cols),
                                               channel_metadata_bytes(config, cols)) : 0;
    return checked_add(checked_add(activations, weights), metadata);
}

std::uint64_t compute_panel_cycles(const Config& config, std::uint64_t rows,
                                   std::uint64_t cols, std::uint64_t k) {
    const auto& compute = config.arch.compute;
    std::uint64_t work = 0;
    if (compute.array_rows != 0) {
        const auto waves = checked_mul(ceil_div(rows, compute.array_rows),
                                        ceil_div(cols, compute.array_cols));
        const auto fill_drain = checked_add(compute.array_rows, compute.array_cols) - 2;
        work = checked_mul(waves, checked_add(k, fill_drain));
    } else {
        work = ceil_div(checked_mul(checked_mul(rows, cols), k), compute.macs_per_cycle);
    }
    auto cycles = checked_add(work, compute.pipeline_cycles);
    if (config.weights.encoding != WeightEncoding::Dense) {
        const auto count = checked_mul(k, cols);
        const auto rate = compute.weight_decode_values_per_cycle;
        cycles = checked_add(cycles, ceil_div(count, rate));
        if (config.weights.encoding == WeightEncoding::Sparse)
            cycles = checked_add(cycles,
                                 ceil_div(ceil_div(count, config.weights.sparse_block), rate));
    }
    return cycles;
}

std::uint64_t all_read_bytes(const Config& config, std::uint64_t rows,
                             std::uint64_t cols) {
    std::uint64_t total = 0;
    for (std::uint64_t block = 0; block < reduction_blocks(config); ++block)
        total = checked_add(total,
                            panel_bytes(config, panel_k(config, block), rows, cols, block == 0));
    return total;
}

std::uint64_t compute_cycles(const Config& config, std::uint64_t rows,
                             std::uint64_t cols) {
    std::uint64_t total = 0;
    for (std::uint64_t block = 0; block < reduction_blocks(config); ++block)
        total = checked_add(total,
                            compute_panel_cycles(config, rows, cols, panel_k(config, block)));
    return total;
}

// C++17 leaves >> of a negative value implementation-defined; TOSA specifies
// an arithmetic shift, which is floor division by 2^shift.
std::int64_t floor_shift(std::int64_t value, std::int32_t shift) {
    return value >= 0 ? value >> shift : -((-(value + 1)) >> shift) - 1;
}

// Requantization arithmetic. The TOSA modes are apply_scale_32 from TOSA 1.0.2
// RESCALE (scale32, int32 input, int8 output). With tosa.shift = 31 - tfl.shift
// and the same Q31 multiplier they are bit-identical to:
//   tosa-single: TFLite MultiplyByQuantizedMultiplier built with
//                TFLITE_SINGLE_ROUNDING, and CMSIS-NN arm_nn_requantize built
//                with CMSIS_NN_USE_SINGLE_ROUNDING (ties toward +infinity);
//   tosa-double: the TFLite default, gemmlowp SaturatingRoundingDoublingHighMul
//                then RoundingDivideByPOT, and the CMSIS-NN default (ties up at
//                bit 31, then away from zero at the final shift).
// RESCALE.tosac at tag v1.0.2 sign-extends `value` from itself; main reads
// in_value. The int32 accumulator here is the in_value of that corrected form,
// with input_zp = 0. Native rounds the magnitude of acc * multiplier, ties
// away from zero or to even; it matches neither TOSA mode on negative ties.
std::int32_t scale32(std::int32_t value, std::int32_t multiplier, std::int32_t shift,
                     bool double_round) {
    const auto half = INT64_C(1) << (shift - 1);
    if (value < -half || value >= half)
        throw std::overflow_error("accumulator " + std::to_string(value) + " is outside [-" +
                                  std::to_string(half) + ", " + std::to_string(half) +
                                  "), the apply_scale_32 input range for shift " +
                                  std::to_string(shift));
    auto round = half;
    if (double_round && shift > 31) round += value >= 0 ? INT64_C(1) << 30 : -(INT64_C(1) << 30);
    // |value| <= 2^31 and multiplier < 2^31, so the sum stays below 2^63, and
    // the input range keeps the result below 2^30 in magnitude.
    return static_cast<std::int32_t>(
        floor_shift(static_cast<std::int64_t>(value) * multiplier + round, shift));
}

std::int64_t native_scale(std::int32_t accumulator, std::int32_t multiplier,
                          std::int32_t shift, Rounding rounding) {
    // INT32 * at most INT32_MAX fits in INT64, including INT32_MIN.
    const auto product = static_cast<std::int64_t>(accumulator) * multiplier;
    const auto magnitude = static_cast<std::uint64_t>(product < 0 ? -product : product);
    auto rounded_magnitude = magnitude;
    if (shift != 0) {
        const auto bits = static_cast<unsigned>(shift);
        const auto half = UINT64_C(1) << (bits - 1);
        if (rounding == Rounding::HalfToEven) {
            const auto quotient = magnitude >> bits;
            const auto remainder = magnitude & ((UINT64_C(1) << bits) - 1);
            const bool increment = remainder > half ||
                (remainder == half && (quotient & UINT64_C(1)) != 0);
            rounded_magnitude = quotient + static_cast<std::uint64_t>(increment);
        } else {
            rounded_magnitude = (magnitude + half) >> bits;
        }
    }
    return product < 0 ? -static_cast<std::int64_t>(rounded_magnitude) :
                         static_cast<std::int64_t>(rounded_magnitude);
}

std::int8_t requantize_unchecked(std::int32_t accumulator, const Requant& quant,
                                 std::uint64_t channel) {
    if ((!quant.multipliers.empty() && channel >= quant.multipliers.size()) ||
        (!quant.shifts.empty() && channel >= quant.shifts.size()))
        throw std::out_of_range("requantization channel is outside parameter vectors");
    const auto index = static_cast<std::size_t>(channel);
    const auto multiplier = quant.multipliers.empty() ? quant.multiplier : quant.multipliers[index];
    const auto shift = quant.shifts.empty() ? quant.shift : quant.shifts[index];
    std::int64_t scaled = 0;
    switch (quant.rescale) {
    case RescaleMode::Native:
        scaled = native_scale(accumulator, multiplier, shift, quant.rounding);
        break;
    case RescaleMode::TosaSingle:
        scaled = scale32(accumulator, multiplier, shift, false);
        break;
    case RescaleMode::TosaDouble:
        scaled = scale32(accumulator, multiplier, shift, true);
        break;
    }
    scaled += quant.output_zero;
    if (quant.relu) scaled = std::max(scaled, static_cast<std::int64_t>(quant.output_zero));
    return static_cast<std::int8_t>(std::clamp<std::int64_t>(scaled, -128, 127));
}

std::uint64_t tile_host_bytes(const Config& config) {
    auto bytes = checked_mul(tile_record_bytes, tile_count(config));
    if (!config.sim.trace_path.empty())
        bytes = checked_add(bytes, checked_mul(trace_panel_bytes,
                                               checked_mul(tile_count(config),
                                                           reduction_blocks(config))));
    return bytes;
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

std::uint64_t total_macs(const ShapeConfig& shape) {
    return checked_mul(checked_mul(checked_mul(shape.m, shape.n), shape.k), shape.groups);
}

std::uint64_t tile_count(const Config& config) {
    const auto& shape = config.shape;
    const auto& map = config.mapping;
    validate_shape(config);
    require(map.tile_m >= 1 && map.tile_n >= 1, "tile dimensions must be positive");
    return checked_mul(shape.groups, checked_mul(ceil_div(shape.m, map.tile_m),
                                                 ceil_div(shape.n, map.tile_n)));
}

std::uint64_t dma_transfer_cycles(const Config& config, std::uint64_t bytes) {
    const auto& memory = config.arch.memory;
    if (!bytes) return 0;
    if (!memory.burst_bytes)
        return checked_add(memory.latency, ceil_div(bytes, memory.bytes_per_cycle));
    const auto full = bytes / memory.burst_bytes;
    const auto tail = bytes % memory.burst_bytes;
    auto cycles = checked_mul(full, checked_add(memory.latency,
        ceil_div(memory.burst_bytes, memory.bytes_per_cycle)));
    if (tail) cycles = checked_add(cycles, checked_add(memory.latency,
        ceil_div(tail, memory.bytes_per_cycle)));
    return cycles;
}

std::uint64_t scratchpad_bytes(const Config& config) {
    const auto& map = config.mapping;
    validate_shape(config);
    validate_mapping(config);
    if (!dimension(map.tile_m) || !dimension(map.tile_n) || map.buffer_slots < 1 ||
        map.buffer_slots > 64)
        fail("scratchpad tile dimensions must be in " + dimension_range() +
             " and slots in 1..64");
    const auto rows = std::min(map.tile_m, config.shape.m);
    const auto cols = std::min(map.tile_n, config.shape.n);
    const auto depth = std::min(config.shape.k, panel_size(config));
    // A slot holds the tile's bias and channel metadata for its lifetime, its
    // panel buffers, each sized for the deepest panel, and INT32 partial sums
    // plus INT8 outputs.
    const auto panel = checked_add(checked_mul(rows, depth),
                                   encoded_weight_bound(config, checked_mul(depth, cols)));
    const auto metadata = checked_add(checked_mul(4, cols), channel_metadata_bytes(config, cols));
    const auto per_slot = checked_add(
        checked_add(metadata, checked_mul(slot_panel_buffers(config), panel)),
        checked_mul(5, checked_mul(rows, cols)));
    return checked_add(checked_mul(per_slot, map.buffer_slots), config.arch.weight_cache_bytes);
}

std::uint64_t slot_panel_buffers(const Config& config) {
    return std::min(config.mapping.panel_buffers, reduction_blocks(config));
}

void validate(const Config& config) {
    const auto& shape = config.shape;
    const auto& map = config.mapping;
    const auto& arch = config.arch;
    validate_shape(config);
    const auto output_channels = shape.groups * shape.n;
    validate_quant_channels(config.quant, output_channels);
    validate_mapping(config);
    if (config.weights.bits == 4) {
        require(config.quant.weight_zero >= -8 && config.quant.weight_zero <= 7,
                "INT4 weight zero point must be in -8..7");
        for (const auto value : config.quant.weight_zero_points)
            require(value >= -8 && value <= 7,
                    "INT4 per-channel weight zero points must be in -8..7");
    }
    if (!dimension(map.tile_m) || !dimension(map.tile_n))
        fail("tile dimensions must be in " + dimension_range());
    require(map.buffer_slots >= 1 && map.buffer_slots <= 64, "buffer slots must be in 1..64");
    require(arch.compute.macs_per_cycle >= 1 && arch.compute.macs_per_cycle <= max_timing_value &&
                arch.memory.bytes_per_cycle >= 1 && arch.memory.bytes_per_cycle <= max_timing_value,
            "compute and memory rates must be in 1..1000000000");
    require(arch.memory.latency <= max_timing_value &&
                arch.compute.pipeline_cycles <= max_timing_value,
            "latency and pipeline cycles must be at most 1000000000");
    require(arch.memory.burst_bytes <= max_timing_value &&
                arch.weight_cache_bytes <= max_timing_value,
            "DMA burst and weight cache sizes must be at most 1000000000 bytes");
    require(std::isfinite(arch.frequency_mhz) && arch.frequency_mhz >= 1e-6 &&
                arch.frequency_mhz <= static_cast<double>(max_timing_value),
            "frequency MHz must be finite and in 0.000001..1000000000 (at least 1 Hz)");
    // Compressed panel sizes and decode cycles come from the weight values.
    require(!config.sim.timing_only || config.weights.encoding == WeightEncoding::Dense,
            "--timing-only supports dense weights only; palette and sparse sizes depend on "
            "the weight values");
    require(arch.sram_bytes >= scratchpad_bytes(config),
            "tile buffers exceed configured SRAM capacity");

    // Host storage before any allocation: tile records and the optional stage
    // trace always; for a bit-exact run also the inputs, both reference and
    // simulated outputs, INT64 tile scratch and encoded panels. Accelerator
    // SRAM and simulator host storage are separate constraints.
    const auto tile_rows = std::min(shape.m, map.tile_m);
    const auto tile_cols = std::min(shape.n, map.tile_n);
    auto host_bytes = checked_add(tile_host_bytes(config),
                                  channel_metadata_bytes(config, output_channels));
    if (!config.sim.timing_only) {
        const auto outputs = checked_mul(shape.m, output_channels);
        host_bytes = checked_add(host_bytes, checked_mul(shape.m, shape.groups * shape.k));
        host_bytes = checked_add(host_bytes, checked_mul(shape.k, output_channels));
        host_bytes = checked_add(host_bytes, checked_mul(4, output_channels));
        host_bytes = checked_add(host_bytes, checked_mul(10, outputs));
        host_bytes = checked_add(host_bytes,
                                 checked_mul(8, checked_mul(tile_rows, tile_cols)));
        if (config.weights.encoding != WeightEncoding::Dense) {
            const auto column_panels = checked_mul(shape.groups, ceil_div(shape.n, map.tile_n));
            const auto panels = checked_mul(column_panels, reduction_blocks(config));
            const auto panel_values = checked_mul(tile_cols, std::min(shape.k, panel_size(config)));
            host_bytes = checked_add(host_bytes, checked_mul(panels,
                checked_add(128, encoded_weight_bound(config, panel_values))));
        }
    }
    if (host_bytes > max_host_bytes)
        fail("estimated simulator working storage exceeds " +
             std::to_string(max_host_bytes >> 20) + " MiB");

    // The serial schedule bounds every overlapped schedule. Four tile shape
    // classes per group give the exact serial sum without iterating over huge
    // tile lists.
    std::uint64_t serial_cycles = 0;
    const std::uint64_t row_sizes[] = {map.tile_m, shape.m % map.tile_m};
    const std::uint64_t row_counts[] = {shape.m / map.tile_m, shape.m % map.tile_m ? 1ULL : 0ULL};
    const std::uint64_t col_sizes[] = {map.tile_n, shape.n % map.tile_n};
    const std::uint64_t col_counts[] = {shape.n / map.tile_n, shape.n % map.tile_n ? 1ULL : 0ULL};
    for (unsigned r = 0; r < 2; ++r) {
        for (unsigned c = 0; c < 2; ++c) {
            const auto count = checked_mul(shape.groups, checked_mul(row_counts[r], col_counts[c]));
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
    require(config.sim.timing != TimingMode::Cycle || serial_cycles <= 10000000,
            "cycle mode serial work must be at most 10000000 cycles; use event mode for larger runs");
}

Problem make_problem(const Config& config) {
    validate(config);
    require(!config.sim.timing_only, "--timing-only runs generate no tensors");
    const auto& shape = config.shape;
    Problem problem;
    problem.m = shape.m;
    problem.n = shape.n;
    problem.k = shape.k;
    problem.groups = shape.groups;
    problem.activations.resize(static_cast<std::size_t>(shape.m * shape.groups * shape.k));
    problem.weights.resize(static_cast<std::size_t>(shape.groups * shape.k * shape.n));
    problem.bias.resize(static_cast<std::size_t>(shape.groups * shape.n));
    auto state = static_cast<std::uint32_t>(shape.seed);
    const auto next = [&state]() {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        return state;
    };
    for (auto& value : problem.activations)
        value = static_cast<std::int8_t>(static_cast<int>(next() % 31) - 15);
    for (auto& value : problem.weights)
        value = config.weights.bits == 4 ?
            static_cast<std::int8_t>(static_cast<int>(next() % 16) - 8) :
            static_cast<std::int8_t>(static_cast<int>(next() % 31) - 15);
    for (auto& value : problem.bias)
        value = static_cast<std::int32_t>(next() % 127) - 63;
    return problem;
}

std::vector<Tile> make_tiles(const Config& config) {
    validate(config);
    const auto& shape = config.shape;
    const auto& map = config.mapping;
    std::vector<Tile> tiles;
    tiles.reserve(static_cast<std::size_t>(tile_count(config)));
    // Group-major, then row, then column: a tile never spans two groups.
    for (std::uint64_t group = 0; group < shape.groups; ++group) {
        for (std::uint64_t row = 0; row < shape.m; row += map.tile_m) {
            for (std::uint64_t col = 0; col < shape.n; col += map.tile_n) {
                tiles.push_back({static_cast<std::uint64_t>(tiles.size()), row, col,
                                 std::min(map.tile_m, shape.m - row),
                                 std::min(map.tile_n, shape.n - col), group});
            }
        }
    }
    return tiles;
}

std::int32_t apply_scale_32(std::int32_t value, std::int32_t multiplier, std::int32_t shift,
                            bool double_round) {
    require(multiplier >= 0, "apply_scale_32 multiplier must be nonnegative");
    require(shift >= 2 && shift <= 62, "apply_scale_32 shift must be in 2..62");
    return scale32(value, multiplier, shift, double_round);
}

std::int8_t requantize(std::int32_t accumulator, const Requant& quant) {
    return requantize(accumulator, quant, 0);
}

std::int8_t requantize(std::int32_t accumulator, const Requant& quant,
                       std::uint64_t channel) {
    validate_quant_scalars(quant);
    validate_channel_entry(quant, static_cast<std::size_t>(channel));
    return requantize_unchecked(accumulator, quant, channel);
}

Output reference(const Problem& problem, const Requant& quant) {
    validate_problem(problem);
    const auto width = channels(problem);
    validate_quant_channels(quant, width);
    const auto row_k = problem.groups * problem.k;
    Output result;
    const auto size = static_cast<std::size_t>(problem.m * width);
    result.accumulators.resize(size);
    result.quantized.resize(size);
    // Scalar reference: each complete dot product is evaluated independently.
    for (std::uint64_t row = 0; row < problem.m; ++row) {
        for (std::uint64_t group = 0; group < problem.groups; ++group) {
            for (std::uint64_t col = 0; col < problem.n; ++col) {
                const auto channel = group * problem.n + col;
                std::int64_t accumulator = problem.bias[static_cast<std::size_t>(channel)];
                for (std::uint64_t k = 0; k < problem.k; ++k) {
                    const auto a = static_cast<std::int32_t>(problem.activations[
                        static_cast<std::size_t>(row * row_k + group * problem.k + k)]) -
                        quant.input_zero;
                    const auto b = static_cast<std::int32_t>(problem.weights[
                        static_cast<std::size_t>((group * problem.k + k) * problem.n + col)]) -
                        weight_zero(quant, channel);
                    accumulator += static_cast<std::int64_t>(a) * b;
                }
                const auto index = static_cast<std::size_t>(row * width + channel);
                result.accumulators[index] = checked_accumulator(accumulator);
                result.quantized[index] =
                    requantize_unchecked(result.accumulators[index], quant, channel);
            }
        }
    }
    return result;
}

void compute_tile(const Problem& problem, const Requant& quant,
                  const Tile& tile, Output& output) {
    validate_problem(problem);
    validate_tile(tile, problem.m, problem.n, problem.groups);
    const auto width = channels(problem);
    // Global output channel of the tile's first column.
    const auto base = tile.group * problem.n + tile.col;
    validate_quant_range(quant, width, base, tile.cols);
    const auto output_size = static_cast<std::size_t>(problem.m * width);
    if (output.accumulators.empty() && output.quantized.empty()) {
        output.accumulators.resize(output_size);
        output.quantized.resize(output_size);
    }
    require(output.accumulators.size() == output_size &&
                output.quantized.size() == output_size,
            "output vectors do not match M*G*N");
    const auto row_k = problem.groups * problem.k;
    std::vector<std::int64_t> sums(static_cast<std::size_t>(tile.rows * tile.cols));
    for (std::uint64_t row = 0; row < tile.rows; ++row)
        for (std::uint64_t col = 0; col < tile.cols; ++col)
            sums[static_cast<std::size_t>(row * tile.cols + col)] =
                problem.bias[static_cast<std::size_t>(base + col)];
    // Tiled datapath uses K-major outer products, independent of the reference
    // dot-product loop ordering, and only commits final INT32-safe values.
    for (std::uint64_t k = 0; k < problem.k; ++k) {
        const auto weight_row = (tile.group * problem.k + k) * problem.n + tile.col;
        for (std::uint64_t row = 0; row < tile.rows; ++row) {
            const auto a = static_cast<std::int32_t>(problem.activations[static_cast<std::size_t>(
                (tile.row + row) * row_k + tile.group * problem.k + k)]) - quant.input_zero;
            for (std::uint64_t col = 0; col < tile.cols; ++col) {
                const auto b = static_cast<std::int32_t>(problem.weights[
                    static_cast<std::size_t>(weight_row + col)]) - weight_zero(quant, base + col);
                sums[static_cast<std::size_t>(row * tile.cols + col)] +=
                    static_cast<std::int64_t>(a) * b;
            }
        }
    }
    for (std::uint64_t row = 0; row < tile.rows; ++row) {
        for (std::uint64_t col = 0; col < tile.cols; ++col) {
            const auto accumulator = checked_accumulator(
                sums[static_cast<std::size_t>(row * tile.cols + col)]);
            const auto index = static_cast<std::size_t>((tile.row + row) * width + base + col);
            output.accumulators[index] = accumulator;
            output.quantized[index] = requantize_unchecked(accumulator, quant, base + col);
        }
    }
}


namespace {
void pack_value(std::vector<std::uint8_t>& bytes, std::size_t offset,
                std::uint64_t index, std::uint64_t bits, std::uint8_t value) {
    const auto bit = checked_mul(index, bits);
    const auto at = offset + static_cast<std::size_t>(bit / 8);
    bytes.at(at) |= static_cast<std::uint8_t>((value & ((UINT64_C(1) << bits) - 1)) << (bit % 8));
}

std::uint8_t unpack_value(const std::vector<std::uint8_t>& bytes, std::size_t offset,
                          std::uint64_t index, std::uint64_t bits) {
    const auto bit = checked_mul(index, bits);
    return static_cast<std::uint8_t>((bytes.at(offset + static_cast<std::size_t>(bit / 8)) >>
                                      (bit % 8)) & ((UINT64_C(1) << bits) - 1));
}

std::int8_t signed_packed(std::uint8_t value, std::uint64_t bits) {
    const auto sign = UINT64_C(1) << (bits - 1);
    const auto decoded = (value & sign) ? static_cast<int>(value) - static_cast<int>(UINT64_C(1) << bits) :
                                        static_cast<int>(value);
    return static_cast<std::int8_t>(decoded);
}

template<class Consumer>
void visit_encoded(const Config& config, const EncodedWeightPanel& panel, Consumer consume) {
    const auto& format = config.weights;
    const auto count = checked_mul(panel.k, panel.cols);
    require(panel.buffer.size() >= 4, "encoded weight header is truncated");
    require(panel.buffer[3] == 0, "encoded weight reserved header must be zero");
    const auto field = static_cast<std::uint64_t>(panel.buffer[0]) |
                       (static_cast<std::uint64_t>(panel.buffer[1]) << 8);
    if (format.encoding == WeightEncoding::Palette) {
        require(panel.buffer[2] == format.palette_bits && field >= 1 &&
                    field <= (UINT64_C(1) << format.palette_bits), "invalid palette header");
        const auto offset = static_cast<std::size_t>(4 + field);
        require(panel.buffer.size() ==
                    checked_add(offset, ceil_div(checked_mul(count, format.palette_bits), 8)),
                "encoded palette length is invalid");
        for (std::uint64_t index = 0; index < count; ++index) {
            const auto entry = unpack_value(panel.buffer, offset, index, format.palette_bits);
            require(entry < field, "palette index is outside LUT");
            consume(index, signed_packed(panel.buffer.at(static_cast<std::size_t>(4 + entry)), 8));
        }
    } else {
        require(format.encoding == WeightEncoding::Sparse && field == format.sparse_block &&
                    panel.buffer[2] == format.bits, "invalid sparse weight header");
        const auto blocks = ceil_div(count, format.sparse_block);
        const auto offset = static_cast<std::size_t>(4 + ceil_div(blocks, 8));
        require(panel.buffer.size() == checked_add(offset,
                    ceil_div(checked_mul(panel.retained_values, format.bits), 8)),
                "encoded sparse length is invalid");
        std::uint64_t stored = 0;
        for (std::uint64_t block = 0; block < blocks; ++block) {
            if (!(panel.buffer.at(4 + static_cast<std::size_t>(block / 8)) & (1U << (block % 8)))) continue;
            const auto start = checked_mul(block, format.sparse_block);
            const auto end = std::min(count, checked_add(start, format.sparse_block));
            for (auto index = start; index < end; ++index) {
                require(stored < panel.retained_values, "sparse bitmap exceeds stored payload");
                consume(index, signed_packed(
                    unpack_value(panel.buffer, offset, stored++, format.bits), format.bits));
            }
        }
        require(stored == panel.retained_values, "sparse payload exceeds bitmap");
    }
}

void require_problem_matches(const Config& config, const Problem& problem) {
    const auto& shape = config.shape;
    require(shape.m == problem.m && shape.n == problem.n && shape.k == problem.k &&
                shape.groups == problem.groups,
            "problem shape differs from accelerator configuration");
}
}  // namespace

std::vector<EncodedWeightPanel> encode_weights(const Config& config, const Problem& problem) {
    validate(config);
    validate_problem(problem);
    require_problem_matches(config, problem);
    const auto& shape = config.shape;
    const auto& format = config.weights;
    const auto rate = config.arch.compute.weight_decode_values_per_cycle;
    std::vector<EncodedWeightPanel> panels;
    if (format.encoding == WeightEncoding::Dense) return panels;
    const auto blocks = reduction_blocks(config);
    const auto ksize = panel_size(config);
    const auto tile_n = config.mapping.tile_n;
    panels.reserve(static_cast<std::size_t>(
        checked_mul(shape.groups, checked_mul(ceil_div(shape.n, tile_n), blocks))));
    for (std::uint64_t group = 0; group < shape.groups; ++group) {
        for (std::uint64_t col = 0; col < shape.n; col += tile_n) {
            for (std::uint64_t block = 0; block < blocks; ++block) {
                EncodedWeightPanel panel;
                panel.group = group;
                panel.col = col;
                panel.cols = std::min(tile_n, shape.n - col);
                panel.k_start = checked_mul(block, ksize);
                panel.k = panel_k(config, block);
                const auto count = checked_mul(panel.k, panel.cols);
                const auto channel = group * shape.n + col;
                const auto weight = [&](std::uint64_t index) {
                    return problem.weights.at(static_cast<std::size_t>(
                        (group * shape.k + panel.k_start + index / panel.cols) * shape.n + col +
                        index % panel.cols));
                };
                for (std::uint64_t index = 0; index < count; ++index) {
                    const auto value = weight(index);
                    require(format.bits != 4 || (value >= -8 && value <= 7),
                            "weight value is outside signed INT4 precision");
                    if (value != weight_zero(config.quant, channel + index % panel.cols))
                        ++panel.nonzero_values;
                }
                if (format.encoding == WeightEncoding::Palette) {
                    std::array<int, 256> indices;
                    indices.fill(-1);
                    std::vector<std::uint8_t> lut;
                    for (std::uint64_t index = 0; index < count; ++index) {
                        const auto value = static_cast<std::uint8_t>(weight(index));
                        if (indices[value] != -1) continue;
                        require(lut.size() < (UINT64_C(1) << format.palette_bits),
                                "lossless palette capacity exceeded; increase --palette-bits "
                                "or use dense");
                        indices[value] = static_cast<int>(lut.size());
                        lut.push_back(value);
                    }
                    panel.buffer.assign(static_cast<std::size_t>(checked_add(4 + lut.size(),
                        ceil_div(checked_mul(count, format.palette_bits), 8))), 0);
                    panel.buffer[0] = static_cast<std::uint8_t>(lut.size() & 255U);
                    panel.buffer[1] = static_cast<std::uint8_t>(lut.size() >> 8);
                    panel.buffer[2] = static_cast<std::uint8_t>(format.palette_bits);
                    std::copy(lut.begin(), lut.end(), panel.buffer.begin() + 4);
                    const auto offset = 4 + lut.size();
                    for (std::uint64_t index = 0; index < count; ++index) {
                        const auto entry = indices[static_cast<std::uint8_t>(weight(index))];
                        pack_value(panel.buffer, offset, index, format.palette_bits,
                                   static_cast<std::uint8_t>(entry));
                    }
                    panel.retained_values = count;
                    panel.decode_cycles = ceil_div(count, rate);
                } else {
                    const auto runs = ceil_div(count, format.sparse_block);
                    const auto offset = static_cast<std::size_t>(4 + ceil_div(runs, 8));
                    panel.buffer.assign(offset, 0);
                    panel.buffer[0] = static_cast<std::uint8_t>(format.sparse_block & 255U);
                    panel.buffer[1] = static_cast<std::uint8_t>(format.sparse_block >> 8);
                    panel.buffer[2] = static_cast<std::uint8_t>(format.bits);
                    for (std::uint64_t run = 0; run < runs; ++run) {
                        const auto start = checked_mul(run, format.sparse_block);
                        const auto end = std::min(count, checked_add(start, format.sparse_block));
                        bool keep = false;
                        for (auto index = start; index < end; ++index) {
                            const auto lane = channel + index % panel.cols;
                            keep = keep || weight(index) != weight_zero(config.quant, lane);
                        }
                        if (!keep) continue;
                        panel.buffer.at(4 + static_cast<std::size_t>(run / 8)) |=
                            static_cast<std::uint8_t>(1U << (run % 8));
                        const auto old = panel.retained_values;
                        panel.retained_values = checked_add(panel.retained_values, end - start);
                        panel.buffer.resize(offset + static_cast<std::size_t>(
                            ceil_div(checked_mul(panel.retained_values, format.bits), 8)), 0);
                        for (auto index = start; index < end; ++index)
                            pack_value(panel.buffer, offset, old + index - start, format.bits,
                                       static_cast<std::uint8_t>(weight(index)));
                    }
                    panel.decode_cycles = checked_add(ceil_div(runs, rate),
                                                      ceil_div(panel.retained_values, rate));
                }
                panels.push_back(std::move(panel));
            }
        }
    }
    return panels;
}

const EncodedWeightPanel& weight_panel(const Config& config,
    const std::vector<EncodedWeightPanel>& panels, const Tile& tile, std::uint64_t block) {
    const auto tile_n = config.mapping.tile_n;
    require(tile_n >= 1, "encoded column panel width must be positive");
    require(tile.col % tile_n == 0, "encoded tile must align with output column panels");
    const auto column_panels = ceil_div(config.shape.n, tile_n);
    const auto& panel = panels.at(static_cast<std::size_t>(checked_add(checked_mul(
        checked_add(checked_mul(tile.group, column_panels), tile.col / tile_n),
        reduction_blocks(config)), block)));
    require(panel.group == tile.group && panel.col == tile.col && panel.cols == tile.cols &&
                panel.k == panel_k(config, block) &&
                panel.k_start == checked_mul(block, panel_size(config)),
            "encoded weight panel does not match tile");
    return panel;
}

void compute_tile_encoded(const Problem& problem, const Config& config,
    const std::vector<EncodedWeightPanel>& panels, const Tile& tile, Output& output) {
    // Per-tile checks only: validate(config) ran when the panels were encoded,
    // and repeating its O(G*N) channel scan per tile would be quadratic.
    validate_shape(config);
    validate_mapping(config);
    validate_problem(problem);
    require_problem_matches(config, problem);
    validate_tile(tile, problem.m, problem.n, problem.groups);
    const auto width = channels(problem);
    const auto base = tile.group * problem.n + tile.col;
    validate_quant_range(config.quant, width, base, tile.cols);
    require(config.weights.encoding != WeightEncoding::Dense,
            "encoded compute requires compressed weights");
    const auto size = static_cast<std::size_t>(checked_mul(problem.m, width));
    if (output.accumulators.empty() && output.quantized.empty()) {
        output.accumulators.resize(size);
        output.quantized.resize(size);
    }
    require(output.accumulators.size() == size && output.quantized.size() == size,
            "encoded output vectors do not match M*G*N");
    const auto row_k = problem.groups * problem.k;
    std::vector<std::int64_t> sums(static_cast<std::size_t>(checked_mul(tile.rows, tile.cols)));
    for (std::uint64_t row = 0; row < tile.rows; ++row)
        for (std::uint64_t col = 0; col < tile.cols; ++col)
            sums.at(static_cast<std::size_t>(row * tile.cols + col)) =
                problem.bias.at(static_cast<std::size_t>(base + col));
    for (std::uint64_t block = 0; block < reduction_blocks(config); ++block) {
        const auto& panel = weight_panel(config, panels, tile, block);
        visit_encoded(config, panel, [&](std::uint64_t index, std::int8_t raw) {
            const auto col = index % tile.cols;
            const auto k = panel.k_start + index / tile.cols;
            const auto w = static_cast<std::int32_t>(raw) - weight_zero(config.quant, base + col);
            if (config.weights.encoding == WeightEncoding::Sparse && w == 0) return;
            const auto column = tile.group * problem.k + k;
            for (std::uint64_t row = 0; row < tile.rows; ++row) {
                const auto a = static_cast<std::int32_t>(problem.activations.at(
                    static_cast<std::size_t>((tile.row + row) * row_k + column))) -
                    config.quant.input_zero;
                sums.at(static_cast<std::size_t>(row * tile.cols + col)) += static_cast<std::int64_t>(a) * w;
            }
        });
    }
    for (std::uint64_t row = 0; row < tile.rows; ++row)
        for (std::uint64_t col = 0; col < tile.cols; ++col) {
            const auto value = checked_accumulator(sums.at(static_cast<std::size_t>(row * tile.cols + col)));
            const auto index = static_cast<std::size_t>((tile.row + row) * width + base + col);
            output.accumulators.at(index) = value;
            output.quantized.at(index) = requantize_unchecked(value, config.quant, base + col);
        }
}

std::uint64_t encoded_panel_compute_cycles(const Config& config, const Tile& tile,
    std::uint64_t block, const EncodedWeightPanel& panel) {
    const auto& compute = config.arch.compute;
    if (config.weights.encoding == WeightEncoding::Sparse)
        return checked_add(checked_add(ceil_div(checked_mul(tile.rows, panel.nonzero_values),
                                                compute.macs_per_cycle),
                                       compute.pipeline_cycles),
                           panel.decode_cycles);
    // Remove the conservative decoder bound and add this panel's exact cost.
    return checked_add(panel_compute_cycles(config, tile, block) -
                           ceil_div(checked_mul(panel.k, panel.cols),
                                    compute.weight_decode_values_per_cycle),
                       panel.decode_cycles);
}

std::uint64_t reduction_blocks(const Config& config) {
    validate_shape(config);
    validate_mapping(config);
    return ceil_div(config.shape.k, panel_size(config));
}

std::uint64_t panel_k(const Config& config, std::uint64_t block) {
    const auto blocks = reduction_blocks(config);
    require(block < blocks, "reduction block is outside K");
    const auto size = panel_size(config);
    return std::min(size, config.shape.k - checked_mul(block, size));
}

std::uint64_t panel_read_bytes(const Config& config, const Tile& tile,
                               std::uint64_t block) {
    validate_shape(config);
    validate_tile(tile, config.shape);
    validate_mapping(config);
    return panel_bytes(config, panel_k(config, block), tile.rows, tile.cols, block == 0);
}

std::uint64_t panel_compute_cycles(const Config& config, const Tile& tile,
                                   std::uint64_t block) {
    validate_shape(config);
    validate_tile(tile, config.shape);
    validate_mapping(config);
    validate_compute_rates(config);
    return compute_panel_cycles(config, tile.rows, tile.cols, panel_k(config, block));
}

std::uint64_t peak_macs_per_cycle(const Config& config) {
    validate_mapping(config);
    const auto& compute = config.arch.compute;
    return compute.array_rows ? checked_mul(compute.array_rows, compute.array_cols) :
                                compute.macs_per_cycle;
}

std::uint64_t tile_read_bytes(const Config& config, const Tile& tile) {
    validate_shape(config);
    validate_tile(tile, config.shape);
    validate_mapping(config);
    return all_read_bytes(config, tile.rows, tile.cols);
}

std::uint64_t tile_write_bytes(const Tile& tile) {
    if (!dimension(tile.rows) || !dimension(tile.cols))
        fail("tile extents must be in " + dimension_range());
    return checked_mul(tile.rows, tile.cols);
}

std::uint64_t tile_compute_cycles(const Config& config, const Tile& tile) {
    validate_shape(config);
    validate_compute_rates(config);
    validate_tile(tile, config.shape);
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
