#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace npu {

// EnumNames<E>::names, indexed by enumerator value, is the one spelling of
// each enum that CLI flags, config files and JSON results parse and print.
enum class Rounding : std::uint8_t { HalfAwayFromZero, HalfToEven };
enum class RescaleMode : std::uint8_t { Native, TosaSingle, TosaDouble };
enum class WeightEncoding : std::uint8_t { Dense, Palette, Sparse };
enum class TimingMode : std::uint8_t { Event, Cycle };
enum class Dataflow : std::uint8_t { OutputStationary, WeightStationary };
enum class MemoryModel : std::uint8_t { Serial, Axi };
enum class Partition : std::uint8_t { M, N, Auto };

template <class E>
struct EnumNames;
template <>
struct EnumNames<Rounding> {
    static constexpr std::array<std::string_view, 2> names{"away", "even"};
};
template <>
struct EnumNames<RescaleMode> {
    static constexpr std::array<std::string_view, 3> names{"native", "tosa-single", "tosa-double"};
};
template <>
struct EnumNames<WeightEncoding> {
    static constexpr std::array<std::string_view, 3> names{"dense", "palette", "sparse"};
};
template <>
struct EnumNames<TimingMode> {
    static constexpr std::array<std::string_view, 2> names{"event", "cycle"};
};
template <>
struct EnumNames<Dataflow> {
    static constexpr std::array<std::string_view, 2> names{"os", "ws"};
};
template <>
struct EnumNames<MemoryModel> {
    static constexpr std::array<std::string_view, 2> names{"serial", "axi"};
};
template <>
struct EnumNames<Partition> {
    static constexpr std::array<std::string_view, 3> names{"m", "n", "auto"};
};

template <class E, class = decltype(EnumNames<E>::names)>
constexpr std::string_view to_string(E value) noexcept {
    const auto index = static_cast<std::size_t>(value);
    return index < EnumNames<E>::names.size() ? EnumNames<E>::names[index] : std::string_view{};
}

template <class E, class = decltype(EnumNames<E>::names)>
constexpr std::optional<E> from_string(std::string_view text) noexcept {
    for (std::size_t index = 0; index < EnumNames<E>::names.size(); ++index)
        if (EnumNames<E>::names[index] == text)
            return static_cast<E>(static_cast<std::underlying_type_t<E>>(index));
    return std::nullopt;
}

// "away|even": the accepted spellings, for help text and error messages.
template <class E, class = decltype(EnumNames<E>::names)>
std::string enum_choices() {
    std::string result;
    for (const auto name : EnumNames<E>::names) {
        if (!result.empty()) result += '|';
        result += name;
    }
    return result;
}

// Implicit convolution: the activation operand is the im2col view of an NHWC
// INT8 input tensor of input_h x input_w x input_c at the activation base
// address. Output pixel (oy, ox) is GEMM row oy * W_o + ox; a group's K index
// is (ky * kernel_w + kx) * C_g + ci with C_g = input_c / groups, so a kernel
// row is kernel_w * C_g consecutive K. Pixels outside the input read the input
// zero point. All three input extents zero is a plain GEMM operand.
struct ConvConfig {
    std::uint64_t input_h = 0, input_w = 0, input_c = 0;
    std::uint64_t kernel_h = 1, kernel_w = 1;
    std::uint64_t stride_h = 1, stride_w = 1;
    std::uint64_t dilation_h = 1, dilation_w = 1;
    std::uint64_t pad_top = 0, pad_left = 0, pad_bottom = 0, pad_right = 0;
};

// G independent GEMMs share M. Activations are M x (G*K) row-major, weights are
// G blocks of K x N, bias and outputs hold G*N channels. Depthwise convolution
// is G channels with N = 1 and K = kh*kw.
struct ShapeConfig {
    std::uint64_t m = 16, n = 16, k = 32;
    std::uint64_t groups = 1;
    // Synthetic tensors use the low 32 bits.
    std::uint64_t seed = 1;
    ConvConfig conv{};
};

struct ComputeConfig {
    std::uint64_t macs_per_cycle = 64;
    // Both zero selects the vector datapath; both nonzero a systolic array
    // running `dataflow`. Weight-stationary needs the array.
    std::uint64_t array_rows = 0, array_cols = 0;
    Dataflow dataflow = Dataflow::OutputStationary;
    std::uint64_t pipeline_cycles = 4;
    std::uint64_t weight_decode_values_per_cycle = 16;
    // Channel-wise datapath for depthwise problems (G > 1, N = 1): its tiles
    // span tile_n consecutive groups. Zero runs them as grouped GEMMs.
    std::uint64_t depthwise_macs_per_cycle = 0;
};

// AXI4 port of the axi memory model: independent read (AR/R) and write
// (AW/W/B) channels moving one data_bytes beat per cycle each.
struct AxiConfig {
    std::uint64_t data_bytes = 16;
    // Longest INCR burst; bursts also stop at 4 KiB boundaries.
    std::uint64_t max_burst_beats = 16;
    std::uint64_t read_outstanding = 4, write_outstanding = 4;
    // AR issue to the first R beat; last W beat to the B response.
    std::uint64_t read_latency = 12, write_latency = 12;
};

struct MemoryConfig {
    // Serial model: one DMA server for reads and writes.
    std::uint64_t bytes_per_cycle = 16;
    std::uint64_t latency = 12;
    // Zero issues each transfer as one burst.
    std::uint64_t burst_bytes = 0;
    MemoryModel model = MemoryModel::Serial;
    // Axi model: DRAM data rate shared by both channels; zero is unlimited.
    std::uint64_t dram_bytes_per_cycle = 0;
    AxiConfig axi;
};

struct ArchConfig {
    ComputeConfig compute;
    MemoryConfig memory;
    std::uint64_t sram_bytes = 262144;
    std::uint64_t weight_cache_bytes = 0;
    // LRU of activation panels keyed by (group or channel tile, row tile,
    // reduction block): column tiles of one row tile reuse its panels.
    std::uint64_t activation_cache_bytes = 0;
    double frequency_mhz = 200.0;
    // Core pipelines; each owns its slots, SRAM budget, weight cache and
    // memory port, and all share one DRAM timeline (axi model only).
    std::uint64_t cores = 1;
};

struct MappingConfig {
    std::uint64_t tile_m = 8, tile_n = 8;
    // Zero means one reduction panel spanning K.
    std::uint64_t tile_k = 0;
    std::uint64_t buffer_slots = 2;
    // Reduction-panel buffers per slot, 1 or 2. Panel b of a tile reuses the
    // buffer of panel b - panel_buffers, so its load waits for that panel's
    // compute to finish.
    std::uint64_t panel_buffers = 2;
    // How a cluster splits the output tiles; see ClusterPlan in core.hpp.
    Partition partition = Partition::Auto;
};

struct WeightFormat {
    std::uint64_t bits = 8;
    WeightEncoding encoding = WeightEncoding::Dense;
    std::uint64_t palette_bits = 4;
    std::uint64_t sparse_block = 8;
};

// scale = multiplier / 2^shift. Native accepts multiplier 0..INT32_MAX and
// shift 0..31 and rounds the magnitude with `rounding`; the TOSA modes accept
// shift 2..62 and round as TOSA 1.0.2 apply_scale_32.
struct Requant {
    std::int32_t input_zero = -3;
    std::int32_t weight_zero = 2;
    std::int32_t output_zero = 0;
    std::int32_t multiplier = 1;
    std::int32_t shift = 7;
    bool relu = false;
    RescaleMode rescale = RescaleMode::Native;
    Rounding rounding = Rounding::HalfAwayFromZero;
    // Per-channel overrides; each is empty or holds one entry per output
    // channel, G*N in all.
    std::vector<std::int32_t> weight_zero_points;
    std::vector<std::int32_t> multipliers;
    std::vector<std::int32_t> shifts;
};

struct SimOptions {
    TimingMode timing = TimingMode::Event;
    // Times the schedule without tensors, numerics or the reference.
    bool timing_only = false;
    // Empty writes no stage trace.
    std::string trace_path;
};

struct Config {
    std::string name, description;
    ShapeConfig shape;
    ArchConfig arch;
    MappingConfig mapping;
    WeightFormat weights;
    Requant quant;
    SimOptions sim;
};

}  // namespace npu
