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

// Every closed set is an enum class. Strings exist only at the CLI, config-file
// and JSON edges; EnumNames<E>::names is indexed by the enumerator value and is
// the single spelling used for parsing and printing.
enum class Rounding : std::uint8_t { HalfAwayFromZero, HalfToEven };
enum class RescaleMode : std::uint8_t { Native, TosaSingle, TosaDouble };
enum class WeightEncoding : std::uint8_t { Dense, Palette, Sparse };
enum class TimingMode : std::uint8_t { Event, Cycle };
enum class Dataflow : std::uint8_t { OutputStationary, WeightStationary };
enum class MemoryModel : std::uint8_t { Serial, Axi };
enum class Partition : std::uint8_t { M, N, Auto };

template <class E> struct EnumNames;
template <> struct EnumNames<Rounding> {
    static constexpr std::array<std::string_view, 2> names{"away", "even"};
};
template <> struct EnumNames<RescaleMode> {
    static constexpr std::array<std::string_view, 3> names{"native", "tosa-single", "tosa-double"};
};
template <> struct EnumNames<WeightEncoding> {
    static constexpr std::array<std::string_view, 3> names{"dense", "palette", "sparse"};
};
template <> struct EnumNames<TimingMode> {
    static constexpr std::array<std::string_view, 2> names{"event", "cycle"};
};
template <> struct EnumNames<Dataflow> {
    static constexpr std::array<std::string_view, 2> names{"os", "ws"};
};
template <> struct EnumNames<MemoryModel> {
    static constexpr std::array<std::string_view, 2> names{"serial", "axi"};
};
template <> struct EnumNames<Partition> {
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

struct ShapeConfig {
    std::uint64_t m = 16, n = 16, k = 32;
    // Synthetic tensors use the low 32 bits.
    std::uint64_t seed = 1;
};

struct ComputeConfig {
    std::uint64_t macs_per_cycle = 64;
    // Both zero selects the vector datapath; both nonzero an output-stationary array.
    std::uint64_t array_rows = 0, array_cols = 0;
    std::uint64_t pipeline_cycles = 4;
    std::uint64_t weight_decode_values_per_cycle = 16;
};

struct MemoryConfig {
    std::uint64_t bytes_per_cycle = 16;
    std::uint64_t latency = 12;
    // Zero issues each transfer as one burst.
    std::uint64_t burst_bytes = 0;
};

struct ArchConfig {
    ComputeConfig compute;
    MemoryConfig memory;
    std::uint64_t sram_bytes = 262144;
    std::uint64_t weight_cache_bytes = 0;
    double frequency_mhz = 200.0;
};

struct MappingConfig {
    std::uint64_t tile_m = 8, tile_n = 8;
    // Zero means one reduction panel spanning K.
    std::uint64_t tile_k = 0;
    std::uint64_t buffer_slots = 2;
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
    // Per-channel overrides; each is empty or holds one entry per output channel.
    std::vector<std::int32_t> weight_zero_points;
    std::vector<std::int32_t> multipliers;
    std::vector<std::int32_t> shifts;
};

struct SimOptions {
    TimingMode timing = TimingMode::Event;
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
