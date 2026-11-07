#include "npu/options.hpp"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace npu {
namespace {

template <class> constexpr bool always_false = false;

template <class T> constexpr bool is_enum_field = std::is_enum_v<T>;

#define NPU_FIELD(member)                                       \
    [](Config& c) -> FieldRef { return &c.member; },            \
    [](const Config& c) -> ConstFieldRef { return &c.member; }

// Grouped by path prefix: the JSON writer opens each group once, in this order.
const std::vector<ConfigOption> option_table = {
    {"", "name", "", "configuration name", NPU_FIELD(name)},
    {"", "description", "", "free-text description", NPU_FIELD(description)},

    {"--m", "shape.m", "", "output rows M", NPU_FIELD(shape.m)},
    {"--n", "shape.n", "", "output channels N per group", NPU_FIELD(shape.n)},
    {"--k", "shape.k", "", "reduction depth K per group", NPU_FIELD(shape.k)},
    {"--groups", "shape.groups", "",
     "independent GEMMs G sharing M; depthwise convolution is G channels, N 1, K kh*kw",
     NPU_FIELD(shape.groups)},
    {"--seed", "shape.seed", "", "synthetic-tensor seed; the low 32 bits are used",
     NPU_FIELD(shape.seed)},

    {"--macs-per-cycle", "arch.compute.macs_per_cycle", "",
     "vector-datapath MACs per cycle; unused when an array is configured",
     NPU_FIELD(arch.compute.macs_per_cycle)},
    {"--array-rows", "arch.compute.array_rows", "",
     "systolic array rows; 0 with --array-cols 0 selects the vector datapath",
     NPU_FIELD(arch.compute.array_rows)},
    {"--array-cols", "arch.compute.array_cols", "", "systolic array columns",
     NPU_FIELD(arch.compute.array_cols)},
    {"--dataflow", "arch.compute.dataflow", "",
     "array dataflow, needs --array-rows/--array-cols: os holds an output block in the PEs "
     "and streams K; ws holds a weight block and streams the tile's rows",
     NPU_FIELD(arch.compute.dataflow)},
    {"--pipeline-cycles", "arch.compute.pipeline_cycles", "",
     "fixed cycles charged per reduction panel on top of MAC and decode time",
     NPU_FIELD(arch.compute.pipeline_cycles)},
    {"--weight-decode-values-per-cycle", "arch.compute.weight_decode_values_per_cycle", "",
     "compressed-weight decoder throughput in values per cycle",
     NPU_FIELD(arch.compute.weight_decode_values_per_cycle)},

    {"--memory-model", "arch.memory.model", "",
     "serial: one DMA server alternating reads and writes, one transfer at a time; axi: "
     "independent read and write channels with outstanding bursts and a shared DRAM rate",
     NPU_FIELD(arch.memory.model)},
    {"--memory-bytes-per-cycle", "arch.memory.bytes_per_cycle", "",
     "serial model: DMA bytes transferred per cycle", NPU_FIELD(arch.memory.bytes_per_cycle)},
    {"--memory-latency", "arch.memory.latency", "",
     "serial model: cycles charged once per DMA burst", NPU_FIELD(arch.memory.latency)},
    {"--dma-burst-bytes", "arch.memory.burst_bytes", "",
     "serial model: maximum burst size; 0 issues each transfer as one burst",
     NPU_FIELD(arch.memory.burst_bytes)},
    {"--dram-bytes-per-cycle", "arch.memory.dram_bytes_per_cycle", "",
     "axi model: DRAM data rate shared by the read and write channels; 0 is unlimited",
     NPU_FIELD(arch.memory.dram_bytes_per_cycle)},
    {"--axi-data-bytes", "arch.memory.axi.data_bytes", "8|16|32|64",
     "axi model: bytes per beat; each channel moves one beat per cycle",
     NPU_FIELD(arch.memory.axi.data_bytes)},
    {"--axi-burst-beats", "arch.memory.axi.max_burst_beats", "",
     "axi model: longest INCR burst in beats, 1..256; bursts also split at 4 KiB boundaries",
     NPU_FIELD(arch.memory.axi.max_burst_beats)},
    {"--axi-read-outstanding", "arch.memory.axi.read_outstanding", "",
     "axi model: read bursts in flight, 1..64", NPU_FIELD(arch.memory.axi.read_outstanding)},
    {"--axi-write-outstanding", "arch.memory.axi.write_outstanding", "",
     "axi model: write bursts in flight, 1..64", NPU_FIELD(arch.memory.axi.write_outstanding)},
    {"--axi-read-latency", "arch.memory.axi.read_latency", "",
     "axi model: cycles from AR issue to the first R beat",
     NPU_FIELD(arch.memory.axi.read_latency)},
    {"--axi-write-latency", "arch.memory.axi.write_latency", "",
     "axi model: cycles from the last W beat to the B response",
     NPU_FIELD(arch.memory.axi.write_latency)},

    {"--sram-bytes", "arch.sram_bytes", "",
     "SRAM capacity for tile slots plus the weight cache", NPU_FIELD(arch.sram_bytes)},
    {"--weight-cache-bytes", "arch.weight_cache_bytes", "",
     "LRU weight-panel cache capacity; 0 disables it", NPU_FIELD(arch.weight_cache_bytes)},
    {"--frequency-mhz", "arch.frequency_mhz", "X",
     "clock used to convert cycles to modeled time", NPU_FIELD(arch.frequency_mhz)},

    {"--tile-m", "mapping.tile_m", "", "output tile rows", NPU_FIELD(mapping.tile_m)},
    {"--tile-n", "mapping.tile_n", "", "output tile columns", NPU_FIELD(mapping.tile_n)},
    {"--tile-k", "mapping.tile_k", "", "reduction panel depth; 0 uses one panel of depth K",
     NPU_FIELD(mapping.tile_k)},
    {"--buffer-slots", "mapping.buffer_slots", "",
     "tiles in flight across load, compute and store", NPU_FIELD(mapping.buffer_slots)},
    {"--panel-buffers", "mapping.panel_buffers", "1|2",
     "reduction-panel buffers per slot; with 2 the DMA loads panel b+1 while compute reads "
     "panel b", NPU_FIELD(mapping.panel_buffers)},

    {"--weight-bits", "weights.bits", "4|8", "signed weight precision", NPU_FIELD(weights.bits)},
    {"--weight-encoding", "weights.encoding", "", "lossless weight storage format",
     NPU_FIELD(weights.encoding)},
    {"--palette-bits", "weights.palette_bits", "1|2|4|8", "palette index width",
     NPU_FIELD(weights.palette_bits)},
    {"--sparse-block", "weights.sparse_block", "", "zero-skipping bitmap block length",
     NPU_FIELD(weights.sparse_block)},

    {"--input-zero", "quant.input_zero", "", "INT8 activation zero point",
     NPU_FIELD(quant.input_zero)},
    {"--weight-zero", "quant.weight_zero", "", "weight zero point", NPU_FIELD(quant.weight_zero)},
    {"--output-zero", "quant.output_zero", "", "INT8 output zero point",
     NPU_FIELD(quant.output_zero)},
    {"--multiplier", "quant.multiplier", "",
     "rescale multiplier in 0..2147483647; scale = multiplier / 2^shift",
     NPU_FIELD(quant.multiplier)},
    {"--shift", "quant.shift", "", "rescale right shift: 0..31 native, 2..62 tosa modes",
     NPU_FIELD(quant.shift)},
    {"--relu", "quant.relu", "", "clamp outputs below the output zero point",
     NPU_FIELD(quant.relu)},
    {"--rescale", "quant.rescale", "",
     "native rounds the magnitude of acc*multiplier per --rounding; tosa-single and "
     "tosa-double are TOSA 1.0.2 RESCALE apply_scale_32 (TFLite single rounding and "
     "gemmlowp double rounding with tosa shift = 31 - TFLite shift)",
     NPU_FIELD(quant.rescale)},
    {"--rounding", "quant.rounding", "",
     "native rescale tie rounding: away from zero or to even", NPU_FIELD(quant.rounding)},
    {"", "quant.weight_zero_points", "",
     "per-channel weight zero points, one per output channel (G*N); --quantization-file "
     "sets all three per-channel vectors",
     NPU_FIELD(quant.weight_zero_points)},
    {"", "quant.multipliers", "", "per-channel multipliers, one per output channel (G*N)",
     NPU_FIELD(quant.multipliers)},
    {"", "quant.shifts", "", "per-channel signed shifts, one per output channel (G*N)",
     NPU_FIELD(quant.shifts)},

    {"--timing", "sim.timing", "",
     "event jumps across stage waits; cycle waits one cycle at a time", NPU_FIELD(sim.timing)},
    {"--timing-only", "sim.timing_only", "",
     "time the schedule without tensors, numerics or the reference (dense weights; up to "
     "1e13 MACs; results report numerics \"skipped\")",
     NPU_FIELD(sim.timing_only)},
    {"--trace", "sim.trace_path", "PATH", "stage trace CSV written after a batch run",
     NPU_FIELD(sim.trace_path)},
};

#undef NPU_FIELD

std::string quoted(std::string_view text) { return "'" + std::string(text) + "'"; }

bool json_number_grammar(std::string_view text) {
    std::size_t i = 0;
    const auto digits = [&]() {
        const auto start = i;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') ++i;
        return i > start;
    };
    if (i < text.size() && text[i] == '-') ++i;
    if (i < text.size() && text[i] == '0') ++i;
    else if (!digits()) return false;
    if (i < text.size() && text[i] == '.') {
        ++i;
        if (!digits()) return false;
    }
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        if (i < text.size() && (text[i] == '+' || text[i] == '-')) ++i;
        if (!digits()) return false;
    }
    return i == text.size();
}

template <class T>
std::optional<T> integer_value(std::string_view text) {
    T result{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        return std::nullopt;
    return result;
}

std::string describe(const JsonValue& value) {
    switch (value.type) {
    case JsonValue::Type::Number: return value.text;
    case JsonValue::Type::String: {
        std::ostringstream text;
        write_json_string(text, value.text);
        return text.str();
    }
    case JsonValue::Type::Boolean: return value.boolean ? "true" : "false";
    default: return std::string(json_type_name(value.type));
    }
}

[[noreturn]] void json_error(std::string_view origin, const JsonValue& at,
                             const std::string& reason) {
    throw std::invalid_argument(std::string(origin) + ':' + std::to_string(at.line) + ':' +
                                std::to_string(at.column) + ": " + reason);
}

template <class T>
T json_integer(const JsonValue& value, std::string_view origin, std::string_view path) {
    static_assert(std::is_integral_v<T>);
    if constexpr (std::is_signed_v<T>) {
        const auto parsed = value.as_int64();
        if (parsed && *parsed >= std::numeric_limits<T>::min() &&
            *parsed <= std::numeric_limits<T>::max())
            return static_cast<T>(*parsed);
        json_error(origin, value, std::string(path) + " expects a signed " +
                                      std::to_string(sizeof(T) * 8) + "-bit integer, got " +
                                      describe(value));
    } else {
        const auto parsed = value.as_uint64();
        if (parsed && *parsed <= std::numeric_limits<T>::max()) return static_cast<T>(*parsed);
        json_error(origin, value, std::string(path) + " expects an unsigned " +
                                      std::to_string(sizeof(T) * 8) + "-bit integer, got " +
                                      describe(value));
    }
}

void assign_json(const ConfigOption& option, Config& config, const JsonValue& value,
                 std::string_view origin) {
    const auto path = option.path;
    std::visit([&](auto* field) {
        using T = std::remove_pointer_t<decltype(field)>;
        if constexpr (std::is_same_v<T, std::uint64_t> || std::is_same_v<T, std::uint32_t> ||
                      std::is_same_v<T, std::int32_t>) {
            *field = json_integer<T>(value, origin, path);
        } else if constexpr (std::is_same_v<T, double>) {
            const auto parsed = value.as_double();
            if (!parsed)
                json_error(origin, value, std::string(path) + " expects a finite number, got " +
                                              describe(value));
            *field = *parsed;
        } else if constexpr (std::is_same_v<T, bool>) {
            if (value.type != JsonValue::Type::Boolean)
                json_error(origin, value, std::string(path) + " expects true or false, got " +
                                              describe(value));
            *field = value.boolean;
        } else if constexpr (std::is_same_v<T, std::string>) {
            if (value.type != JsonValue::Type::String)
                json_error(origin, value, std::string(path) + " expects a string, got " +
                                              describe(value));
            *field = value.text;
        } else if constexpr (is_enum_field<T>) {
            const auto parsed = value.type == JsonValue::Type::String ?
                from_string<T>(value.text) : std::nullopt;
            if (!parsed)
                json_error(origin, value, std::string(path) + " expects one of " +
                                              enum_choices<T>() + ", got " + describe(value));
            *field = *parsed;
        } else if constexpr (std::is_same_v<T, std::vector<std::int32_t>>) {
            if (value.type != JsonValue::Type::Array)
                json_error(origin, value, std::string(path) + " expects an array, got " +
                                              describe(value));
            T result;
            result.reserve(value.items.size());
            for (const auto& item : value.items)
                result.push_back(json_integer<typename T::value_type>(item, origin, path));
            *field = std::move(result);
        } else {
            static_assert(always_false<T>, "unhandled option field type");
        }
    }, option.field(config));
}

void write_value(std::ostream& stream, const ConstFieldRef& view) {
    std::visit([&](const auto* field) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(field)>>;
        if constexpr (std::is_same_v<T, std::uint64_t> || std::is_same_v<T, std::uint32_t> ||
                      std::is_same_v<T, std::int32_t>) {
            stream << *field;
        } else if constexpr (std::is_same_v<T, double>) {
            write_json_number(stream, *field);
        } else if constexpr (std::is_same_v<T, bool>) {
            stream << (*field ? "true" : "false");
        } else if constexpr (std::is_same_v<T, std::string>) {
            write_json_string(stream, *field);
        } else if constexpr (is_enum_field<T>) {
            write_json_string(stream, to_string(*field));
        } else {
            stream << '[';
            for (std::size_t i = 0; i < field->size(); ++i) {
                if (i) stream << ',';
                stream << (*field)[i];
            }
            stream << ']';
        }
    }, view);
}

std::vector<std::string_view> split_path(std::string_view path) {
    std::vector<std::string_view> segments;
    while (true) {
        const auto dot = path.find('.');
        segments.push_back(path.substr(0, dot));
        if (dot == std::string_view::npos) return segments;
        path.remove_prefix(dot + 1);
    }
}

bool is_group(std::string_view path) {
    for (const auto& option : option_table) {
        if (option.path.size() > path.size() && option.path.substr(0, path.size()) == path &&
            option.path[path.size()] == '.')
            return true;
    }
    return false;
}

std::string metavar(const ConfigOption& option) {
    if (!option.metavar.empty()) return std::string(option.metavar);
    return std::visit([](const auto* field) -> std::string {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(field)>>;
        if constexpr (is_enum_field<T>) return enum_choices<T>();
        else if constexpr (std::is_same_v<T, bool>) return "";
        else if constexpr (std::is_same_v<T, double>) return "X";
        else if constexpr (std::is_same_v<T, std::string>) return "TEXT";
        else return "N";
    }, option.view(Config{}));
}

std::string default_text(const ConfigOption& option) {
    const Config defaults;
    const auto view = option.view(defaults);
    const bool empty = std::visit([](const auto* field) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(field)>>;
        if constexpr (std::is_same_v<T, std::string> ||
                      std::is_same_v<T, std::vector<std::int32_t>>)
            return field->empty();
        else return false;
    }, view);
    if (empty) return "";
    std::ostringstream text;
    write_value(text, view);
    auto result = text.str();
    if (result.size() >= 2 && result.front() == '"') result = result.substr(1, result.size() - 2);
    return result;
}

// Sources are indexed by table row, so only rows of option_table are accepted.
std::size_t row_index(const ConfigOption& option) {
    const std::less<const ConfigOption*> before;
    const auto* first = option_table.data();
    if (before(&option, first) || !before(&option, first + option_table.size()))
        throw std::logic_error("option is not a row of the option table");
    return static_cast<std::size_t>(&option - first);
}

}  // namespace

const std::vector<ConfigOption>& config_options() { return option_table; }

const ConfigOption* find_option_by_flag(std::string_view flag) {
    if (flag.empty()) return nullptr;
    for (const auto& option : option_table)
        if (option.flag == flag) return &option;
    return nullptr;
}

const ConfigOption* find_option_by_path(std::string_view path) {
    for (const auto& option : option_table)
        if (option.path == path) return &option;
    return nullptr;
}

bool is_switch(const ConfigOption& option) {
    return std::holds_alternative<const bool*>(option.view(Config{}));
}

void ConfigBuilder::mark(const ConfigOption& option, OptionSource source) {
    sources_.at(row_index(option)) = source;
}

void ConfigBuilder::set(const ConfigOption& option, std::string_view text) {
    const auto flag = option.flag;
    std::visit([&](auto* field) {
        using T = std::remove_pointer_t<decltype(field)>;
        if constexpr (std::is_same_v<T, std::uint64_t>) {
            *field = parse_uint64(text, flag);
        } else if constexpr (std::is_same_v<T, std::uint32_t>) {
            *field = parse_uint32(text, flag);
        } else if constexpr (std::is_same_v<T, std::int32_t>) {
            *field = parse_int32(text, flag);
        } else if constexpr (std::is_same_v<T, double>) {
            *field = parse_decimal(text, flag);
        } else if constexpr (std::is_same_v<T, std::string>) {
            if (text.empty())
                throw std::invalid_argument(std::string(flag) + ": expected a nonempty path");
            *field = std::string(text);
        } else if constexpr (is_enum_field<T>) {
            const auto parsed = from_string<T>(text);
            if (!parsed)
                throw std::invalid_argument(std::string(flag) + ": expected " + enum_choices<T>() +
                                            ", got " + quoted(text));
            *field = *parsed;
        } else {
            throw std::logic_error(std::string(option.path) + " takes no command-line value");
        }
    }, option.field(config));
    mark(option, OptionSource::CommandLine);
}

void ConfigBuilder::set_switch(const ConfigOption& option) {
    const auto field = option.field(config);
    const auto* target = std::get_if<bool*>(&field);
    if (!target) throw std::logic_error(std::string(option.flag) + " is not a switch");
    **target = true;
    mark(option, OptionSource::CommandLine);
}

void ConfigBuilder::load_object(const JsonValue& object, const std::string& prefix,
                                std::string_view origin) {
    for (std::size_t i = 0; i < object.keys.size(); ++i) {
        const auto path = prefix.empty() ? object.keys[i] : prefix + '.' + object.keys[i];
        const auto& value = object.items[i];
        if (const auto* option = find_option_by_path(path)) {
            assign_json(*option, config, value, origin);
            mark(*option, OptionSource::File);
        } else if (is_group(path)) {
            if (value.type != JsonValue::Type::Object)
                json_error(origin, value, path + " expects an object, got " + describe(value));
            load_object(value, path, origin);
        } else {
            std::ostringstream key;
            write_json_string(key, path);
            json_error(origin, value, "unknown config key " + key.str());
        }
    }
}

void ConfigBuilder::load(const JsonValue& document, std::string_view origin) {
    if (document.type != JsonValue::Type::Object)
        json_error(origin, document, "a config file holds one JSON object, got " +
                                         describe(document));
    load_object(document, "", origin);
}

void ConfigBuilder::load_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::invalid_argument("cannot open config file " + path);
    std::string text;
    char buffer[65536];
    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
        text.append(buffer, static_cast<std::size_t>(file.gcount()));
        if (text.size() > max_config_file_bytes)
            throw std::invalid_argument(path + ": config file exceeds 64 MiB");
    }
    if (file.bad()) throw std::invalid_argument("cannot read config file " + path);
    load(parse_json(text, path), path);
}

OptionSource ConfigBuilder::source(std::string_view path) const {
    const auto* option = find_option_by_path(path);
    if (!option) throw std::logic_error("unknown option path " + std::string(path));
    return sources_.at(row_index(*option));
}

void write_config_json(std::ostream& stream, const Config& config) {
    std::vector<std::string_view> open;
    std::vector<bool> nonempty{false};
    const auto member = [&](std::string_view key) {
        if (nonempty.back()) stream << ',';
        nonempty.back() = true;
        write_json_string(stream, key);
        stream << ':';
    };
    stream << '{';
    for (const auto& option : option_table) {
        auto segments = split_path(option.path);
        const auto leaf = segments.back();
        segments.pop_back();
        std::size_t common = 0;
        while (common < open.size() && common < segments.size() && open[common] == segments[common])
            ++common;
        for (; open.size() > common; open.pop_back(), nonempty.pop_back()) stream << '}';
        for (; open.size() < segments.size(); nonempty.push_back(false)) {
            member(segments[open.size()]);
            stream << '{';
            open.push_back(segments[open.size()]);
        }
        member(leaf);
        write_value(stream, option.view(config));
    }
    for (; !open.empty(); open.pop_back()) stream << '}';
    stream << '}';
}

std::string config_json(const Config& config) {
    std::ostringstream text;
    write_config_json(text, config);
    return text.str();
}

void write_help_row(std::ostream& stream, std::string_view flag, std::string_view placeholder,
                    std::string_view key, std::string_view help) {
    std::string head = "  ";
    head += flag.empty() ? key : flag;
    if (!placeholder.empty()) {
        head += ' ';
        head += placeholder;
    }
    if (!flag.empty() && !key.empty()) {
        head.append(head.size() < 40 ? 40 - head.size() : 1, ' ');
        head += key;
    }
    stream << head << "\n      " << help << '\n';
}

void write_config_help(std::ostream& stream) {
    for (const auto& option : option_table) {
        std::string help(option.help);
        const auto fallback = default_text(option);
        if (!fallback.empty()) help += " (default " + fallback + ")";
        write_help_row(stream, option.flag, option.flag.empty() ? "" : metavar(option),
                       option.path, help);
    }
}

std::uint64_t parse_uint64(std::string_view text, std::string_view what) {
    if (const auto value = integer_value<std::uint64_t>(text)) return *value;
    throw std::invalid_argument(std::string(what) + ": expected an unsigned 64-bit integer, got " +
                                quoted(text));
}

std::int32_t parse_int32(std::string_view text, std::string_view what) {
    if (const auto value = integer_value<std::int32_t>(text)) return *value;
    throw std::invalid_argument(std::string(what) + ": expected a signed 32-bit integer, got " +
                                quoted(text));
}

std::uint32_t parse_uint32(std::string_view text, std::string_view what) {
    if (const auto value = integer_value<std::uint32_t>(text)) return *value;
    throw std::invalid_argument(std::string(what) + ": expected an unsigned 32-bit integer, got " +
                                quoted(text));
}

double parse_decimal(std::string_view text, std::string_view what) {
    if (!json_number_grammar(text))
        throw std::invalid_argument(std::string(what) + ": expected a decimal number, got " +
                                    quoted(text));
    const std::string copy(text);
    errno = 0;
    const double value = std::strtod(copy.c_str(), nullptr);
    if (errno == ERANGE && std::isinf(value))
        throw std::invalid_argument(std::string(what) + ": number out of range: " + quoted(text));
    return value;
}

}  // namespace npu
