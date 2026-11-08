#pragma once

#include "npu/config.hpp"
#include "npu/json.hpp"

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace npu {

// One Config field. The pointer type held selects how CLI text and JSON values
// are parsed, validated for type and printed. Every enum listed has EnumNames.
template <class... T>
struct FieldTypes {
    using Mutable = std::variant<T*...>;
    using Const = std::variant<const T*...>;
};
using OptionFieldTypes = FieldTypes<std::uint64_t, std::int32_t, std::uint32_t, double, bool,
                                    std::string, Rounding, RescaleMode, WeightEncoding, TimingMode,
                                    Dataflow, MemoryModel, Partition, std::vector<std::int32_t>>;
using FieldRef = OptionFieldTypes::Mutable;
using ConstFieldRef = OptionFieldTypes::Const;

// One row of the option table. The table is the only list of configuration
// fields: CLI parsing, --help, --config loading and the "config" object in
// every JSON result are all generated from it.
struct ConfigOption {
    std::string_view flag;     // "--m"; empty for keys settable only from a config file.
    std::string_view path;     // "shape.m": the config-file key and JSON echo location.
    std::string_view metavar;  // Help placeholder; empty derives one from the field type.
    std::string_view help;
    FieldRef (*field)(Config&);
    ConstFieldRef (*view)(const Config&);
};

const std::vector<ConfigOption>& config_options();
const ConfigOption* find_option_by_flag(std::string_view flag);
const ConfigOption* find_option_by_path(std::string_view path);
// A boolean option is a switch: its flag takes no value and sets true.
bool is_switch(const ConfigOption& option);

enum class OptionSource : std::uint8_t { Default, File, CommandLine };

// Applies config files and command-line values in the order given; a later
// assignment replaces an earlier one. Records where each option was last set
// so callers can detect conflicts such as --input together with --m.
class ConfigBuilder {
public:
    Config config;

    // Parses one command-line value. Errors name the flag.
    void set(const ConfigOption& option, std::string_view text);
    void set_switch(const ConfigOption& option);
    // Applies a parsed config document. Errors carry origin:line:column.
    void load(const JsonValue& document, std::string_view origin);
    void load_file(const std::string& path);
    OptionSource source(std::string_view path) const;

private:
    std::vector<OptionSource> sources_ =
        std::vector<OptionSource>(config_options().size(), OptionSource::Default);
    void load_object(const JsonValue& object, const std::string& prefix, std::string_view origin);
    void mark(const ConfigOption& option, OptionSource source);
};

constexpr std::uint64_t max_config_file_bytes = 64ULL * 1024 * 1024;

// The complete configuration as one nested JSON object. Loading the output
// with ConfigBuilder::load reproduces the Config exactly.
void write_config_json(std::ostream& stream, const Config& config);
std::string config_json(const Config& config);

// Help rows share one layout: flag and placeholder, config-file key, then the
// description on an indented line.
void write_help_row(std::ostream& stream, std::string_view flag, std::string_view metavar,
                    std::string_view key, std::string_view help);
void write_config_help(std::ostream& stream);

// Strict scalar parsers for command-line and text-file tokens: no '+', spaces
// or trailing characters; only parse_int32 accepts '-'. Errors start with `what`.
std::uint64_t parse_uint64(std::string_view text, std::string_view what);
std::int32_t parse_int32(std::string_view text, std::string_view what);
std::uint32_t parse_uint32(std::string_view text, std::string_view what);
// The decimal forms strtod reads: optional sign, digits with an optional point
// on either side, optional exponent. No hex, inf, nan or surrounding whitespace.
double parse_decimal(std::string_view text, std::string_view what);

}  // namespace npu
