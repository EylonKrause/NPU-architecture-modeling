// Option table, enum spellings, strict JSON reader and config-file loader.
#include "npu/config.hpp"
#include "npu/core.hpp"
#include "npu/json.hpp"
#include "npu/options.hpp"

#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Only the named exception type counts, and its message must contain `expected`.
template <class Exception = std::invalid_argument, class Function>
void rejects(Function function, std::string_view expected) {
    try {
        function();
    } catch (const Exception& error) {
        const std::string_view message(error.what());
        if (message.find(expected) == std::string_view::npos)
            throw std::runtime_error("message '" + std::string(message) + "' lacks '" +
                                     std::string(expected) + "'");
        return;
    }
    throw std::runtime_error("expected rejection containing '" + std::string(expected) + "'");
}

const npu::ConfigOption& option(std::string_view flag) {
    const auto* found = npu::find_option_by_flag(flag);
    require(found != nullptr, "missing option " + std::string(flag));
    return *found;
}

void load_text(npu::ConfigBuilder& builder, std::string_view text) {
    builder.load(npu::parse_json(text, "test"), "test");
}

template <class E>
void check_enum_round_trip() {
    for (const auto name : npu::EnumNames<E>::names) {
        const auto value = npu::from_string<E>(name);
        require(value.has_value() && npu::to_string(*value) == name,
                "enum spelling round trip: " + std::string(name));
    }
    require(!npu::from_string<E>(""), "empty enum spelling");
    require(!npu::from_string<E>(std::string(npu::EnumNames<E>::names[0]) + " "),
            "enum spellings match exactly");
}

void check_enums() {
    check_enum_round_trip<npu::Rounding>();
    check_enum_round_trip<npu::RescaleMode>();
    check_enum_round_trip<npu::WeightEncoding>();
    check_enum_round_trip<npu::TimingMode>();
    check_enum_round_trip<npu::Dataflow>();
    check_enum_round_trip<npu::MemoryModel>();
    check_enum_round_trip<npu::Partition>();
    require(npu::to_string(npu::Rounding::HalfAwayFromZero) == "away" &&
                npu::to_string(npu::Rounding::HalfToEven) == "even",
            "rounding spellings");
    require(npu::to_string(npu::RescaleMode::TosaDouble) == "tosa-double", "rescale spelling");
    require(npu::to_string(npu::Dataflow::WeightStationary) == "ws", "dataflow spelling");
    require(npu::to_string(npu::Partition::Auto) == "auto", "partition spelling");
    require(!npu::from_string<npu::Rounding>("AWAY"), "spellings are case-sensitive");
    require(npu::enum_choices<npu::WeightEncoding>() == "dense|palette|sparse", "choice list");
}

void check_option_table() {
    const auto& table = npu::config_options();
    std::set<std::string_view> flags, paths;
    for (const auto& entry : table) {
        require(entry.flag.empty() || entry.flag.substr(0, 2) == "--", "flags start with --");
        require(entry.flag.empty() || flags.insert(entry.flag).second,
                "duplicate flag " + std::string(entry.flag));
        require(paths.insert(entry.path).second, "duplicate path " + std::string(entry.path));
        require(!entry.help.empty(), "help text for " + std::string(entry.path));
    }
    // No path is both a leaf and a group, and each group's rows are contiguous
    // so the JSON writer opens every group once.
    for (const auto& entry : table) {
        for (auto dot = entry.path.find('.'); dot != std::string_view::npos;
             dot = entry.path.find('.', dot + 1)) {
            const auto group = std::string(entry.path.substr(0, dot + 1));
            require(!paths.count(entry.path.substr(0, dot)), "leaf shadows group " + group);
            std::size_t first = table.size(), last = 0, count = 0;
            for (std::size_t i = 0; i < table.size(); ++i) {
                if (table[i].path.substr(0, group.size()) != group) continue;
                first = std::min(first, i);
                last = i;
                ++count;
            }
            require(last - first + 1 == count, "group rows are contiguous: " + group);
        }
    }
    // Every flag the simulator accepted before the option table still works.
    for (const auto* legacy : {"--m", "--n", "--k", "--tile-m", "--tile-n", "--tile-k",
             "--weight-bits", "--weight-encoding", "--palette-bits", "--sparse-block",
             "--weight-decode-values-per-cycle", "--array-rows", "--array-cols",
             "--macs-per-cycle", "--memory-bytes-per-cycle", "--memory-latency",
             "--dma-burst-bytes", "--weight-cache-bytes", "--pipeline-cycles",
             "--buffer-slots", "--sram-bytes", "--seed", "--frequency-mhz", "--timing",
             "--input-zero", "--weight-zero", "--output-zero", "--multiplier", "--shift",
             "--rounding", "--relu", "--trace"})
        option(legacy);
    require(npu::is_switch(option("--relu")) && !npu::is_switch(option("--m")), "switches");
    require(npu::find_option_by_flag("--rescale")->path == "quant.rescale", "rescale option");
    std::ostringstream help;
    npu::write_config_help(help);
    for (const auto& entry : table) {
        require(help.str().find(entry.path) != std::string::npos,
                "help lists " + std::string(entry.path));
        require(entry.flag.empty() || help.str().find(entry.flag) != std::string::npos,
                "help lists " + std::string(entry.flag));
    }
    require(help.str().find("dense|palette|sparse") != std::string::npos &&
                help.str().find("native|tosa-single|tosa-double") != std::string::npos,
            "help shows choices");
    require(help.str().find("(default 16)") != std::string::npos, "help shows defaults");
}

void check_json_reader() {
    const std::pair<std::string, std::string> malformed[] = {
        {"", "1:1: unexpected end of input"},
        {"   ", "unexpected end of input"},
        {"{", "expected a string member name"},
        {"{\"a\":1,}", "expected a string member name"},
        {"{'a':1}", "expected a string member name"},
        {"{\"a\" 1}", "expected ':' after member name"},
        {"{\"a\":1 \"b\":2}", "expected ',' or '}' in object"},
        {"{\"a\":1", "unterminated object"},
        {"[1,2", "unterminated array"},
        {"[1 2]", "expected ',' or ']' in array"},
        {"[1,]", "unexpected character ']'"},
        {"{\"a\":01}", "1:7: expected ',' or '}' in object"},
        {"{\"a\":1.}", "expected a digit after the decimal point"},
        {"{\"a\":-}", "expected a digit"},
        {"{\"a\":1e}", "expected an exponent digit"},
        {"{\"a\":+1}", "unexpected character '+'"},
        {"{\"a\":.5}", "unexpected character '.'"},
        {"{\"a\":NaN}", "unexpected character 'N'"},
        {"{\"a\":tru}", "invalid literal"},
        {"{\"a\":nul}", "invalid literal"},
        {"{\"a\":\"x}", "unterminated string"},
        {"{\"a\":\"\\q\"}", "invalid escape sequence"},
        {"{\"a\":\"\\u12\"}", "invalid \\u escape"},
        {"{\"a\":\"\\ud800\"}", "unpaired high surrogate"},
        {"{\"a\":\"\\ud800\\u0041\"}", "unpaired high surrogate"},
        {"{\"a\":\"\\udc00\"}", "unpaired low surrogate"},
        {"{\"a\":\"\x01\"}", "control character in string"},
        {"{\"a\":\"\t\"}", "control character in string"},
        {"{\"a\":\"\xff\"}", "invalid UTF-8"},
        {"{\"a\":\"\xc3\"}", "invalid UTF-8"},
        {"{\"a\":\"\xc0\xaf\"}", "invalid UTF-8"},
        {"{\"a\":\"\xed\xa0\x80\"}", "invalid UTF-8"},
        {"{\"a\":\"\xf4\x90\x80\x80\"}", "invalid UTF-8"},
        {"{\"a\":1,\"a\":2}", "1:8: duplicate member \"a\""},
        {"{\n  \"a\": 1,\n  \"a\": 2\n}", "test:3:3: duplicate member \"a\""},
        {"{} {}", "unexpected content after the JSON value"},
        {"{}x", "1:3: unexpected content after the JSON value"},
        {"// note\n{}", "unexpected character '/'"},
        {std::string(65, '[') + std::string(65, ']'), "nesting exceeds 64 levels"},
    };
    for (const auto& [text, expected] : malformed)
        rejects([&] { npu::parse_json(text, "test"); }, expected);

    const auto nested = npu::parse_json(std::string(64, '[') + std::string(64, ']'), "test");
    require(nested.type == npu::JsonValue::Type::Array, "64 nesting levels are accepted");
    const auto bom = npu::parse_json("\xEF\xBB\xBF{}", "test");
    require(bom.type == npu::JsonValue::Type::Object && bom.keys.empty(), "byte order mark");

    const auto document = npu::parse_json(
        "{\"s\":\"\\u00e9\\ud83d\\ude00\\n\\/\", \"u\":18446744073709551615, "
        "\"big\":18446744073709551616, \"neg\":-1, \"fraction\":1.0, \"e\":2E3, \"huge\":1e999, "
        "\"tiny\":1e-400, \"t\":true, \"f\":false, \"z\":null, \"list\":[1,[2],{}]}", "test");
    require(document.keys.size() == 12 && document.keys[0] == "s" && document.keys[11] == "list",
            "member order is preserved");
    const auto& items = document.items;
    require(items[0].text == "\xc3\xa9\xf0\x9f\x98\x80\n/", "escapes decode to UTF-8");
    require(items[1].as_uint64() == std::numeric_limits<std::uint64_t>::max(), "uint64 maximum");
    require(!items[2].as_uint64() && items[2].as_double(), "uint64 overflow");
    require(!items[3].as_uint64() && items[3].as_int64() == -1, "negative integer");
    require(!items[4].integer && !items[4].as_uint64() && items[4].as_double() == 1.0,
            "a fraction is not an integer");
    require(!items[5].integer && items[5].as_double() == 2000.0, "exponent");
    require(!items[6].as_double(), "overflowing double");
    require(items[7].as_double() && *items[7].as_double() >= 0.0, "underflow rounds toward zero");
    require(items[8].boolean && !items[9].boolean &&
                items[10].type == npu::JsonValue::Type::Null, "literals");
    require(items[11].items.size() == 3 && items[11].items[1].items[0].as_uint64() == 2U,
            "nested arrays");
    require(items[1].line == 1 && items[1].column > 1, "values carry positions");

    std::string every_byte;
    for (int c = 1; c < 128; ++c) every_byte += static_cast<char>(c);
    every_byte += "\xc3\xa9";
    std::ostringstream encoded;
    encoded << "{\"k\":";
    npu::write_json_string(encoded, every_byte);
    encoded << '}';
    require(npu::parse_json(encoded.str(), "test").items[0].text == every_byte,
            "string writer round trip");

    const auto number = [](double value) {
        std::ostringstream text;
        npu::write_json_number(text, value);
        return text.str();
    };
    require(number(200.0) == "200" && number(1000.0) == "1000" && number(0.1) == "0.1" &&
                number(1e-6) == "1e-06" && number(1500.5) == "1500.5" && number(1e20) == "1e+20",
            "shortest number form");
    require(number(std::numeric_limits<double>::infinity()) == "null" &&
                number(std::numeric_limits<double>::quiet_NaN()) == "null",
            "non-finite numbers");
    for (const double value :
         {1.0 / 3.0, 2.0 / 3.0e-7, 123456789.125, 5e-324, 1.7976931348623157e308})
        require(npu::parse_json(number(value), "test").as_double() == value,
                "number writer round trip " + number(value));
}

void check_loader(const std::string& example_path) {
    npu::ConfigBuilder builder;
    load_text(builder,
              "{\"name\":\"probe\",\"shape\":{\"m\":4},\"arch\":{\"frequency_mhz\":1.5e3,"
              "\"memory\":{\"burst_bytes\":64}},\"quant\":{\"relu\":true,\"rounding\":\"even\","
              "\"shifts\":[1,-2,3,62],\"weight_zero_points\":[-1,0,1,2],"
              "\"rescale\":\"tosa-double\"},\"weights\":{\"encoding\":\"palette\"},"
              "\"sim\":{\"timing\":\"cycle\"}}");
    const auto& c = builder.config;
    require(c.name == "probe" && c.shape.m == 4 && c.arch.frequency_mhz == 1500.0 &&
                c.arch.memory.burst_bytes == 64 && c.quant.relu &&
                c.quant.rounding == npu::Rounding::HalfToEven &&
                c.quant.shifts == std::vector<std::int32_t>{1, -2, 3, 62} &&
                c.quant.rescale == npu::RescaleMode::TosaDouble &&
                c.quant.weight_zero_points == std::vector<std::int32_t>{-1, 0, 1, 2} &&
                c.weights.encoding == npu::WeightEncoding::Palette &&
                c.sim.timing == npu::TimingMode::Cycle,
            "config file assigns typed fields");
    require(c.shape.n == npu::Config{}.shape.n, "unmentioned keys keep their defaults");
    require(builder.source("shape.m") == npu::OptionSource::File &&
                builder.source("shape.n") == npu::OptionSource::Default,
            "sources record file assignments");

    const std::pair<std::string, std::string> invalid[] = {
        {"[1]", "test:1:1: a config file holds one JSON object, got an array"},
        {"{\"foo\":1}", "test:1:8: unknown config key \"foo\""},
        {"{\"shape\":{\"q\":1}}", "unknown config key \"shape.q\""},
        {"{\"shape\":{\"m\":1,\"name\":\"x\"}}", "unknown config key \"shape.name\""},
        {"{\"arch\":{\"compute\":{\"macs\":1}}}", "unknown config key \"arch.compute.macs\""},
        {"{\"shape\":5}", "shape expects an object, got 5"},
        {"{\"arch\":{\"memory\":[]}}", "arch.memory expects an object, got an array"},
        {"{\"shape\":{\"m\":{}}}", "shape.m expects an unsigned 64-bit integer, got an object"},
        {"{\"shape\":{\"m\":-1}}", "shape.m expects an unsigned 64-bit integer, got -1"},
        {"{\"shape\":{\"m\":1.5}}", "shape.m expects an unsigned 64-bit integer, got 1.5"},
        {"{\"shape\":{\"m\":16.0}}", "got 16.0"},
        {"{\"shape\":{\"m\":\"16\"}}", "shape.m expects an unsigned 64-bit integer, got \"16\""},
        {"{\"shape\":{\"m\":18446744073709551616}}", "got 18446744073709551616"},
        {"{\"shape\":{\"m\":null}}", "got null"},
        {"{\"quant\":{\"input_zero\":2147483648}}", "quant.input_zero expects a signed 32-bit"},
        {"{\"quant\":{\"multiplier\":2147483648}}", "quant.multiplier expects a signed 32-bit"},
        {"{\"quant\":{\"rescale\":\"tosa\"}}",
         "quant.rescale expects one of native|tosa-single|tosa-double, got \"tosa\""},
        {"{\"quant\":{\"relu\":1}}", "quant.relu expects true or false, got 1"},
        {"{\"quant\":{\"rounding\":\"floor\"}}",
         "quant.rounding expects one of away|even, got \"floor\""},
        {"{\"weights\":{\"encoding\":3}}", "weights.encoding expects one of dense|palette|sparse"},
        {"{\"arch\":{\"frequency_mhz\":\"fast\"}}", "arch.frequency_mhz expects a finite number"},
        {"{\"arch\":{\"frequency_mhz\":1e999}}", "got 1e999"},
        {"{\"quant\":{\"shifts\":3}}", "quant.shifts expects an array, got 3"},
        {"{\"quant\":{\"shifts\":[1,2147483648]}}", "1:23: quant.shifts expects a signed 32-bit"},
        {"{\"quant\":{\"weight_zero_points\":[true]}}",
         "expects a signed 32-bit integer, got true"},
        {"{\"name\":5}", "name expects a string, got 5"},
        {"{\"sim\":{\"trace_path\":false}}", "sim.trace_path expects a string, got false"},
    };
    for (const auto& [text, expected] : invalid) {
        npu::ConfigBuilder fresh;
        rejects([&] { load_text(fresh, text); }, expected);
    }
    npu::ConfigBuilder missing;
    rejects([&] { missing.load_file(example_path + ".missing"); }, "cannot open config file");

    npu::ConfigBuilder example;
    example.load_file(example_path);
    require(!example.config.name.empty() && !example.config.description.empty(),
            "example names itself");
    npu::validate(example.config);
}

void check_override_order() {
    npu::ConfigBuilder builder;
    const auto& m = option("--m");
    builder.set(m, "4");
    require(builder.config.shape.m == 4 &&
                builder.source("shape.m") == npu::OptionSource::CommandLine, "flag sets value");
    load_text(builder, "{\"shape\":{\"m\":8}}");
    require(builder.config.shape.m == 8 && builder.source("shape.m") == npu::OptionSource::File,
            "a later file replaces an earlier flag");
    builder.set(m, "2");
    require(builder.config.shape.m == 2, "a later flag replaces the file");
    load_text(builder, "{\"shape\":{\"n\":3}}");
    require(builder.config.shape.m == 2 && builder.config.shape.n == 3 &&
                builder.source("shape.m") == npu::OptionSource::CommandLine,
            "a file leaves keys it does not mention");
    builder.set_switch(option("--relu"));
    require(builder.config.quant.relu, "switch sets true");
    load_text(builder, "{\"quant\":{\"relu\":false}}");
    require(!builder.config.quant.relu, "a file can clear a switch");
}

void check_command_line_values() {
    npu::ConfigBuilder builder;
    const std::pair<std::pair<const char*, const char*>, std::string> invalid[] = {
        {{"--m", "4junk"}, "--m: expected an unsigned 64-bit integer, got '4junk'"},
        {{"--m", ""}, "--m: expected an unsigned 64-bit integer, got ''"},
        {{"--m", "+4"}, "got '+4'"},
        {{"--m", " 4"}, "got ' 4'"},
        {{"--m", "-1"}, "got '-1'"},
        {{"--m", "18446744073709551616"}, "got '18446744073709551616'"},
        {{"--input-zero", "2147483648"}, "--input-zero: expected a signed 32-bit integer"},
        {{"--multiplier", "2147483648"},
         "--multiplier: expected a signed 32-bit integer, got '2147483648'"},
        {{"--shift", "1.5"}, "--shift: expected a signed 32-bit integer, got '1.5'"},
        {{"--rescale", "tosa"}, "--rescale: expected native|tosa-single|tosa-double, got 'tosa'"},
        {{"--frequency-mhz", "abc"}, "--frequency-mhz: expected a decimal number, got 'abc'"},
        {{"--frequency-mhz", "nan"}, "--frequency-mhz: expected a decimal number, got 'nan'"},
        {{"--frequency-mhz", "inf"}, "--frequency-mhz: expected a decimal number"},
        {{"--frequency-mhz", " 200"}, "--frequency-mhz: expected a decimal number, got ' 200'"},
        {{"--frequency-mhz", "0x10"}, "--frequency-mhz: expected a decimal number"},
        {{"--frequency-mhz", "200MHz"}, "--frequency-mhz: expected a decimal number"},
        {{"--frequency-mhz", "1e999"}, "--frequency-mhz: number out of range: '1e999'"},
        {{"--rounding", "floor"}, "--rounding: expected away|even, got 'floor'"},
        {{"--weight-encoding", "zip"}, "--weight-encoding: expected dense|palette|sparse"},
        {{"--timing", "rtl"}, "--timing: expected event|cycle, got 'rtl'"},
        {{"--trace", ""}, "--trace: expected a nonempty path"},
    };
    for (const auto& [arguments, expected] : invalid)
        rejects([&] { builder.set(option(arguments.first), arguments.second); }, expected);
    rejects<std::logic_error>([&] { builder.set(option("--relu"), "1"); }, "no command-line value");
    rejects<std::logic_error>([&] { builder.set_switch(option("--m")); }, "not a switch");
    const auto detached = option("--m");
    rejects<std::logic_error>([&] { builder.set(detached, "4"); }, "not a row of the option table");
    for (const auto* text : {"200", "1.5", "2e2", "0.000001", "-3"})
        builder.set(option("--frequency-mhz"), text);
    require(builder.config.arch.frequency_mhz == -3.0, "decimal values parse");
    builder.set(option("--frequency-mhz"), "2E2");
    require(builder.config.arch.frequency_mhz == 200.0, "exponent form");
    builder.set(option("--input-zero"), "-129");
    require(builder.config.quant.input_zero == -129, "range checks belong to validate()");
    rejects([&] { npu::validate(builder.config); }, "input zero point must be signed INT8");
    builder.set(option("--input-zero"), "0");
    builder.set(option("--multiplier"), "-1");
    builder.set(option("--shift"), "-3");
    require(builder.config.quant.multiplier == -1 && builder.config.quant.shift == -3,
            "multiplier and shift parse as signed");
    rejects([&] { npu::validate(builder.config); },
            "requantization multiplier must be in 0..2147483647");
    builder.set(option("--multiplier"), "1");
    rejects([&] { npu::validate(builder.config); },
            "requantization shift must be in 0..31 for native rescale");
    builder.set(option("--rescale"), "tosa-single");
    rejects([&] { npu::validate(builder.config); },
            "requantization shift must be in 2..62 for tosa-single rescale");
    builder.set(option("--shift"), "62");
    npu::validate(builder.config);
    builder.set(option("--rounding"), "even");
    rejects([&] { npu::validate(builder.config); }, "rounding even applies to native rescale only");
    builder.set(option("--rounding"), "away");
    builder.set(option("--weight-encoding"), "sparse");
    builder.set(option("--timing"), "cycle");
    require(builder.config.weights.encoding == npu::WeightEncoding::Sparse &&
                builder.config.sim.timing == npu::TimingMode::Cycle, "enum flags");
    rejects([&] { npu::parse_uint64("x", "NPU1 header"); },
            "NPU1 header: expected an unsigned 64-bit integer, got 'x'");
}

void collect_leaves(const npu::JsonValue& value, const std::string& prefix,
                    std::vector<std::string>& leaves) {
    if (value.type != npu::JsonValue::Type::Object) {
        leaves.push_back(prefix);
        return;
    }
    for (std::size_t i = 0; i < value.keys.size(); ++i)
        collect_leaves(value.items[i],
                       prefix.empty() ? value.keys[i] : prefix + '.' + value.keys[i], leaves);
}

void check_echo() {
    const auto defaults = npu::parse_json(npu::config_json(npu::Config{}), "echo");
    require(defaults.keys == std::vector<std::string>{"name", "description", "shape", "arch",
                                                       "mapping", "weights", "quant", "sim"},
            "top-level groups in table order");
    require(defaults.items[3].keys == std::vector<std::string>{"compute", "memory", "sram_bytes",
                                                               "weight_cache_bytes",
                                                               "frequency_mhz"},
            "arch nests compute and memory");
    std::vector<std::string> leaves;
    collect_leaves(defaults, "", leaves);
    const auto& table = npu::config_options();
    require(leaves.size() == table.size(), "echo holds every option once");
    for (std::size_t i = 0; i < table.size(); ++i)
        require(leaves[i] == table[i].path, "echo order follows the table: " + leaves[i]);

    npu::Config c;
    c.name = "round \"trip\"\n";
    c.description = "\xc3\xa9";
    c.shape = {5, 6, 7, 1ULL << 40};
    c.arch.compute = {3, 2, 4, 9, 11};
    c.arch.memory = {8, 30, 512};
    c.arch.sram_bytes = 1000000;
    c.arch.weight_cache_bytes = 4096;
    c.arch.frequency_mhz = 933.3333333333334;
    c.mapping = {3, 4, 5, 6};
    c.weights = {4, npu::WeightEncoding::Sparse, 2, 16};
    c.quant.input_zero = -128;
    c.quant.weight_zero = 7;
    c.quant.output_zero = 127;
    c.quant.multiplier = 2147483647;
    c.quant.shift = 31;
    c.quant.relu = true;
    c.quant.rescale = npu::RescaleMode::TosaSingle;
    c.quant.rounding = npu::Rounding::HalfToEven;
    c.quant.weight_zero_points = {-8, 0, 7, 1, 2, 3};
    c.quant.multipliers = {0, 1, 2147483647, 3, 4, 5};
    c.quant.shifts = {0, 31, -2, 62, 4, -2147483647 - 1};
    c.sim.timing = npu::TimingMode::Cycle;
    c.sim.trace_path = "C:\\traces\\run 1.csv";
    const auto text = npu::config_json(c);
    npu::ConfigBuilder builder;
    load_text(builder, text);
    require(npu::config_json(builder.config) == text, "echo reloads to the same config");
    for (const auto& entry : table)
        require(builder.source(entry.path) == npu::OptionSource::File,
                "echo assigns " + std::string(entry.path));
    const auto& r = builder.config;
    require(r.shape.seed == (1ULL << 40) &&
                r.arch.frequency_mhz == c.arch.frequency_mhz &&
                r.quant.multipliers == c.quant.multipliers && r.quant.shifts == c.quant.shifts &&
                r.quant.rescale == c.quant.rescale && r.name == c.name &&
                r.sim.trace_path == c.sim.trace_path,
            "echoed values are exact");
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc != 2) throw std::runtime_error("usage: npu-config-tests configs/example.json");
        check_enums();
        check_option_table();
        check_json_reader();
        check_loader(argv[1]);
        check_override_order();
        check_command_line_values();
        check_echo();
        std::cout << "config tests: all checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "config tests FAILED: " << error.what() << '\n';
        return 1;
    }
}
