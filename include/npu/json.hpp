#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace npu {

// Strict RFC 8259 reader for configuration files. Rejects duplicate object
// keys, trailing content, invalid UTF-8, unpaired surrogates and nesting
// deeper than max_json_depth. Numbers keep their source text, so integers
// convert exactly without a round trip through double.
struct JsonValue {
    enum class Type : std::uint8_t { Null, Boolean, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    bool integer = false;           // Number without fraction or exponent.
    std::string text;               // String contents or number literal.
    std::vector<std::string> keys;  // Object member names, in document order.
    std::vector<JsonValue> items;   // Array elements or object member values.
    std::size_t line = 1, column = 1;

    std::optional<std::uint64_t> as_uint64() const;
    std::optional<std::int64_t> as_int64() const;
    std::optional<double> as_double() const;
};

constexpr std::size_t max_json_depth = 64;

std::string_view json_type_name(JsonValue::Type type);

// Throws std::invalid_argument("<origin>:<line>:<column>: <reason>").
JsonValue parse_json(std::string_view text, std::string_view origin);

void write_json_string(std::ostream& stream, std::string_view text);
// Shortest decimal that reads back to the same double; null when not finite.
void write_json_number(std::ostream& stream, double value);

}  // namespace npu
