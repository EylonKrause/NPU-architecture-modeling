// Strict JSON reader and writer used for configuration files.
#include "npu/json.hpp"

#include <iostream>
#include <limits>
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

}  // namespace

int main() {
    try {
        check_json_reader();
        std::cout << "config tests: all checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "config tests FAILED: " << error.what() << '\n';
        return 1;
    }
}
