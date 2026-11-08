#include "npu/json.hpp"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>

namespace npu {
namespace {

class Parser {
public:
    Parser(std::string_view text, std::string_view origin) : text_(text), origin_(origin) {}

    JsonValue document() {
        // RFC 8259 section 8.1 permits ignoring a UTF-8 byte order mark.
        if (text_.substr(0, 3) == "\xEF\xBB\xBF") pos_ = 3;
        skip_whitespace();
        auto value = parse_value(0);
        skip_whitespace();
        if (!at_end()) fail("unexpected content after the JSON value");
        return value;
    }

private:
    std::string_view text_, origin_;
    std::size_t pos_ = 0, line_ = 1, column_ = 1;

    [[noreturn]] void fail(const std::string& reason) const {
        throw std::invalid_argument(std::string(origin_) + ':' + std::to_string(line_) + ':' +
                                    std::to_string(column_) + ": " + reason);
    }

    bool at_end() const { return pos_ >= text_.size(); }
    unsigned char peek() const { return static_cast<unsigned char>(text_[pos_]); }

    void advance() {
        if (text_[pos_] == '\n') {
            ++line_;
            column_ = 1;
        } else {
            ++column_;
        }
        ++pos_;
    }

    void skip_whitespace() {
        while (!at_end() && (peek() == ' ' || peek() == '\t' || peek() == '\n' || peek() == '\r'))
            advance();
    }

    void expect(char wanted, const char* reason) {
        if (at_end() || peek() != static_cast<unsigned char>(wanted)) fail(reason);
        advance();
    }

    JsonValue parse_value(std::size_t depth) {
        if (depth >= max_json_depth) fail("nesting exceeds 64 levels");
        if (at_end()) fail("unexpected end of input");
        JsonValue value;
        value.line = line_;
        value.column = column_;
        const auto c = peek();
        if (c == '{') parse_object(value, depth);
        else if (c == '[') parse_array(value, depth);
        else if (c == '"') {
            value.type = JsonValue::Type::String;
            value.text = parse_string();
        } else if (c == '-' || (c >= '0' && c <= '9')) parse_number(value);
        else if (c == 't' || c == 'f' || c == 'n') parse_literal(value);
        else
            fail(c < 0x20 || c >= 0x7f ?
                     std::string("unexpected byte") :
                     std::string("unexpected character '") + static_cast<char>(c) + "'");
        return value;
    }

    void parse_object(JsonValue& value, std::size_t depth) {
        value.type = JsonValue::Type::Object;
        advance();
        skip_whitespace();
        if (!at_end() && peek() == '}') {
            advance();
            return;
        }
        std::set<std::string> seen;
        while (true) {
            skip_whitespace();
            if (at_end() || peek() != '"') fail("expected a string member name");
            const auto line = line_, column = column_;
            auto key = parse_string();
            if (!seen.insert(key).second) {
                line_ = line;
                column_ = column;
                fail("duplicate member \"" + key + "\"");
            }
            skip_whitespace();
            expect(':', "expected ':' after member name");
            skip_whitespace();
            value.keys.push_back(std::move(key));
            value.items.push_back(parse_value(depth + 1));
            skip_whitespace();
            if (at_end()) fail("unterminated object");
            if (peek() == ',') {
                advance();
                continue;
            }
            expect('}', "expected ',' or '}' in object");
            return;
        }
    }

    void parse_array(JsonValue& value, std::size_t depth) {
        value.type = JsonValue::Type::Array;
        advance();
        skip_whitespace();
        if (!at_end() && peek() == ']') {
            advance();
            return;
        }
        while (true) {
            skip_whitespace();
            value.items.push_back(parse_value(depth + 1));
            skip_whitespace();
            if (at_end()) fail("unterminated array");
            if (peek() == ',') {
                advance();
                continue;
            }
            expect(']', "expected ',' or ']' in array");
            return;
        }
    }

    void parse_literal(JsonValue& value) {
        for (const std::string_view word : {"true", "false", "null"}) {
            if (text_.substr(pos_, word.size()) != word) continue;
            for (std::size_t i = 0; i < word.size(); ++i) advance();
            value.type = word == "null" ? JsonValue::Type::Null : JsonValue::Type::Boolean;
            value.boolean = word == "true";
            return;
        }
        fail("invalid literal");
    }

    bool digit() const { return !at_end() && peek() >= '0' && peek() <= '9'; }

    void digits(const char* reason) {
        if (!digit()) fail(reason);
        while (digit()) advance();
    }

    void parse_number(JsonValue& value) {
        value.type = JsonValue::Type::Number;
        value.integer = true;
        const auto start = pos_;
        if (peek() == '-') advance();
        if (!digit()) fail("expected a digit");
        if (peek() == '0') advance();
        else digits("expected a digit");
        if (!at_end() && peek() == '.') {
            value.integer = false;
            advance();
            digits("expected a digit after the decimal point");
        }
        if (!at_end() && (peek() == 'e' || peek() == 'E')) {
            value.integer = false;
            advance();
            if (!at_end() && (peek() == '+' || peek() == '-')) advance();
            digits("expected an exponent digit");
        }
        value.text = std::string(text_.substr(start, pos_ - start));
    }

    unsigned hex4() {
        unsigned unit = 0;
        for (int i = 0; i < 4; ++i) {
            if (at_end()) fail("truncated \\u escape");
            const auto c = peek();
            unsigned nibble = 0;
            if (c >= '0' && c <= '9') nibble = static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') nibble = static_cast<unsigned>(c - 'a') + 10U;
            else if (c >= 'A' && c <= 'F') nibble = static_cast<unsigned>(c - 'A') + 10U;
            else fail("invalid \\u escape");
            unit = unit * 16U + nibble;
            advance();
        }
        return unit;
    }

    static void append_utf8(std::string& out, unsigned code) {
        if (code < 0x80) {
            out += static_cast<char>(code);
        } else if (code < 0x800) {
            out += static_cast<char>(0xC0U | (code >> 6));
            out += static_cast<char>(0x80U | (code & 0x3FU));
        } else if (code < 0x10000) {
            out += static_cast<char>(0xE0U | (code >> 12));
            out += static_cast<char>(0x80U | ((code >> 6) & 0x3FU));
            out += static_cast<char>(0x80U | (code & 0x3FU));
        } else {
            out += static_cast<char>(0xF0U | (code >> 18));
            out += static_cast<char>(0x80U | ((code >> 12) & 0x3FU));
            out += static_cast<char>(0x80U | ((code >> 6) & 0x3FU));
            out += static_cast<char>(0x80U | (code & 0x3FU));
        }
    }

    void escape(std::string& out) {
        advance();
        if (at_end()) fail("unterminated string");
        const auto c = peek();
        const char* simple = nullptr;
        switch (c) {
            case '"':
                simple = "\"";
                break;
            case '\\':
                simple = "\\";
                break;
            case '/':
                simple = "/";
                break;
            case 'b':
                simple = "\b";
                break;
            case 'f':
                simple = "\f";
                break;
            case 'n':
                simple = "\n";
                break;
            case 'r':
                simple = "\r";
                break;
            case 't':
                simple = "\t";
                break;
            case 'u':
                break;
            default:
                fail("invalid escape sequence");
        }
        advance();
        if (simple) {
            out += simple;
            return;
        }
        auto code = hex4();
        if (code >= 0xDC00 && code <= 0xDFFF) fail("unpaired low surrogate");
        if (code >= 0xD800 && code <= 0xDBFF) {
            if (text_.substr(pos_, 2) != "\\u") fail("unpaired high surrogate");
            advance();
            advance();
            const auto low = hex4();
            if (low < 0xDC00 || low > 0xDFFF) fail("unpaired high surrogate");
            code = 0x10000U + ((code - 0xD800U) << 10) + (low - 0xDC00U);
        }
        append_utf8(out, code);
    }

    void utf8_sequence(std::string& out) {
        const auto lead = peek();
        std::size_t length = 0;
        unsigned char low = 0x80, high = 0xBF;
        if (lead >= 0xC2 && lead <= 0xDF) length = 2;
        else if (lead >= 0xE0 && lead <= 0xEF) {
            length = 3;
            if (lead == 0xE0) low = 0xA0;
            if (lead == 0xED) high = 0x9F;  // Excludes encoded surrogates.
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            length = 4;
            if (lead == 0xF0) low = 0x90;
            if (lead == 0xF4) high = 0x8F;
        } else {
            fail("invalid UTF-8");
        }
        out += static_cast<char>(lead);
        advance();
        for (std::size_t i = 1; i < length; ++i) {
            if (at_end()) fail("truncated UTF-8 sequence");
            const auto next = peek();
            if (next < low || next > high) fail("invalid UTF-8");
            low = 0x80;
            high = 0xBF;
            out += static_cast<char>(next);
            advance();
        }
    }

    std::string parse_string() {
        advance();
        std::string out;
        while (true) {
            if (at_end()) fail("unterminated string");
            const auto c = peek();
            if (c == '"') {
                advance();
                return out;
            }
            if (c == '\\') escape(out);
            else if (c < 0x20) fail("control character in string");
            else if (c < 0x80) {
                out += static_cast<char>(c);
                advance();
            } else {
                utf8_sequence(out);
            }
        }
    }
};

}  // namespace

std::optional<std::uint64_t> JsonValue::as_uint64() const {
    if (type != Type::Number || !integer || text.empty() || text[0] == '-') return std::nullopt;
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) return std::nullopt;
    return result;
}

std::optional<std::int64_t> JsonValue::as_int64() const {
    if (type != Type::Number || !integer) return std::nullopt;
    std::int64_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) return std::nullopt;
    return result;
}

std::optional<double> JsonValue::as_double() const {
    if (type != Type::Number) return std::nullopt;
    // The literal already matched the JSON number grammar, which strtod accepts
    // in the "C" locale this program runs under.
    errno = 0;
    char* end = nullptr;
    const double result = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size() || (errno == ERANGE && std::isinf(result)))
        return std::nullopt;
    return result;
}

std::string_view json_type_name(JsonValue::Type type) {
    switch (type) {
        case JsonValue::Type::Null:
            return "null";
        case JsonValue::Type::Boolean:
            return "a boolean";
        case JsonValue::Type::Number:
            return "a number";
        case JsonValue::Type::String:
            return "a string";
        case JsonValue::Type::Array:
            return "an array";
        case JsonValue::Type::Object:
            return "an object";
    }
    return "an unknown value";
}

JsonValue parse_json(std::string_view text, std::string_view origin) {
    return Parser(text, origin).document();
}

void write_json_string(std::ostream& stream, std::string_view text) {
    static constexpr char hex[] = "0123456789abcdef";
    stream << '"';
    for (const char c : text) {
        const auto byte = static_cast<unsigned char>(c);
        switch (c) {
            case '"':
                stream << "\\\"";
                break;
            case '\\':
                stream << "\\\\";
                break;
            case '\b':
                stream << "\\b";
                break;
            case '\f':
                stream << "\\f";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            default:
                if (byte < 0x20) stream << "\\u00" << hex[byte >> 4] << hex[byte & 15U];
                else stream << c;
        }
    }
    stream << '"';
}

void write_json_number(std::ostream& stream, double value) {
    if (!std::isfinite(value)) {
        stream << "null";
        return;
    }
    std::ostringstream text;
    text.imbue(std::locale::classic());
    // Integral values of magnitude below 2^53 print in fixed notation: 200 as 200.
    if (value == std::floor(value) && std::fabs(value) < 9007199254740992.0) {
        text << std::fixed << std::setprecision(0) << value;
    } else {
        for (int precision = 1; precision <= 17; ++precision) {
            text.str("");
            text << std::setprecision(precision) << value;
            if (std::strtod(text.str().c_str(), nullptr) == value) break;
        }
    }
    stream << text.str();
}

}  // namespace npu
