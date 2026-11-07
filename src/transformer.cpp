#include "npu/paths.hpp"

#include <systemc>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Vector = std::vector<float>;
using Sequence = std::vector<Vector>;
using U64 = std::uint64_t;
using namespace sc_core;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "the storage contract requires IEEE-754 binary32");

void diagnostic_report(const sc_report& report, const sc_actions& actions) {
    if (actions & SC_DISPLAY) std::cerr << report.what() << '\n';
    sc_report_handler::default_handler(report, actions & ~static_cast<sc_actions>(SC_DISPLAY));
}

void require(bool okay, const char* message) {
    if (!okay) throw std::invalid_argument(message);
}

std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float from_bits(std::uint32_t value) {
    float result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

// Integer conversion is independent of the host floating-point rounding mode.
std::uint32_t round_shift(std::uint32_t value, unsigned shift) {
    const auto truncated = value >> shift;
    const auto remainder = value & ((std::uint32_t{1} << shift) - 1U);
    const auto halfway = std::uint32_t{1} << (shift - 1U);
    return truncated + static_cast<std::uint32_t>(remainder > halfway ||
                                                  (remainder == halfway && (truncated & 1U)));
}

std::uint16_t to_half(float value) {
    const auto raw = bits(value);
    const auto sign = static_cast<std::uint16_t>((raw >> 16U) & 0x8000U);
    const auto exponent = (raw >> 23U) & 0xffU;
    const auto fraction = raw & 0x7fffffU;
    if (exponent == 0xffU) {
        return static_cast<std::uint16_t>(sign | 0x7c00U |
            (fraction ? ((fraction >> 13U) | 0x200U) : 0U));
    }
    if (exponent == 0) return sign;  // Every binary32 subnormal underflows in binary16.
    const int power = static_cast<int>(exponent) - 127;
    if (power > 15) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (power < -25) return sign;
    const auto significand = fraction | 0x800000U;
    if (power < -14) {
        const auto rounded = round_shift(significand, static_cast<unsigned>(-power - 1));
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const auto rounded = round_shift(significand, 13);
    const auto result = static_cast<std::uint32_t>(power + 14) * 1024U + rounded;
    return static_cast<std::uint16_t>(sign | result);
}

float from_half(std::uint16_t value) {
    const auto sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    const auto exponent = (value >> 10U) & 0x1fU;
    std::uint32_t fraction = value & 0x3ffU;
    if (exponent == 0x1fU)
        return from_bits(sign | 0x7f800000U | (fraction << 13U));
    if (exponent != 0)
        return from_bits(sign | ((static_cast<std::uint32_t>(exponent) + 112U) << 23U) |
                         (fraction << 13U));
    if (fraction == 0) return from_bits(sign);
    int power = -14;
    while ((fraction & 0x400U) == 0) { fraction <<= 1U; --power; }
    return from_bits(sign | (static_cast<std::uint32_t>(power + 127) << 23U) |
                     ((fraction & 0x3ffU) << 13U));
}

std::uint16_t to_bfloat(float value) {
    const auto raw = bits(value);
    if ((raw & 0x7f800000U) == 0x7f800000U && (raw & 0x7fffffU))
        return static_cast<std::uint16_t>((raw >> 16U) | 0x40U);
    return static_cast<std::uint16_t>((raw + 0x7fffU + ((raw >> 16U) & 1U)) >> 16U);
}

enum class Precision { fp32, fp16, bf16 };

float store(float value, Precision precision) {
    if (precision == Precision::fp16) return from_half(to_half(value));
    if (precision == Precision::bf16)
        return from_bits(static_cast<std::uint32_t>(to_bfloat(value)) << 16U);
    return value;
}

// Volatile boundaries specify separate binary32 multiply/add operations and
// prevent a compiler from silently turning these reductions into fused FMAs.
float add(float lhs, float rhs) { volatile float result = lhs + rhs; return result; }
float mul(float lhs, float rhs) { volatile float result = lhs * rhs; return result; }
float div32(float lhs, float rhs) { volatile float result = lhs / rhs; return result; }

struct Config {
    unsigned dim = 16, heads = 4, kv_heads = 2, ff_dim = 32, context = 32;
    unsigned prefill = 3, decode = 3, seed = 1;
    unsigned tensor_rate = 64, vector_rate = 16, dma_rate = 32, dma_startup = 2;
    U64 sram_bytes = 1024U * 1024U;
    Precision precision = Precision::fp16;
    std::string precision_name = "fp16", vcd, progress;
    bool dump = false, lifecycle = false;
    unsigned head_dim() const { return dim / heads; }
    unsigned kv_dim() const { return head_dim() * kv_heads; }
    unsigned storage_bytes() const { return precision == Precision::fp32 ? 4U : 2U; }
    U64 kv_capacity_bytes() const { return U64{2} * context * kv_dim() * storage_bytes(); }
    U64 scratch_bytes() const {
        // Conservative explicitly reserved FP32 scratch, including all heads'
        // score/probability vectors and simultaneously live FFN intermediates.
        return U64{4} * (U64{12} * dim + U64{3} * ff_dim + U64{2} * heads * context);
    }
    U64 required_sram() const { return kv_capacity_bytes() + scratch_bytes(); }
};

struct Counters {
    U64 tensor_macs = 0, vector_ops = 0, tensor_cycles = 0, vector_cycles = 0;
    U64 dma_cycles = 0, dma_read_bytes = 0, dma_write_bytes = 0, dma_transactions = 0;
    U64 kv_read_bytes = 0, kv_write_bytes = 0, tokens = 0, executed_tokens = 0, requests = 0;
};

enum class Kind : unsigned { idle = 0, dma = 1, tensor = 2, vector = 3 };
using Stage = std::function<void(Kind, U64, bool)>;

Vector fixture_input(const Config& c, unsigned position) {
    Vector out(c.dim);
    for (unsigned col = 0; col < c.dim; ++col) {
        const auto integer = static_cast<int>((U64{position} * 19U + U64{col} * 7U +
                                               c.seed % 97U) % 37U) - 18;
        out[col] = store(static_cast<float>(integer) / 16.0F, c.precision);
    }
    return out;
}

Vector weight(const Config& c, unsigned rows, unsigned columns, unsigned matrix) {
    Vector out(static_cast<std::size_t>(rows) * columns);
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned col = 0; col < columns; ++col) {
            const auto integer = static_cast<int>((U64{row} * 17U + U64{col} * 13U +
                                                   c.seed % 101U + matrix * 7U) % 31U) - 15;
            out[static_cast<std::size_t>(row) * columns + col] =
                store(static_cast<float>(integer) / 64.0F, c.precision);
        }
    return out;
}

class Block {
public:
    explicit Block(const Config& config)
        : c(config), q(weight(c, c.dim, c.dim, 1)), k(weight(c, c.dim, c.kv_dim(), 2)),
          v(weight(c, c.dim, c.kv_dim(), 3)), o(weight(c, c.dim, c.dim, 4)),
          gate(weight(c, c.dim, c.ff_dim, 5)), up(weight(c, c.dim, c.ff_dim, 6)),
          down(weight(c, c.ff_dim, c.dim, 7)) {}

    Sequence keys, values;
    const Config& c;
    Vector q, k, v, o, gate, up, down;

    Vector dense(const Vector& input, const Vector& weights, unsigned columns, const Stage& stage) const {
        stage(Kind::dma, static_cast<U64>(weights.size()) * c.storage_bytes(), false);
        stage(Kind::tensor, static_cast<U64>(input.size()) * columns, false);
        Vector out(columns, 0.0F);
        for (unsigned col = 0; col < columns; ++col) {
            float sum = 0;
            for (std::size_t row = 0; row < input.size(); ++row)
                sum = add(sum, mul(input[row], weights[row * columns + col]));
            out[col] = store(sum, c.precision);
        }
        return out;
    }

    Vector rms(const Vector& input, unsigned norm, const Stage& stage) const {
        stage(Kind::dma, U64{c.dim} * c.storage_bytes(), false);
        stage(Kind::vector, U64{5} * c.dim + 2U, false);
        float square_sum = 0;
        for (float value : input) square_sum = add(square_sum, mul(value, value));
        require(std::isfinite(square_sum), "non-finite RMSNorm reduction");
        const float mean = div32(square_sum, static_cast<float>(c.dim));
        const float inverse = div32(1.0F, static_cast<float>(std::sqrt(add(mean, 0.00001F))));
        Vector out(c.dim);
        for (unsigned col = 0; col < c.dim; ++col) {
            const auto integer = static_cast<int>((U64{col} * 3U + c.seed % 101U + norm * 11U) % 9U) - 4;
            const float gamma = store(1.0F + static_cast<float>(integer) / 32.0F, c.precision);
            out[col] = store(mul(mul(input[col], inverse), gamma), c.precision);
        }
        return out;
    }

    void rope(Vector& data, unsigned heads, unsigned position, const Stage& stage) const {
        stage(Kind::vector, U64{heads} * c.head_dim() * 5U, false);
        for (unsigned head = 0; head < heads; ++head)
            for (unsigned pair = 0; pair < c.head_dim(); pair += 2) {
                const double angle = static_cast<double>(position) /
                    std::pow(10000.0, static_cast<double>(pair) / c.head_dim());
                const float cosine = static_cast<float>(std::cos(angle));
                const float sine = static_cast<float>(std::sin(angle));
                const auto at = static_cast<std::size_t>(head) * c.head_dim() + pair;
                const float lhs = data[at], rhs = data[at + 1];
                data[at] = store(add(mul(lhs, cosine), -mul(rhs, sine)), c.precision);
                data[at + 1] = store(add(mul(lhs, sine), mul(rhs, cosine)), c.precision);
            }
    }

    Vector residual(const Vector& lhs, const Vector& rhs, const Stage& stage) const {
        stage(Kind::vector, lhs.size(), false);
        Vector out(lhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) out[i] = store(add(lhs[i], rhs[i]), c.precision);
        return out;
    }

    Vector token(const Vector& input, const Stage& stage, Counters* stats) {
        const auto position = static_cast<unsigned>(keys.size());
        Vector norm = rms(input, 1, stage);
        Vector query = dense(norm, q, c.dim, stage);
        Vector key = dense(norm, k, c.kv_dim(), stage);
        Vector value = dense(norm, v, c.kv_dim(), stage);
        rope(query, c.heads, position, stage);
        rope(key, c.kv_heads, position, stage);
        keys.push_back(std::move(key));
        values.push_back(std::move(value));
        if (stats) {
            stats->kv_write_bytes += U64{2} * c.kv_dim() * c.storage_bytes();
            stats->kv_read_bytes += U64{2} * keys.size() * c.kv_dim() * c.storage_bytes();
        }
        // Only the causal prefix has been committed: no future key is read.
        stage(Kind::tensor, U64{2} * c.heads * keys.size() * c.head_dim(), false);
        stage(Kind::vector, U64{c.heads} * (U64{6} * keys.size() + 1U), false);
        Vector attention(c.dim, 0.0F);
        const float scale = div32(1.0F, static_cast<float>(std::sqrt(static_cast<float>(c.head_dim()))));
        for (unsigned head = 0; head < c.heads; ++head) {
            const unsigned kv_head = head / (c.heads / c.kv_heads);
            Vector scores(keys.size(), 0.0F);
            for (std::size_t past = 0; past < keys.size(); ++past) {
                float score = 0;
                for (unsigned d = 0; d < c.head_dim(); ++d)
                    score = add(score, mul(query[head * c.head_dim() + d],
                                           keys[past][kv_head * c.head_dim() + d]));
                scores[past] = mul(score, scale);
            }
            const float maximum = *std::max_element(scores.begin(), scores.end());
            float denominator = 0;
            for (float& score : scores) {
                score = static_cast<float>(std::exp(static_cast<double>(add(score, -maximum))));
                denominator = add(denominator, score);
            }
            for (float& score : scores) score = div32(score, denominator);
            for (unsigned d = 0; d < c.head_dim(); ++d) {
                float accumulated = 0;
                for (std::size_t past = 0; past < keys.size(); ++past)
                    accumulated = add(accumulated, mul(scores[past], values[past][kv_head * c.head_dim() + d]));
                attention[head * c.head_dim() + d] = store(accumulated, c.precision);
            }
        }
        Vector first = residual(input, dense(attention, o, c.dim, stage), stage);
        norm = rms(first, 2, stage);
        Vector gated = dense(norm, gate, c.ff_dim, stage);
        const Vector upper = dense(norm, up, c.ff_dim, stage);
        stage(Kind::vector, U64{5} * c.ff_dim, false);
        for (unsigned i = 0; i < c.ff_dim; ++i) {
            const float exponential = static_cast<float>(std::exp(-static_cast<double>(gated[i])));
            const float silu = div32(gated[i], add(1.0F, exponential));
            gated[i] = store(mul(silu, upper[i]), c.precision);
        }
        Vector output = residual(first, dense(gated, down, c.dim, stage), stage);
        for (float element : output) require(std::isfinite(element), "non-finite transformer output");
        return output;
    }
};

struct Request {
    Sequence inputs, outputs;
    bool reset = false, okay = false;
    std::string error;
    sc_event done;
};

class Engine final : public sc_module {
public:
    sc_fifo<Request*> commands{1};
    sc_signal<unsigned> stage_trace{"stage"}, token_trace{"tokens_completed"}, kv_trace{"kv_tokens"};
    sc_signal<unsigned> executed_trace{"tokens_executed"};
    sc_signal<bool> busy_trace{"busy"};
    Counters stats;
    unsigned rejections = 0, resets = 0;
    Block block;
    Engine(sc_module_name name, const Config& config, std::ostream* progress)
        : sc_module(name), block(config), c(config), progress_(progress) {
        SC_THREAD(run);
    }

private:
    const Config& c;
    std::ostream* progress_;

    void emit(const char* event, Kind kind) {
        if (!progress_) return;
        *progress_ << "{\"event\":\"" << event << "\",\"stage\":" << static_cast<unsigned>(kind)
            << ",\"cycles\":" << static_cast<U64>(sc_time_stamp() / sc_time(1, SC_NS))
            << ",\"tokens_completed\":" << stats.tokens << ",\"tokens_executed\":" << stats.executed_tokens
            << ",\"kv_tokens\":" << kv_trace.read()
            << ",\"tensor_macs\":" << stats.tensor_macs << ",\"vector_ops\":" << stats.vector_ops
            << ",\"dma_read_bytes\":" << stats.dma_read_bytes
            << ",\"dma_write_bytes\":" << stats.dma_write_bytes << "}\n" << std::flush;
        if (!*progress_) throw std::runtime_error("could not write progress stream");
    }

    void stage(Kind kind, U64 amount, bool writing) {
        U64 cycles = 0;
        if (kind == Kind::dma) {
            cycles = c.dma_startup + (amount + c.dma_rate - 1U) / c.dma_rate;
            stats.dma_cycles += cycles;
            ++stats.dma_transactions;
            if (writing) stats.dma_write_bytes += amount;
            else stats.dma_read_bytes += amount;
        } else if (kind == Kind::tensor) {
            stats.tensor_macs += amount;
            cycles = (amount + c.tensor_rate - 1U) / c.tensor_rate;
            stats.tensor_cycles += cycles;
        } else if (kind == Kind::vector) {
            stats.vector_ops += amount;
            cycles = (amount + c.vector_rate - 1U) / c.vector_rate;
            stats.vector_cycles += cycles;
        }
        stage_trace.write(static_cast<unsigned>(kind));
        if (cycles) wait(sc_time(static_cast<double>(cycles), SC_NS));
        emit("stage_complete", kind);
    }

    void run() {
        while (true) {
            Request* request = nullptr;
            commands.read(request);
            if (!request) throw std::runtime_error("null internal request");
            request->okay = false;
            request->outputs.clear();
            if (request->reset) {
                block.keys.clear(); block.values.clear();
                kv_trace.write(0); ++resets;
                request->okay = true;
            } else {
                try {
                    // Validate the entire request before timing, DMA, or cache mutation.
                    require(!request->inputs.empty(), "empty inference request");
                    require(request->inputs.size() <= c.context - block.keys.size(), "KV cache capacity exceeded");
                    for (const auto& input : request->inputs) {
                        require(input.size() == c.dim, "input width mismatch");
                        for (float value : input) require(std::isfinite(value), "non-finite input");
                    }
                    // Numerical failures also leave cache content and output publication atomic.
                    Block candidate = block;
                    Sequence outputs;
                    outputs.reserve(request->inputs.size());
                    busy_trace.write(true);
                    stage(Kind::dma, static_cast<U64>(request->inputs.size()) * c.dim * c.storage_bytes(), false);
                    const Stage timed = [this](Kind kind, U64 amount, bool write) { stage(kind, amount, write); };
                    for (const auto& input : request->inputs) {
                        outputs.push_back(candidate.token(input, timed, &stats));
                        ++stats.executed_tokens;
                        executed_trace.write(static_cast<unsigned>(stats.executed_tokens));
                        // Completed arithmetic is observable, but remains tentative
                        // until every token in the request has succeeded.
                        wait(SC_ZERO_TIME);
                        emit("token_executed", Kind::idle);
                    }
                    stage(Kind::dma, static_cast<U64>(outputs.size()) * c.dim * c.storage_bytes(), true);
                    block.keys = std::move(candidate.keys); block.values = std::move(candidate.values);
                    request->outputs = std::move(outputs);
                    stats.tokens += request->inputs.size();
                    token_trace.write(static_cast<unsigned>(stats.tokens));
                    kv_trace.write(static_cast<unsigned>(block.keys.size()));
                    request->okay = true; ++stats.requests;
                } catch (const std::exception& error) {
                    request->error = error.what(); ++rejections;
                    kv_trace.write(static_cast<unsigned>(block.keys.size()));
                }
                if (request->okay) {
                    wait(SC_ZERO_TIME);
                    emit("request_commit", Kind::idle);
                }
            }
            stage_trace.write(0); busy_trace.write(false);
            request->done.notify(SC_ZERO_TIME);
        }
    }
};

class Driver final : public sc_module {
public:
    Sequence outputs;
    Counters workload;
    U64 prefill_cycles = 0, decode_cycles = 0, workload_cycles = 0;
    bool completed = false, partition_match = false, lifecycle_verified = false, rollback_verified = false;
    U64 rejected_work_tokens = 0, rejected_work_cycles = 0;
    std::string failure;
    Driver(sc_module_name name, Engine& engine, const Config& config)
        : sc_module(name), e(engine), c(config) { SC_THREAD(run); }

private:
    Engine& e;
    const Config& c;
    void submit(Request& request) { e.commands.write(&request); wait(request.done); }
    U64 now() const { return static_cast<U64>(sc_time_stamp() / sc_time(1, SC_NS)); }

    void run() {
        try {
            Request prefill;
            for (unsigned pos = 0; pos < c.prefill; ++pos) prefill.inputs.push_back(fixture_input(c, pos));
            submit(prefill);
            require(prefill.okay, "prefill failed");
            outputs = prefill.outputs; prefill_cycles = now();
            for (unsigned pos = c.prefill; pos < c.prefill + c.decode; ++pos) {
                Request decode; decode.inputs.push_back(fixture_input(c, pos)); submit(decode);
                require(decode.okay, "decode failed");
                outputs.push_back(decode.outputs.front());
            }
            workload_cycles = now(); decode_cycles = workload_cycles - prefill_cycles;
            workload = e.stats;

            // Partition equivalence is a structural check, not an independent
            // numerical oracle. The black-box Python tests supply that oracle.
            Block reference(c);
            const Stage untimed = [](Kind, U64, bool) {};
            Sequence expected;
            for (unsigned pos = 0; pos < c.prefill + c.decode; ++pos)
                expected.push_back(reference.token(fixture_input(c, pos), untimed, nullptr));
            partition_match = expected == outputs;
            require(partition_match, "prefill/decode partition mismatch");

            if (c.lifecycle) {
                const Sequence saved_keys = e.block.keys, saved_values = e.block.values;
                const auto saved_time = now();
                Request overflow;
                overflow.inputs.resize(c.context - e.block.keys.size() + 1U, Vector(c.dim, 0));
                submit(overflow);
                require(!overflow.okay && overflow.outputs.empty(), "overflow was not rejected atomically");
                Request empty; submit(empty);
                require(!empty.okay, "empty input was accepted");
                // The malformed width/NaN requests are tested after reset so
                // capacity validation cannot conceal their input validation.
                require(saved_keys == e.block.keys && saved_values == e.block.values && now() == saved_time,
                        "invalid request modified state or simulation time");
                Request reset; reset.reset = true; submit(reset);
                require(reset.okay && e.block.keys.empty() && e.block.values.empty(), "cache reset failed");
                Request malformed; malformed.inputs.push_back(Vector(c.dim + 1U, 0)); submit(malformed);
                require(!malformed.okay && e.block.keys.empty(), "malformed width accepted");
                Request invalid; invalid.inputs.push_back(Vector(c.dim, std::numeric_limits<float>::quiet_NaN()));
                submit(invalid);
                require(!invalid.okay && e.block.keys.empty(), "non-finite input accepted");
                const auto committed_before = e.stats.tokens, executed_before = e.stats.executed_tokens;
                const auto requests_before = e.stats.requests, work_start = now();
                const auto macs_before = e.stats.tensor_macs, vector_before = e.stats.vector_ops;
                Request mixed;
                const U64 executable_prefix = c.context >= 2 ? 1U : 0U;
                if (executable_prefix) mixed.inputs.push_back(fixture_input(c, 0));
                mixed.inputs.push_back(Vector(c.dim, std::numeric_limits<float>::max()));
                submit(mixed);
                require(!mixed.okay && mixed.outputs.empty() && e.block.keys.empty() && e.block.values.empty(),
                        "partially executed rejection committed output or cache");
                require(e.stats.tokens == committed_before && e.stats.requests == requests_before &&
                        e.token_trace.read() == committed_before && e.kv_trace.read() == 0,
                        "partially executed rejection published committed tokens");
                require(e.stats.executed_tokens == executed_before + executable_prefix &&
                        (executable_prefix == 0 || e.stats.tensor_macs > macs_before) &&
                        e.stats.vector_ops > vector_before && now() > work_start,
                        "rejected execution did not retain actual work accounting");
                rejected_work_tokens = e.stats.executed_tokens - executed_before;
                rejected_work_cycles = now() - work_start;
                rollback_verified = true;
                Request replay;
                for (unsigned pos = 0; pos < c.prefill + c.decode; ++pos)
                    replay.inputs.push_back(fixture_input(c, pos));
                submit(replay);
                require(replay.okay && replay.outputs == outputs && e.block.keys == saved_keys &&
                        e.block.values == saved_values, "reset/replay or full-batch equivalence failed");
                lifecycle_verified = true;
            }
            completed = true;
        } catch (const std::exception& error) { failure = error.what(); }
        wait(SC_ZERO_TIME);
        sc_stop();
    }
};

U64 parse_unsigned(const std::string& text) {
    U64 result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(), "expected unsigned decimal integer");
    return result;
}

unsigned bounded(const std::string& text, unsigned minimum, unsigned maximum) {
    const auto value = parse_unsigned(text);
    require(value >= minimum && value <= maximum, "argument outside supported bounds");
    return static_cast<unsigned>(value);
}

void print_sequence(const Sequence& outputs) {
    std::cout << '[';
    for (std::size_t row = 0; row < outputs.size(); ++row) {
        if (row) std::cout << ',';
        std::cout << '[';
        for (std::size_t col = 0; col < outputs[row].size(); ++col) {
            if (col) std::cout << ',';
            std::cout << outputs[row][col];
        }
        std::cout << ']';
    }
    std::cout << ']';
}

void rounding_probe(const std::string& text, Precision precision) {
    require(!text.empty() && text.size() <= 65536, "rounding probe requires 1..4096 decimal binary32 words");
    std::vector<std::uint32_t> words;
    std::size_t start = 0;
    do {
        const auto comma = text.find(',', start);
        const auto token = text.substr(start, comma == std::string::npos ? comma : comma - start);
        const auto word = parse_unsigned(token);
        require(word <= std::numeric_limits<std::uint32_t>::max(), "binary32 word out of range");
        require(words.size() < 4096, "rounding probe exceeds 4096 values");
        words.push_back(bits(store(from_bits(static_cast<std::uint32_t>(word)), precision)));
        if (comma == std::string::npos) break;
        start = comma + 1;
    } while (true);
    std::cout << "{\"rounded_f32_bits\":[";
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << words[i];
    }
    std::cout << "]}\n";
}
}  // namespace

int sc_main(int argc, char* argv[]) {
    try {
        Config c;
        std::string probe;
        for (int i = 1; i < argc; ++i) {
            const std::string flag(argv[i]);
            if (flag == "--help") {
                std::cout << "Usage: npu-transformer-demo [--dim 2..128] [--heads 1..16] [--kv-heads 1..16]\n"
                    " [--ff-dim 1..256] [--context 1..512] [--prefill 1..128] [--decode 0..128]\n"
                    " [--precision fp32|fp16|bf16] [--seed UINT32] [--dump-output] [--verify-lifecycle]\n"
                    " [--tensor-macs-per-cycle 1..4096] [--vector-ops-per-cycle 1..4096]\n"
                    " [--dma-bytes-per-cycle 1..4096] [--dma-startup 0..1000] [--sram-bytes 1..16777216]\n"
                    " [--vcd BASENAME] [--progress-jsonl PATH] [--round-f32-bits COMMA_SEPARATED_UINT32]\n";
                return 0;
            }
            if (flag == "--dump-output") { c.dump = true; continue; }
            if (flag == "--verify-lifecycle") { c.lifecycle = true; continue; }
            require(i + 1 < argc, "missing argument value");
            const std::string value(argv[++i]);
            if (flag == "--dim") c.dim = bounded(value, 2, 128);
            else if (flag == "--heads") c.heads = bounded(value, 1, 16);
            else if (flag == "--kv-heads") c.kv_heads = bounded(value, 1, 16);
            else if (flag == "--ff-dim") c.ff_dim = bounded(value, 1, 256);
            else if (flag == "--context") c.context = bounded(value, 1, 512);
            else if (flag == "--prefill") c.prefill = bounded(value, 1, 128);
            else if (flag == "--decode") c.decode = bounded(value, 0, 128);
            else if (flag == "--seed") c.seed = bounded(value, 0, std::numeric_limits<unsigned>::max());
            else if (flag == "--tensor-macs-per-cycle") c.tensor_rate = bounded(value, 1, 4096);
            else if (flag == "--vector-ops-per-cycle") c.vector_rate = bounded(value, 1, 4096);
            else if (flag == "--dma-bytes-per-cycle") c.dma_rate = bounded(value, 1, 4096);
            else if (flag == "--dma-startup") c.dma_startup = bounded(value, 0, 1000);
            else if (flag == "--sram-bytes") c.sram_bytes = bounded(value, 1, 16777216);
            else if (flag == "--vcd") { require(!value.empty(), "empty VCD basename"); c.vcd = value; }
            else if (flag == "--progress-jsonl") { require(!value.empty(), "empty progress path"); c.progress = value; }
            else if (flag == "--round-f32-bits") { require(!value.empty(), "empty rounding probe"); probe = value; }
            else if (flag == "--precision") {
                c.precision_name = value;
                if (value == "fp32") c.precision = Precision::fp32;
                else if (value == "fp16") c.precision = Precision::fp16;
                else if (value == "bf16") c.precision = Precision::bf16;
                else throw std::invalid_argument("precision must be fp32, fp16, or bf16");
            } else throw std::invalid_argument("unknown argument: " + flag);
        }
        if (!probe.empty()) { rounding_probe(probe, c.precision); return 0; }
        require(c.dim % c.heads == 0 && c.head_dim() % 2 == 0, "dim must be divisible by heads with even head width for RoPE");
        require(c.heads % c.kv_heads == 0, "kv-heads must divide heads");
        require(c.prefill + c.decode <= c.context, "prefill plus decode exceeds KV cache capacity");
        require(c.required_sram() <= c.sram_bytes, "SRAM budget cannot hold reserved KV cache and FP32 scratch");
        // SystemC appends .vcd internally; accept an optional suffix without
        // appending it twice, and compare the actual destination before opening
        // either artifact. Equivalence also detects symlink and hard-link aliases.
        if (!c.vcd.empty() && c.vcd.size() > 4U && c.vcd.compare(c.vcd.size() - 4U, 4U, ".vcd") == 0)
            c.vcd.resize(c.vcd.size() - 4U);
        if (!c.vcd.empty() && !c.progress.empty())
            require(!npu::same_file_path(c.vcd + ".vcd", c.progress), "VCD and progress output paths conflict");
        sc_report_handler::set_handler(diagnostic_report);
        sc_report_handler::set_actions("/OSCI/SystemC", SC_INFO, SC_DO_NOTHING);
        sc_report_handler::set_actions("tracing timescale unit set", SC_INFO, SC_DO_NOTHING);
        sc_set_time_resolution(1, SC_PS);
        std::ofstream progress;
        if (!c.progress.empty()) {
            progress.open(c.progress, std::ios::out | std::ios::trunc);
            require(progress.good(), "cannot open progress JSONL file");
        }
        Engine engine("decoder", c, c.progress.empty() ? nullptr : &progress);
        Driver driver("host", engine, c);
        sc_trace_file* trace = nullptr;
        if (!c.vcd.empty()) {
            trace = sc_create_vcd_trace_file(c.vcd.c_str());
            require(trace != nullptr, "cannot create VCD trace");
            trace->set_time_unit(1, SC_PS);
            trace->delta_cycles(true);
            sc_trace(trace, engine.stage_trace, "decoder_stage");
            sc_trace(trace, engine.busy_trace, "decoder_busy");
            sc_trace(trace, engine.token_trace, "tokens_completed");
            sc_trace(trace, engine.executed_trace, "tokens_executed");
            sc_trace(trace, engine.kv_trace, "kv_tokens");
        }
        sc_start(sc_time(1, SC_SEC));
        if (trace) sc_close_vcd_trace_file(trace);
        if (!driver.failure.empty()) throw std::runtime_error(driver.failure);
        require(driver.completed, "simulation watchdog expired");
        const auto& s = driver.workload;
        const U64 live_bytes = U64{2} * (c.prefill + c.decode) * c.kv_dim() * c.storage_bytes();
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
            << "{\"schema_version\":1,\"model\":\"systemc-mobile-transformer-block\",\"precision\":\""
            << c.precision_name << "\",\"seed\":" << c.seed << ",\"dim\":" << c.dim
            << ",\"heads\":" << c.heads << ",\"kv_heads\":" << c.kv_heads << ",\"ff_dim\":" << c.ff_dim
            << ",\"context_capacity\":" << c.context << ",\"prefill_tokens\":" << c.prefill
            << ",\"decode_tokens\":" << c.decode << ",\"tokens_completed\":" << s.tokens
            << ",\"tokens_executed\":" << s.executed_tokens
            << ",\"requests_completed\":" << s.requests << ",\"causal\":true,\"rope\":true"
            << ",\"fp32_accumulation\":true,\"partition_match\":" << (driver.partition_match ? "true" : "false")
            << ",\"lifecycle_verified\":" << (driver.lifecycle_verified ? "true" : "false")
            << ",\"transaction_rollback_verified\":" << (driver.rollback_verified ? "true" : "false")
            << ",\"rejected_work_tokens\":" << driver.rejected_work_tokens
            << ",\"rejected_work_cycles\":" << driver.rejected_work_cycles
            << ",\"cache_resets\":" << engine.resets << ",\"rejected_requests\":" << engine.rejections
            << ",\"workload_cycles\":" << driver.workload_cycles
            << ",\"prefill_cycles\":" << driver.prefill_cycles << ",\"decode_cycles\":" << driver.decode_cycles
            << ",\"simulation_cycles\":" << static_cast<U64>(sc_time_stamp() / sc_time(1, SC_NS))
            << ",\"tensor_macs\":" << s.tensor_macs << ",\"tensor_cycles\":" << s.tensor_cycles
            << ",\"vector_ops\":" << s.vector_ops << ",\"vector_cycles\":" << s.vector_cycles
            << ",\"dma_cycles\":" << s.dma_cycles << ",\"dma_transactions\":" << s.dma_transactions
            << ",\"dma_read_bytes\":" << s.dma_read_bytes << ",\"dma_write_bytes\":" << s.dma_write_bytes
            << ",\"kv_read_bytes\":" << s.kv_read_bytes << ",\"kv_write_bytes\":" << s.kv_write_bytes
            << ",\"kv_live_bytes\":" << live_bytes << ",\"kv_capacity_bytes\":" << c.kv_capacity_bytes()
            << ",\"scratch_bytes\":" << c.scratch_bytes() << ",\"reserved_sram_bytes\":" << c.required_sram()
            << ",\"sram_budget_bytes\":" << c.sram_bytes << ",\"outputs_finite\":true";
        if (c.dump) { std::cout << ",\"outputs\":"; print_sequence(driver.outputs); }
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "npu-transformer-demo: " << error.what() << '\n';
        return 2;
    }
}
