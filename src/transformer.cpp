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
#include <iterator>
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
              "fp32 storage requires IEEE-754 binary32");

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

// Round-half-to-even right shift in integer arithmetic, independent of the
// host floating-point rounding mode.
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

// volatile keeps every binary32 multiply and add separate (no FMA contraction),
// as the Python oracle evaluates them.
float add(float lhs, float rhs) { volatile float result = lhs + rhs; return result; }
float mul(float lhs, float rhs) { volatile float result = lhs * rhs; return result; }
float div32(float lhs, float rhs) { volatile float result = lhs / rhs; return result; }

struct Config {
    unsigned dim = 16, heads = 4, kv_heads = 2, ff_dim = 32, context = 32, layers = 1;
    unsigned prefill = 3, decode = 3, seed = 1;
    unsigned tensor_rate = 64, vector_rate = 16, dma_rate = 32, dma_startup = 2;
    U64 sram_bytes = 1024U * 1024U;
    Precision precision = Precision::fp16;
    std::string precision_name = "fp16", vcd, progress;
    bool dump = false, lifecycle = false, inject_fault = false;
    unsigned fault_row = 0;
    unsigned head_dim() const { return dim / heads; }
    unsigned kv_dim() const { return head_dim() * kv_heads; }
    unsigned storage_bytes() const { return precision == Precision::fp32 ? 4U : 2U; }
    // DRAM holds K and V rows for every layer and position.
    U64 kv_dram_capacity_bytes() const {
        return U64{2} * layers * context * kv_dim() * storage_bytes();
    }
    // Largest request the driver submits: the prompt is one batch, and the
    // lifecycle replay resubmits the workload's own partition.
    unsigned batch_rows() const { return prefill; }
    U64 layer_weight_count() const {
        return U64{2} * dim * dim + U64{2} * dim * kv_dim() + U64{3} * dim * ff_dim;
    }
    // Projections, plus QK^T and PV over p + 1 visible keys at position p:
    // 2 * dim * (p + 1), summed over p = 0..T-1.
    U64 workload_macs() const {
        const U64 tokens = U64{prefill} + decode;
        return U64{layers} * (tokens * layer_weight_count() + U64{dim} * tokens * (tokens + 1));
    }
    // Live per-row tensors at storage precision, at the peak of each half of a
    // layer: x, Q, K, V and the normalized input or the attention output during
    // attention (3d + 2kv); the residual, normalized input, gate and up during
    // the FFN (2d + 2ff).
    U64 activation_buffer_bytes() const {
        const U64 attention = U64{3} * dim + U64{2} * kv_dim(), ffn = U64{2} * dim + U64{2} * ff_dim;
        return U64{batch_rows()} * storage_bytes() * std::max(attention, ffn);
    }
    // FP32 scores and probabilities of every head for one query row.
    U64 score_buffer_bytes() const { return U64{4} * 2 * heads * context; }
    // One layer's past K and V rows, streamed in from DRAM for attention.
    U64 kv_buffer_bytes() const { return U64{2} * context * kv_dim() * storage_bytes(); }
    U64 fixed_sram_bytes() const {
        return activation_buffer_bytes() + score_buffer_bytes() + kv_buffer_bytes();
    }
    // Q and O are dim x dim, K and V dim x kv_dim, gate and up dim x ff, down
    // ff x dim; a column of down holds ff values, a column of the others dim.
    U64 largest_matrix_bytes() const { return U64{dim} * std::max(dim, ff_dim) * storage_bytes(); }
    U64 weight_column_bytes() const { return U64{std::max(dim, ff_dim)} * storage_bytes(); }
    // Weights stream in column panels through a buffer that takes the SRAM
    // the other buffers leave, up to the largest matrix.
    U64 weight_buffer_bytes() const {
        const auto fixed = fixed_sram_bytes();
        return fixed >= sram_bytes ? 0 : std::min(largest_matrix_bytes(), sram_bytes - fixed);
    }
    U64 required_sram() const { return fixed_sram_bytes() + weight_buffer_bytes(); }
};

struct Counters {
    U64 tensor_macs = 0, vector_ops = 0, tensor_cycles = 0, vector_cycles = 0;
    U64 dma_cycles = 0, dma_read_bytes = 0, dma_write_bytes = 0, dma_transactions = 0;
    U64 weight_read_bytes = 0, kv_dram_read_bytes = 0, kv_dram_write_bytes = 0;
    U64 tokens = 0, executed_tokens = 0, requests = 0;
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

// Fixture weights are integers in -15..15 over 8 * 2^e, where 4^e <= rows < 4^(e+1)
// and e >= 3: the divisor is 64 below 256 rows and doubles per factor of four in
// the reduction depth, so dot products over 2048 or 8192 rows keep fp16
// activations finite. Power-of-two divisors keep every weight exact in fp16 and
// bf16.
float weight_divisor(unsigned rows) {
    unsigned exponent = 3;
    while ((U64{4} << (2U * exponent)) <= rows) ++exponent;
    return 8.0F * static_cast<float>(1U << exponent);
}

Vector weight(const Config& c, unsigned rows, unsigned columns, unsigned matrix) {
    Vector out(static_cast<std::size_t>(rows) * columns);
    const float divisor = weight_divisor(rows);
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned col = 0; col < columns; ++col) {
            const auto integer = static_cast<int>((U64{row} * 17U + U64{col} * 13U +
                                                   c.seed % 101U + U64{matrix} * 7U) % 31U) - 15;
            out[static_cast<std::size_t>(row) * columns + col] =
                store(static_cast<float>(integer) / divisor, c.precision);
        }
    return out;
}

// One layer's K and V rows in DRAM, one row per position, at storage precision.
struct KvCache {
    Sequence keys, values;
};

struct Layer {
    unsigned index = 0;
    Vector q, k, v, o, gate, up, down;
};

// Weights live in DRAM and stream through the DMA once per layer per request,
// so a batch of S rows pays for each weight byte once.
class Model {
public:
    explicit Model(const Config& config) : c(config) {
        for (unsigned l = 0; l < c.layers; ++l) {
            const unsigned base = 7 * l;
            layers_.push_back({l, weight(c, c.dim, c.dim, base + 1),
                               weight(c, c.dim, c.kv_dim(), base + 2),
                               weight(c, c.dim, c.kv_dim(), base + 3),
                               weight(c, c.dim, c.dim, base + 4),
                               weight(c, c.dim, c.ff_dim, base + 5),
                               weight(c, c.dim, c.ff_dim, base + 6),
                               weight(c, c.ff_dim, c.dim, base + 7)});
        }
    }

    // Runs the rows at positions cache[l].keys.size() onwards through every
    // layer and appends their K/V rows to the cache.
    Sequence forward(const Sequence& inputs, std::vector<KvCache>& cache, const Stage& stage,
                     Counters* stats) const {
        Sequence x = inputs;
        for (const auto& layer : layers_) x = block(layer, x, cache[layer.index], stage, stats);
        for (const auto& row : x)
            for (float element : row) require(std::isfinite(element), "non-finite transformer output");
        return x;
    }

private:
    const Config& c;
    std::vector<Layer> layers_;

    // DRAM holds each matrix packed by column panel, as many columns as the
    // weight buffer holds; each panel's DMA precedes its tensor stage. Row-major
    // weights: the output accumulators of one input row advance together over
    // the reduction, each in the order 0..rows-1, whatever the panel split.
    Sequence dense(const Sequence& inputs, const Vector& weights, unsigned columns, const Stage& stage,
                   Counters* stats) const {
        const auto depth = inputs.front().size();
        const auto column_bytes = static_cast<U64>(depth) * c.storage_bytes();
        const auto panel = std::min<U64>(columns, c.weight_buffer_bytes() / column_bytes);
        require(panel > 0, "the weight buffer cannot hold one weight column");
        for (U64 first = 0; first < columns; first += panel) {
            const auto width = std::min<U64>(panel, columns - first);
            stage(Kind::dma, width * column_bytes, false);
            if (stats) stats->weight_read_bytes += width * column_bytes;
            stage(Kind::tensor, static_cast<U64>(inputs.size()) * depth * width, false);
        }
        Sequence out;
        out.reserve(inputs.size());
        Vector sums(columns);
        for (const auto& input : inputs) {
            std::fill(sums.begin(), sums.end(), 0.0F);
            for (std::size_t row = 0; row < depth; ++row) {
                const float* line = weights.data() + row * columns;
                for (unsigned col = 0; col < columns; ++col)
                    sums[col] = add(sums[col], mul(input[row], line[col]));
            }
            Vector result(columns);
            for (unsigned col = 0; col < columns; ++col) result[col] = store(sums[col], c.precision);
            out.push_back(std::move(result));
        }
        return out;
    }

    Sequence rms(const Sequence& inputs, unsigned norm, const Stage& stage, Counters* stats) const {
        const auto bytes = U64{c.dim} * c.storage_bytes();
        stage(Kind::dma, bytes, false);
        if (stats) stats->weight_read_bytes += bytes;
        stage(Kind::vector, static_cast<U64>(inputs.size()) * (U64{5} * c.dim + 2U), false);
        Vector gamma(c.dim);
        for (unsigned col = 0; col < c.dim; ++col) {
            const auto integer = static_cast<int>((U64{col} * 3U + c.seed % 101U + U64{norm} * 11U) % 9U) - 4;
            gamma[col] = store(1.0F + static_cast<float>(integer) / 32.0F, c.precision);
        }
        Sequence out;
        out.reserve(inputs.size());
        for (const auto& input : inputs) {
            float square_sum = 0;
            for (float value : input) square_sum = add(square_sum, mul(value, value));
            require(std::isfinite(square_sum), "non-finite RMSNorm reduction");
            const float mean = div32(square_sum, static_cast<float>(c.dim));
            const float inverse = div32(1.0F, static_cast<float>(std::sqrt(add(mean, 0.00001F))));
            Vector row(c.dim);
            for (unsigned col = 0; col < c.dim; ++col)
                row[col] = store(mul(mul(input[col], inverse), gamma[col]), c.precision);
            out.push_back(std::move(row));
        }
        return out;
    }

    void rope(Sequence& data, unsigned heads, unsigned start, const Stage& stage) const {
        stage(Kind::vector, static_cast<U64>(data.size()) * heads * c.head_dim() * 5U, false);
        for (std::size_t r = 0; r < data.size(); ++r) {
            const auto position = start + static_cast<unsigned>(r);
            for (unsigned head = 0; head < heads; ++head)
                for (unsigned pair = 0; pair < c.head_dim(); pair += 2) {
                    const double angle = static_cast<double>(position) /
                        std::pow(10000.0, static_cast<double>(pair) / c.head_dim());
                    const float cosine = static_cast<float>(std::cos(angle));
                    const float sine = static_cast<float>(std::sin(angle));
                    const auto at = static_cast<std::size_t>(head) * c.head_dim() + pair;
                    const float lhs = data[r][at], rhs = data[r][at + 1];
                    data[r][at] = store(add(mul(lhs, cosine), -mul(rhs, sine)), c.precision);
                    data[r][at + 1] = store(add(mul(lhs, sine), mul(rhs, cosine)), c.precision);
                }
        }
    }

    Sequence residual(const Sequence& lhs, const Sequence& rhs, const Stage& stage) const {
        stage(Kind::vector, static_cast<U64>(lhs.size()) * c.dim, false);
        Sequence out = lhs;
        for (std::size_t r = 0; r < lhs.size(); ++r)
            for (std::size_t i = 0; i < lhs[r].size(); ++i)
                out[r][i] = store(add(lhs[r][i], rhs[r][i]), c.precision);
        return out;
    }

    // New K/V rows are written to DRAM; the rows of earlier requests are read
    // back. The batch's own rows stay on chip for its attention.
    void exchange_kv(KvCache& cache, Sequence keys, Sequence values, const Stage& stage,
                     Counters* stats) const {
        const auto row_bytes = U64{c.kv_dim()} * c.storage_bytes();
        const auto past = static_cast<U64>(cache.keys.size()), rows = static_cast<U64>(keys.size());
        stage(Kind::dma, rows * row_bytes, true);
        stage(Kind::dma, rows * row_bytes, true);
        if (past) {
            stage(Kind::dma, past * row_bytes, false);
            stage(Kind::dma, past * row_bytes, false);
        }
        if (stats) {
            stats->kv_dram_write_bytes += 2 * rows * row_bytes;
            stats->kv_dram_read_bytes += 2 * past * row_bytes;
        }
        for (auto& row : keys) cache.keys.push_back(std::move(row));
        for (auto& row : values) cache.values.push_back(std::move(row));
    }

    // Causal: the query at position p reads keys and values 0..p only.
    Sequence attend(const Sequence& query, const KvCache& cache, unsigned start, const Stage& stage) const {
        U64 macs = 0, ops = 0;
        for (std::size_t r = 0; r < query.size(); ++r) {
            const U64 visible = start + r + 1;
            macs += U64{2} * c.heads * visible * c.head_dim();
            ops += U64{c.heads} * (U64{6} * visible + 1U);
        }
        stage(Kind::tensor, macs, false);
        stage(Kind::vector, ops, false);
        const float scale = div32(1.0F, static_cast<float>(std::sqrt(static_cast<float>(c.head_dim()))));
        Sequence out;
        out.reserve(query.size());
        for (std::size_t r = 0; r < query.size(); ++r) {
            const std::size_t visible = start + r + 1;
            Vector attention(c.dim, 0.0F);
            for (unsigned head = 0; head < c.heads; ++head) {
                const unsigned kv_head = head / (c.heads / c.kv_heads);
                Vector scores(visible, 0.0F);
                for (std::size_t past = 0; past < visible; ++past) {
                    float score = 0;
                    for (unsigned d = 0; d < c.head_dim(); ++d)
                        score = add(score, mul(query[r][head * c.head_dim() + d],
                                               cache.keys[past][kv_head * c.head_dim() + d]));
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
                    for (std::size_t past = 0; past < visible; ++past)
                        accumulated = add(accumulated,
                                          mul(scores[past], cache.values[past][kv_head * c.head_dim() + d]));
                    attention[head * c.head_dim() + d] = store(accumulated, c.precision);
                }
            }
            out.push_back(std::move(attention));
        }
        return out;
    }

    Sequence block(const Layer& layer, const Sequence& x, KvCache& cache, const Stage& stage,
                   Counters* stats) const {
        const auto start = static_cast<unsigned>(cache.keys.size());
        Sequence norm = rms(x, 2 * layer.index + 1, stage, stats);
        Sequence query = dense(norm, layer.q, c.dim, stage, stats);
        Sequence key = dense(norm, layer.k, c.kv_dim(), stage, stats);
        Sequence value = dense(norm, layer.v, c.kv_dim(), stage, stats);
        rope(query, c.heads, start, stage);
        rope(key, c.kv_heads, start, stage);
        exchange_kv(cache, std::move(key), std::move(value), stage, stats);
        const Sequence attention = attend(query, cache, start, stage);
        const Sequence first = residual(x, dense(attention, layer.o, c.dim, stage, stats), stage);
        norm = rms(first, 2 * layer.index + 2, stage, stats);
        Sequence gated = dense(norm, layer.gate, c.ff_dim, stage, stats);
        const Sequence upper = dense(norm, layer.up, c.ff_dim, stage, stats);
        stage(Kind::vector, static_cast<U64>(x.size()) * 5U * c.ff_dim, false);
        for (std::size_t r = 0; r < gated.size(); ++r)
            for (unsigned i = 0; i < c.ff_dim; ++i) {
                const float exponential = static_cast<float>(std::exp(-static_cast<double>(gated[r][i])));
                const float silu = div32(gated[r][i], add(1.0F, exponential));
                gated[r][i] = store(mul(silu, upper[r][i]), c.precision);
            }
        return residual(first, dense(gated, layer.down, c.dim, stage, stats), stage);
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
    std::vector<KvCache> cache;
    Engine(sc_module_name name, const Config& config, std::ostream* progress)
        : sc_module(name), cache(config.layers), c(config), model_(config), progress_(progress) {
        SC_THREAD(run);
    }

    unsigned cached_tokens() const { return static_cast<unsigned>(cache.front().keys.size()); }
    const Model& model() const { return model_; }

private:
    const Config& c;
    Model model_;
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
        if (!*progress_) throw std::runtime_error("cannot write progress stream");
    }

    // DMA, tensor and vector stages run back to back; 1 cycle = 1 ns.
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

    void execute(Request& request) {
        // Validate the whole request before any timing or cache change.
        require(!request.inputs.empty(), "empty inference request");
        require(request.inputs.size() <= c.context - cached_tokens(), "KV cache capacity exceeded");
        require(request.inputs.size() <= c.batch_rows(), "request exceeds the SRAM batch reservation");
        for (const auto& input : request.inputs) {
            require(input.size() == c.dim, "input width mismatch");
            for (float value : input) require(std::isfinite(value), "non-finite input");
        }
        const auto rows = static_cast<U64>(request.inputs.size());
        const auto committed = cached_tokens();
        busy_trace.write(true);
        try {
            stage(Kind::dma, rows * c.dim * c.storage_bytes(), false);
            const Stage timed = [this](Kind kind, U64 amount, bool write) { stage(kind, amount, write); };
            Sequence outputs = model_.forward(request.inputs, cache, timed, &stats);
            stats.executed_tokens += rows;
            executed_trace.write(static_cast<unsigned>(stats.executed_tokens));
            wait(SC_ZERO_TIME);
            emit("batch_executed", Kind::idle);
            stage(Kind::dma, rows * c.dim * c.storage_bytes(), true);
            request.outputs = std::move(outputs);
        } catch (...) {
            // The sequence position does not advance: the K/V rows this request
            // wrote are dropped, and the next request writes those positions.
            for (auto& layer : cache) {
                layer.keys.resize(committed);
                layer.values.resize(committed);
            }
            throw;
        }
        stats.tokens += rows;
        ++stats.requests;
        token_trace.write(static_cast<unsigned>(stats.tokens));
        kv_trace.write(cached_tokens());
    }

    void run() {
        while (true) {
            Request* request = nullptr;
            commands.read(request);
            if (!request) throw std::runtime_error("null internal request");
            request->okay = false;
            request->outputs.clear();
            if (request->reset) {
                for (auto& layer : cache) { layer.keys.clear(); layer.values.clear(); }
                kv_trace.write(0); ++resets;
                request->okay = true;
            } else {
                try {
                    execute(*request);
                    request->okay = true;
                } catch (const std::exception& error) {
                    request->error = error.what(); ++rejections;
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
    U64 prefill_cycles = 0, decode_cycles = 0, workload_cycles = 0, rejected_work_cycles = 0;
    std::vector<U64> decode_token_cycles;
    bool completed = false, partition_match = false, lifecycle_verified = false;
    std::string failure;
    Driver(sc_module_name name, Engine& engine, const Config& config)
        : sc_module(name), e(engine), c(config) { SC_THREAD(run); }

private:
    Engine& e;
    const Config& c;
    void submit(Request& request) { e.commands.write(&request); wait(request.done); }
    U64 now() const { return static_cast<U64>(sc_time_stamp() / sc_time(1, SC_NS)); }

    bool cache_equals(const std::vector<KvCache>& other) const {
        for (std::size_t l = 0; l < other.size(); ++l)
            if (e.cache[l].keys != other[l].keys || e.cache[l].values != other[l].values) return false;
        return true;
    }

    void lifecycle() {
        const auto saved = e.cache;
        const auto saved_time = now();
        Request overflow;
        overflow.inputs.resize(c.context - e.cached_tokens() + 1U, Vector(c.dim, 0));
        submit(overflow);
        require(!overflow.okay && overflow.outputs.empty(), "overflow request was accepted");
        Request empty; submit(empty);
        require(!empty.okay, "empty request was accepted");
        require(cache_equals(saved) && now() == saved_time, "rejected request changed the cache or time");
        Request reset; reset.reset = true; submit(reset);
        require(reset.okay && e.cached_tokens() == 0, "cache reset failed");
        // Width and NaN checks run on an empty cache so the capacity check
        // cannot mask them.
        Request malformed; malformed.inputs.push_back(Vector(c.dim + 1U, 0)); submit(malformed);
        require(!malformed.okay && e.cached_tokens() == 0, "malformed width accepted");
        Request invalid; invalid.inputs.push_back(Vector(c.dim, std::numeric_limits<float>::quiet_NaN()));
        submit(invalid);
        require(!invalid.okay && e.cached_tokens() == 0, "non-finite input accepted");
        // A finite input that overflows mid-layer fails after doing work: the
        // work stays in the counters, the cache and outputs do not change.
        const auto before = e.stats;
        const auto work_start = now();
        Request mixed;
        if (c.batch_rows() >= 2) mixed.inputs.push_back(fixture_input(c, 0));
        mixed.inputs.push_back(Vector(c.dim, std::numeric_limits<float>::max()));
        submit(mixed);
        require(!mixed.okay && mixed.outputs.empty() && e.cached_tokens() == 0,
                "failed request published output or advanced the cache");
        require(e.stats.tokens == before.tokens && e.stats.requests == before.requests &&
                e.stats.executed_tokens == before.executed_tokens &&
                e.token_trace.read() == before.tokens && e.kv_trace.read() == 0,
                "failed request counted as completed");
        require(e.stats.dma_read_bytes > before.dma_read_bytes && e.stats.vector_ops > before.vector_ops &&
                now() > work_start, "failed request lost its work accounting");
        rejected_work_cycles = now() - work_start;
        // The workload's own partition, rerun on the reset cache, must
        // reproduce the outputs and every layer's cache.
        const auto replayed = requests(nullptr);
        require(replayed == outputs && cache_equals(saved),
                "replay after reset differs from the first run");
        lifecycle_verified = true;
    }

    // The prompt as one request, then one request per decode token; appends
    // each request's latency to `latencies` when given.
    Sequence requests(std::vector<U64>* latencies) {
        Request prefill;
        for (unsigned pos = 0; pos < c.prefill; ++pos)
            prefill.inputs.push_back(fixture_input(c, pos));
        auto begin = now();
        submit(prefill);
        if (!prefill.okay) throw std::runtime_error("prefill failed: " + prefill.error);
        if (latencies) latencies->push_back(now() - begin);
        Sequence result = std::move(prefill.outputs);
        for (unsigned pos = c.prefill; pos < c.prefill + c.decode; ++pos) {
            begin = now();
            Request decode; decode.inputs.push_back(fixture_input(c, pos)); submit(decode);
            if (!decode.okay) throw std::runtime_error("decode failed: " + decode.error);
            result.push_back(decode.outputs.front());
            if (latencies) latencies->push_back(now() - begin);
        }
        return result;
    }

    void run() {
        try {
            std::vector<U64> latencies;
            outputs = requests(&latencies);
            prefill_cycles = latencies.front();
            decode_token_cycles.assign(latencies.begin() + 1, latencies.end());
            workload_cycles = now(); decode_cycles = workload_cycles - prefill_cycles;
            workload = e.stats;
            if (c.lifecycle) lifecycle();
            if (c.inject_fault) {
                auto& value = outputs[c.fault_row].front();
                value = std::nextafter(value, std::numeric_limits<float>::infinity());
            }

            // Untimed one-row-at-a-time replay: catches batching and
            // prefill/decode partition bugs; the Python oracle checks values.
            // A mismatch is reported, and sc_main exits 2 after the report.
            const Model& reference = e.model();
            std::vector<KvCache> cache(c.layers);
            const Stage untimed = [](Kind, U64, bool) {};
            Sequence expected;
            for (unsigned pos = 0; pos < c.prefill + c.decode; ++pos)
                expected.push_back(
                    reference.forward({fixture_input(c, pos)}, cache, untimed, nullptr).front());
            partition_match = expected == outputs;
            completed = true;
        } catch (const std::exception& error) { failure = error.what(); }
        wait(SC_ZERO_TIME);
        sc_stop();
    }
};

U64 parse_unsigned(const std::string& flag, const std::string& text) {
    U64 result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::invalid_argument(flag + ": expected an unsigned decimal integer, got '" + text + "'");
    return result;
}

unsigned in_range(const std::string& flag, const std::string& text, unsigned minimum,
                  unsigned maximum) {
    const auto value = parse_unsigned(flag, text);
    if (value < minimum || value > maximum)
        throw std::invalid_argument(flag + ": must be " + std::to_string(minimum) + ".." +
                                    std::to_string(maximum) + ", got " + text);
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

constexpr std::size_t probe_words = 65536;

// Comma-separated decimal binary32 words; surrounding whitespace is ignored.
void rounding_probe(std::string text, Precision precision) {
    const auto first = text.find_first_not_of(" \t\r\n");
    const auto last = text.find_last_not_of(" \t\r\n");
    text = first == std::string::npos ? std::string() : text.substr(first, last - first + 1);
    if (text.empty() || text.size() > probe_words * 11U)
        throw std::invalid_argument("--round-f32-bits: expected 1..65536 comma-separated binary32 words");
    std::vector<std::uint32_t> words;
    std::size_t start = 0;
    do {
        const auto comma = text.find(',', start);
        const auto token = text.substr(start, comma == std::string::npos ? comma : comma - start);
        const auto word = parse_unsigned("--round-f32-bits", token);
        if (word > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("--round-f32-bits: binary32 word out of range: " + token);
        if (words.size() == probe_words)
            throw std::invalid_argument("--round-f32-bits: more than 65536 words");
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

constexpr unsigned max_dim = 2048, max_ff = 8192, max_heads = 64, max_layers = 16, max_tokens = 4096;
constexpr U64 max_weight_bytes = U64{256} << 20U;   // host binary32 copy of every weight
constexpr U64 max_host_macs = 4000000000ULL;        // functional MACs, all passes

constexpr const char* usage =
    "Usage: npu-transformer-demo [--dim 2..2048] [--heads 1..64] [--kv-heads 1..64]\n"
    " [--ff-dim 1..8192] [--layers 1..16] [--context 1..4096] [--prefill 1..4096] [--decode 0..4096]\n"
    " [--precision fp32|fp16|bf16] [--seed UINT32] [--dump-output] [--verify-lifecycle]\n"
    " [--tensor-macs-per-cycle 1..4096] [--vector-ops-per-cycle 1..4096]\n"
    " [--dma-bytes-per-cycle 1..4096] [--dma-startup 0..1000] [--sram-bytes 1..67108864]\n"
    " [--vcd BASENAME] [--progress-jsonl PATH] [--round-f32-bits WORDS|-]\n"
    " [--inject-output-fault ROW]\n"
    "The prompt runs as one batch; each decode token is its own request.\n"
    "--inject-output-fault moves output row ROW's first value up one binary32 step\n"
    "to exercise the batched/one-row check; the run then exits 2.\n"
    "--round-f32-bits rounds comma-separated binary32 words to --precision and exits;\n"
    "'-' reads the list from stdin.\n";
}  // namespace

int sc_main(int argc, char* argv[]) {
    try {
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--help") { std::cout << usage; return 0; }
        Config c;
        std::string probe;
        bool probing = false;
        for (int i = 1; i < argc; ++i) {
            const std::string flag(argv[i]);
            if (flag == "--dump-output") { c.dump = true; continue; }
            if (flag == "--verify-lifecycle") { c.lifecycle = true; continue; }
            static const char* const valued[] = {
                "--dim", "--heads", "--kv-heads", "--ff-dim", "--layers", "--context", "--prefill",
                "--decode", "--seed", "--tensor-macs-per-cycle", "--vector-ops-per-cycle",
                "--dma-bytes-per-cycle", "--dma-startup", "--sram-bytes", "--vcd", "--progress-jsonl",
                "--round-f32-bits", "--precision", "--inject-output-fault"};
            if (std::find(std::begin(valued), std::end(valued), flag) == std::end(valued))
                throw std::invalid_argument("unknown argument: " + flag);
            if (i + 1 == argc) throw std::invalid_argument(flag + ": missing value");
            const std::string value(argv[++i]);
            if (flag == "--dim") c.dim = in_range(flag, value, 2, max_dim);
            else if (flag == "--heads") c.heads = in_range(flag, value, 1, max_heads);
            else if (flag == "--kv-heads") c.kv_heads = in_range(flag, value, 1, max_heads);
            else if (flag == "--ff-dim") c.ff_dim = in_range(flag, value, 1, max_ff);
            else if (flag == "--layers") c.layers = in_range(flag, value, 1, max_layers);
            else if (flag == "--context") c.context = in_range(flag, value, 1, max_tokens);
            else if (flag == "--prefill") c.prefill = in_range(flag, value, 1, max_tokens);
            else if (flag == "--decode") c.decode = in_range(flag, value, 0, max_tokens);
            else if (flag == "--seed") c.seed = in_range(flag, value, 0, std::numeric_limits<unsigned>::max());
            else if (flag == "--tensor-macs-per-cycle") c.tensor_rate = in_range(flag, value, 1, 4096);
            else if (flag == "--vector-ops-per-cycle") c.vector_rate = in_range(flag, value, 1, 4096);
            else if (flag == "--dma-bytes-per-cycle") c.dma_rate = in_range(flag, value, 1, 4096);
            else if (flag == "--dma-startup") c.dma_startup = in_range(flag, value, 0, 1000);
            else if (flag == "--sram-bytes") c.sram_bytes = in_range(flag, value, 1, 67108864);
            else if (flag == "--inject-output-fault") {
                c.fault_row = in_range(flag, value, 0, max_tokens);
                c.inject_fault = true;
            }
            else if (flag == "--vcd") { require(!value.empty(), "--vcd: empty basename"); c.vcd = value; }
            else if (flag == "--progress-jsonl") {
                require(!value.empty(), "--progress-jsonl: empty path");
                c.progress = value;
            } else if (flag == "--round-f32-bits") {
                require(!value.empty(), "--round-f32-bits: empty word list");
                probe = value;
                probing = true;
            } else {
                c.precision_name = value;
                if (value == "fp32") c.precision = Precision::fp32;
                else if (value == "fp16") c.precision = Precision::fp16;
                else if (value == "bf16") c.precision = Precision::bf16;
                else throw std::invalid_argument("--precision: must be fp32, fp16 or bf16, got " + value);
            }
        }
        if (probing) {
            if (probe == "-") probe.assign(std::istreambuf_iterator<char>(std::cin), {});
            rounding_probe(probe, c.precision);
            return 0;
        }
        require(c.dim % c.heads == 0 && c.head_dim() % 2 == 0,
                "--dim must be a multiple of --heads with an even head width for RoPE");
        require(c.heads % c.kv_heads == 0, "--kv-heads must divide --heads");
        require(c.prefill + c.decode <= c.context, "--prefill plus --decode exceeds --context");
        require(!c.inject_fault || c.fault_row < c.prefill + c.decode,
                "--inject-output-fault: must name an output row below --prefill plus --decode");
        // Host cost: every weight is held as binary32, and the arithmetic runs
        // twice (timed batches, then the one-row replay), three times with
        // --verify-lifecycle.
        require(U64{4} * c.layers * c.layer_weight_count() <= max_weight_bytes,
                "weights exceed the 256 MiB host limit (--layers, --dim, --ff-dim)");
        require((c.lifecycle ? 3U : 2U) * c.workload_macs() <= max_host_macs,
                "functional work exceeds 4e9 host MACs (--layers, --dim, --ff-dim, --prefill, --decode)");
        require(c.fixed_sram_bytes() + c.weight_column_bytes() <= c.sram_bytes,
                "--sram-bytes cannot hold the batch activations, attention scores, one layer's K/V "
                "and one weight column");
        // SystemC appends .vcd: strip a user suffix, and refuse a VCD path that
        // aliases the progress file (including through links).
        if (!c.vcd.empty() && c.vcd.size() > 4U && c.vcd.compare(c.vcd.size() - 4U, 4U, ".vcd") == 0)
            c.vcd.resize(c.vcd.size() - 4U);
        if (!c.vcd.empty() && !c.progress.empty())
            require(!npu::same_file_path(c.vcd + ".vcd", c.progress),
                    "VCD and progress output paths conflict");
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
        // Runs until the driver stops it, or until no event is pending (a lost
        // request handshake), which leaves driver.completed false.
        sc_start();
        if (trace) sc_close_vcd_trace_file(trace);
        if (!driver.failure.empty()) throw std::runtime_error(driver.failure);
        require(driver.completed, "simulation stopped before the driver completed");
        const auto& s = driver.workload;
        const U64 live_bytes = U64{2} * c.layers * (c.prefill + c.decode) * c.kv_dim() * c.storage_bytes();
        const auto decoded = driver.decode_token_cycles.size();
        const auto tpot_mean = decoded == 0 ? 0.0
            : static_cast<double>(driver.decode_cycles) / static_cast<double>(decoded);
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
            << "{\"schema_version\":1,\"model\":\"systemc-mobile-transformer-block\",\"precision\":\""
            << c.precision_name << "\",\"seed\":" << c.seed << ",\"dim\":" << c.dim
            << ",\"heads\":" << c.heads << ",\"kv_heads\":" << c.kv_heads << ",\"ff_dim\":" << c.ff_dim
            << ",\"layers\":" << c.layers << ",\"context_capacity\":" << c.context
            << ",\"prefill_tokens\":" << c.prefill
            << ",\"decode_tokens\":" << c.decode << ",\"tokens_completed\":" << s.tokens
            << ",\"tokens_executed\":" << s.executed_tokens
            << ",\"requests_completed\":" << s.requests
            << ",\"partition_match\":" << (driver.partition_match ? "true" : "false")
            << ",\"lifecycle_verified\":" << (driver.lifecycle_verified ? "true" : "false")
            << ",\"rejected_work_cycles\":" << driver.rejected_work_cycles
            << ",\"cache_resets\":" << engine.resets << ",\"rejected_requests\":" << engine.rejections
            << ",\"workload_cycles\":" << driver.workload_cycles
            << ",\"prefill_cycles\":" << driver.prefill_cycles
            << ",\"decode_cycles\":" << driver.decode_cycles
            << ",\"ttft_cycles\":" << driver.prefill_cycles << ",\"tpot_cycles\":[";
        for (std::size_t i = 0; i < driver.decode_token_cycles.size(); ++i)
            std::cout << (i ? "," : "") << driver.decode_token_cycles[i];
        std::cout << "],\"tpot_mean_cycles\":" << tpot_mean
            << ",\"simulation_cycles\":" << static_cast<U64>(sc_time_stamp() / sc_time(1, SC_NS))
            << ",\"tensor_macs\":" << s.tensor_macs << ",\"tensor_cycles\":" << s.tensor_cycles
            << ",\"vector_ops\":" << s.vector_ops << ",\"vector_cycles\":" << s.vector_cycles
            << ",\"dma_cycles\":" << s.dma_cycles << ",\"dma_transactions\":" << s.dma_transactions
            << ",\"dma_read_bytes\":" << s.dma_read_bytes << ",\"dma_write_bytes\":" << s.dma_write_bytes
            << ",\"weight_read_bytes\":" << s.weight_read_bytes
            << ",\"kv_dram_read_bytes\":" << s.kv_dram_read_bytes
            << ",\"kv_dram_write_bytes\":" << s.kv_dram_write_bytes
            << ",\"kv_live_bytes\":" << live_bytes
            << ",\"kv_dram_capacity_bytes\":" << c.kv_dram_capacity_bytes()
            << ",\"activation_buffer_bytes\":" << c.activation_buffer_bytes()
            << ",\"score_buffer_bytes\":" << c.score_buffer_bytes()
            << ",\"kv_buffer_bytes\":" << c.kv_buffer_bytes()
            << ",\"weight_buffer_bytes\":" << c.weight_buffer_bytes()
            << ",\"reserved_sram_bytes\":" << c.required_sram()
            << ",\"sram_budget_bytes\":" << c.sram_bytes;
        if (c.dump) { std::cout << ",\"outputs\":"; print_sequence(driver.outputs); }
        std::cout << "}\n";
        require(driver.partition_match, "batched and one-row execution disagree");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "npu-transformer-demo: " << error.what() << '\n';
        return 2;
    }
}
