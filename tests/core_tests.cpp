#include "npu/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception = std::exception, typename Function>
void rejects(Function function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("expected rejection: " + message);
}

// Only std::invalid_argument counts, and its text must contain `expected`.
template <typename Function>
void rejects_with(Function function, const std::string& expected) {
    try {
        function();
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find(expected) == std::string::npos)
            throw std::runtime_error("message '" + std::string(error.what()) + "' lacks '" +
                                     expected + "'");
        return;
    }
    throw std::runtime_error("expected rejection containing '" + expected + "'");
}

void check_arithmetic() {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    require(npu::checked_add(maximum, 0) == maximum, "add boundary");
    require(npu::checked_mul(maximum, 0) == 0, "zero multiply");
    require(npu::checked_mul(maximum, 1) == maximum, "multiply boundary");
    require(npu::ceil_div(maximum, maximum) == 1, "ceil division without overflow");
    require(npu::ceil_div(maximum, 2) == maximum / 2 + 1, "ceil division large");
    require(npu::ceil_div(0, 7) == 0, "ceil division zero numerator");
    rejects<std::overflow_error>([&] { npu::checked_add(maximum, 1); }, "addition overflow");
    rejects<std::overflow_error>([&] { npu::checked_mul(maximum, 2); }, "multiplication overflow");
    rejects<std::invalid_argument>([] { npu::ceil_div(0, 0); }, "zero denominator");
}

void check_requantization() {
    npu::Requant q;
    q.input_zero = q.weight_zero = q.output_zero = 0;
    q.multiplier = 1;
    q.shift = 1;
    const std::vector<std::int32_t> inputs = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
    const std::vector<int> expected = {-3, -2, -2, -1, -1, 0, 1, 1, 2, 2, 3};
    for (std::size_t i = 0; i < inputs.size(); ++i)
        require(npu::requantize(inputs[i], q) == expected[i], "ties away from zero");
    q.output_zero = 17;
    require(npu::requantize(1, q) == 18 && npu::requantize(-1, q) == 16,
            "round before adding output zero point");
    q.relu = true;
    require(npu::requantize(-1000, q) == 17, "ReLU clamps at quantized zero");
    q.relu = false;
    q.multiplier = 0;
    require(npu::requantize(std::numeric_limits<std::int32_t>::min(), q) == 17,
            "zero multiplier");
    q.output_zero = 0;
    q.multiplier = std::numeric_limits<std::int32_t>::max();
    q.shift = 0;
    require(npu::requantize(std::numeric_limits<std::int32_t>::min(), q) == -128,
            "negative extreme saturation without signed overflow");
    require(npu::requantize(std::numeric_limits<std::int32_t>::max(), q) == 127,
            "positive extreme saturation without signed overflow");
    q.shift = 31;
    require(npu::requantize(-1, q) == -1 && npu::requantize(1, q) == 1,
            "maximum shift boundary");
    q.output_zero = -7;
    q.relu = true;
    require(npu::requantize(std::numeric_limits<std::int32_t>::min(), q) == -7,
            "negative output zero ReLU");
    q.shift = 32;
    rejects<std::invalid_argument>([&] { npu::requantize(0, q); }, "shift exceeds 31");
    q.shift = -1;
    rejects<std::invalid_argument>([&] { npu::requantize(0, q); }, "negative native shift");
    q.shift = 0;
    q.multiplier = -1;
    rejects<std::invalid_argument>([&] { npu::requantize(0, q); }, "negative multiplier");
    q.multiplier = 1;
    q.input_zero = -129;
    rejects<std::invalid_argument>([&] { npu::requantize(0, q); }, "input zero is not INT8");
    q.input_zero = 0;
    q.weight_zero = 128;
    rejects<std::invalid_argument>([&] { npu::requantize(0, q); }, "weight zero is not INT8");
    q.weight_zero = 0;
    q.output_zero = 128;
    rejects<std::invalid_argument>([&] { npu::requantize(0, q); }, "output zero is not INT8");
}

void check_even_rounding() {
    npu::Requant q;
    q.input_zero = q.weight_zero = q.output_zero = 0;
    q.rounding = npu::Rounding::HalfToEven;
    q.multiplier = 1;
    q.shift = 1;
    const std::vector<std::int32_t> inputs = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
    const std::vector<int> expected = {-2, -2, -2, -1, 0, 0, 0, 1, 2, 2, 2};
    for (std::size_t i = 0; i < inputs.size(); ++i)
        require(npu::requantize(inputs[i], q) == expected[i], "nearest ties to even");
    q.output_zero = 7;
    require(npu::requantize(-1, q) == 7 && npu::requantize(1, q) == 7 &&
                npu::requantize(-3, q) == 5 && npu::requantize(3, q) == 9,
            "even rounding precedes output zero point");
    q.output_zero = 0;
    q.multiplier = INT32_MAX;
    q.shift = 0;
    require(npu::requantize(INT32_MAX, q) == 127 && npu::requantize(INT32_MIN, q) == -128,
            "even extreme products saturate without overflow");
    q.multipliers = {1, 1, 1};
    q.shifts = {1, 2, 3};
    require(npu::requantize(5, q, 0) == 2 && npu::requantize(-5, q, 0) == -2 &&
                npu::requantize(6, q, 1) == 2 && npu::requantize(-6, q, 1) == -2 &&
                npu::requantize(20, q, 2) == 2 && npu::requantize(-20, q, 2) == -2,
            "per-channel nearest-even at different scales");
    require(!npu::from_string<npu::Rounding>("floor") &&
                npu::from_string<npu::Rounding>("even") == npu::Rounding::HalfToEven,
            "rounding spellings are closed");
}

void check_known_dense() {
    npu::Problem p;
    p.m = p.n = 2;
    p.k = 3;
    p.activations = {1, 2, 3, -1, 0, 2};
    p.weights = {4, -2, 5, 1, -3, 2};
    p.bias = {7, -5};
    npu::Requant q;
    q.input_zero = 1;
    q.weight_zero = -1;
    q.output_zero = 3;
    q.shift = 1;
    const auto result = npu::reference(p, q);
    require(result.accumulators == std::vector<std::int32_t>{9, 3, -11, -2},
            "hand-calculated zero-point dot products");
    require(result.quantized == std::vector<std::int8_t>{8, 5, -3, 2},
            "hand-calculated quantized outputs");
    q.relu = true;
    require(npu::reference(p, q).quantized == std::vector<std::int8_t>{8, 5, 3, 3},
            "fused ReLU dense layer");
    npu::Output tiled;
    npu::compute_tile(p, q, {0, 0, 0, 2, 1}, tiled);
    npu::compute_tile(p, q, {1, 0, 1, 2, 1}, tiled);
    require(tiled.accumulators == result.accumulators, "tiled known accumulator");
    require(tiled.quantized == std::vector<std::int8_t>{8, 5, 3, 3}, "tiled known output");
}

void check_per_channel_quantization() {
    npu::Problem p;
    p.m = 2;
    p.n = 3;
    p.k = 2;
    p.activations = {1, 3, -2, 4};
    p.weights = {2, -3, 7, -4, 6, -8};
    p.bias = {5, -7, 3};
    npu::Requant q;
    q.input_zero = 1;
    q.weight_zero = 0;
    q.weight_zero_points = {1, -2, 3};
    q.output_zero = 5;
    q.multipliers = {1, 3, 2};
    q.shifts = {1, 2, 0};
    const auto result = npu::reference(p, q);
    // Row0: [5 + 0*1 + 2*(-5), -7 + 0*(-1) + 2*8,
    //        3 + 0*4 + 2*(-11)] = [-5, 9, -19].
    // Row1: [5 - 3*1 - 3*5, -7 + 3 + 24, 3 - 12 - 33].
    require(result.accumulators == std::vector<std::int32_t>{-5, 9, -19, -13, 20, -42},
            "hand-calculated per-channel weight zero points");
    require(result.quantized == std::vector<std::int8_t>{2, 12, -33, -2, 20, -79},
            "hand-calculated per-channel scales and rounding");
    npu::Output tiled;
    // Nonzero global tile column must index channel1, not local channel0.
    npu::compute_tile(p, q, {1, 0, 1, 2, 2}, tiled);
    npu::compute_tile(p, q, {0, 0, 0, 2, 1}, tiled);
    require(tiled.accumulators == result.accumulators && tiled.quantized == result.quantized,
            "per-channel tiled global-column indexing");
    q.relu = true;
    require(npu::reference(p, q).quantized == std::vector<std::int8_t>{5, 12, 5, 5, 20, 5},
            "per-channel fused ReLU clamps at output zero");
    q.weight_zero_points.clear();
    q.multipliers.clear();
    q.shifts = {0, 1, 2};
    q.relu = false;
    require(npu::requantize(-1, q, 1) == 4 && npu::requantize(-2, q, 2) == 4,
            "independently optional per-channel shifts and negative ties");
    q.multipliers = {0, INT32_MAX, 2};
    q.shifts = {0, 0, 31};
    require(npu::requantize(INT32_MIN, q, 0) == 5 &&
                npu::requantize(INT32_MAX, q, 1) == 127 &&
                npu::requantize(INT32_MIN, q, 1) == -128 &&
                npu::requantize(INT32_MIN, q, 2) == 3,
            "per-channel zero multiplier, extreme products and saturation");
    rejects<std::out_of_range>([&] { npu::requantize(1, q, 3); },
                               "per-channel requantization channel bounds");
    q.multipliers = {1};
    rejects<std::invalid_argument>([&] { npu::reference(p, q); },
                                   "per-channel parameter count is N");
    q.multipliers.clear();
    q.shifts = {0, 32, 0};
    rejects<std::invalid_argument>([&] { npu::reference(p, q); },
                                   "per-channel shift exceeds 31");
    q.shifts.clear();
    q.weight_zero_points = {0, 128, 0};
    rejects<std::invalid_argument>([&] { npu::reference(p, q); },
                                   "per-channel zero point is INT8");
}

void check_int4() {
    npu::Config c;
    c.shape.m = c.shape.n = c.shape.k = 2;
    c.weights.bits = 4;
    c.quant.weight_zero = 0;
    const auto p = npu::make_problem(c);
    require(p.weights == std::vector<std::int8_t>{-8, 7, -6, 1},
            "signed INT4 fixture uses exact two's-complement range");
    require(std::all_of(p.weights.begin(), p.weights.end(), [](std::int8_t w) {
        return w >= -8 && w <= 7;
    }), "INT4 fixture legal weights");
    npu::Problem known;
    known.m = 1; known.n = 2; known.k = 2;
    known.activations = {127, -128};
    known.weights = {-8, 7, 7, -8};
    known.bias = {1, -1};
    npu::Requant q;
    q.input_zero = q.weight_zero = q.output_zero = 0;
    q.shift = 0;
    const auto result = npu::reference(known, q);
    require(result.accumulators == std::vector<std::int32_t>{-1911, 1912} &&
                result.quantized == std::vector<std::int8_t>{-128, 127},
            "INT4 negative endpoints and saturated INT8 output");
    c.quant.weight_zero = -9;
    rejects<std::invalid_argument>([&] { npu::validate(c); }, "INT4 scalar zero outside range");
    c.quant.weight_zero = 0;
    c.quant.weight_zero_points = {-8, 8};
    rejects<std::invalid_argument>([&] { npu::validate(c); }, "INT4 channel zero outside range");
    c.quant.weight_zero_points = {-8, 7};
    npu::validate(c);
    c.weights.bits = 3;
    rejects<std::invalid_argument>([&] { npu::validate(c); }, "weight width must be 4 or 8");
}

void check_fixture() {
    npu::Config c;
    c.shape.m = 2;
    c.shape.n = 2;
    c.shape.k = 2;
    const auto p = npu::make_problem(c);
    require(p.activations == std::vector<std::int8_t>{-9, 6, -5, 5},
            "LCG activation golden fixture");
    require(p.weights == std::vector<std::int8_t>{13, -10, 0, 4},
            "LCG weight golden fixture");
    require(p.bias == std::vector<std::int32_t>{39, 35}, "LCG bias golden fixture");
    c.shape.seed += UINT64_C(1) << 32;
    const auto wrap = npu::make_problem(c);
    require(p.activations == wrap.activations && p.weights == wrap.weights && p.bias == wrap.bias,
            "seed masks to low 32 bits");
    require(npu::output_hash({{-1, INT32_C(0x01020304)}, {-128, 127}}) ==
                UINT64_C(3663678977158583484),
            "FNV-1a signed and little-endian golden fixture");
    require(npu::output_hash({{}, {}}) == UINT64_C(14695981039346656037),
            "FNV-1a empty offset");
    rejects<std::invalid_argument>([] { npu::output_hash({{1}, {}}); }, "malformed hash output");
}

void check_tiles_and_timing() {
    npu::Config c;
    c.shape.m = 5;
    c.shape.n = 7;
    c.shape.k = 3;
    c.mapping.tile_m = 3;
    c.mapping.tile_n = 4;
    c.arch.compute.macs_per_cycle = 7;
    c.arch.compute.pipeline_cycles = 2;
    c.mapping.buffer_slots = 2;
    const auto tiles = npu::make_tiles(c);
    require(tiles.size() == 4, "edge tile count");
    require(tiles[0].id == 0 && tiles[0].row == 0 && tiles[0].col == 0 &&
                tiles[0].rows == 3 && tiles[0].cols == 4, "first tile");
    require(tiles[3].id == 3 && tiles[3].row == 3 && tiles[3].col == 4 &&
                tiles[3].rows == 2 && tiles[3].cols == 3, "last edge tile");
    require(npu::tile_read_bytes(c, tiles[0]) == 37, "activation + weights + bias reads");
    require(npu::tile_read_bytes(c, tiles[3]) == 27, "edge reads");
    require(npu::tile_write_bytes(tiles[3]) == 6, "INT8 output write bytes");
    require(npu::tile_compute_cycles(c, tiles[0]) == 8, "ceil MAC throughput plus pipeline");
    require(npu::tile_compute_cycles(c, tiles[3]) == 5, "edge compute duration");
    require(npu::scratchpad_bytes(c) == 194, "SRAM includes all slot buffers");
    std::vector<unsigned> coverage(35, 0);
    for (const auto& tile : tiles)
        for (auto r = tile.row; r < tile.row + tile.rows; ++r)
            for (auto col = tile.col; col < tile.col + tile.cols; ++col)
                ++coverage[static_cast<std::size_t>(r * c.shape.n + col)];
    require(std::all_of(coverage.begin(), coverage.end(), [](unsigned n) { return n == 1; }),
            "tiles cover every output exactly once");
    c.mapping.tile_m = c.mapping.tile_n = 4096;
    require(npu::make_tiles(c).size() == 1, "oversized mapping handles one edge tile");
    require(npu::scratchpad_bytes(c) == 2 * (15 + 21 + 28 + 5 * 35),
            "oversized mapping allocates the clipped extents");
}

void check_reduction_panels_and_systolic() {
    npu::Config c;
    c.shape.m = 5; c.shape.n = 7; c.shape.k = 5;
    c.mapping.tile_m = 3; c.mapping.tile_n = 4; c.mapping.tile_k = 2;
    c.weights.bits = 4;
    c.arch.compute.macs_per_cycle = 7;
    c.arch.compute.pipeline_cycles = 2;
    c.quant.weight_zero_points.assign(7, 0);
    c.quant.multipliers.assign(7, 1);
    c.quant.shifts.assign(7, 0);
    const npu::Tile tile{0, 0, 0, 3, 4};
    npu::validate(c);
    require(npu::reduction_blocks(c) == 3 && npu::panel_k(c, 0) == 2 &&
                npu::panel_k(c, 2) == 1, "K panels include final edge");
    require(npu::panel_read_bytes(c, tile, 0) == 74 &&
                npu::panel_read_bytes(c, tile, 1) == 10 &&
                npu::panel_read_bytes(c, tile, 2) == 5 &&
                npu::tile_read_bytes(c, tile) == 89,
            "INT4 panel traffic includes first-panel bias and quant metadata");
    // Per slot: 16 B bias + 48 B channel vectors, panel buffers of
    // 3*2 activations + 4 INT4 weight bytes, 5 B per accumulator and output.
    require(npu::slot_panel_buffers(c) == 2 && npu::scratchpad_bytes(c) == 2 * (64 + 2 * 10 + 60),
            "SRAM holds two K panels plus persistent accumulators and metadata per slot");
    c.mapping.panel_buffers = 1;
    require(npu::slot_panel_buffers(c) == 1 && npu::scratchpad_bytes(c) == 2 * (64 + 10 + 60),
            "one panel buffer per slot");
    for (const std::uint64_t buffers : {0, 3}) {
        c.mapping.panel_buffers = buffers;
        rejects_with([&] { npu::validate(c); }, "panel buffers must be 1 or 2");
    }
    c.mapping.panel_buffers = 2;
    c.mapping.tile_k = 0;
    require(npu::slot_panel_buffers(c) == 1, "a single K panel uses one buffer");
    c.mapping.tile_k = 2;
    require(npu::panel_compute_cycles(c, tile, 0) == 6 &&
                npu::panel_compute_cycles(c, tile, 2) == 4 &&
                npu::tile_compute_cycles(c, tile) == 16,
            "vector pipeline charged for every K panel");
    c.arch.compute.array_rows = 2; c.arch.compute.array_cols = 3;
    // os: ceil(3/2) * ceil(4/3) = 4 output waves of K cycles back to back,
    // skew 2 + 3 - 2 once per panel, pipeline 2.
    require(npu::peak_macs_per_cycle(c) == 6 &&
                npu::panel_compute_cycles(c, tile, 0) == 4 * 2 + 3 + 2 &&
                npu::panel_compute_cycles(c, tile, 2) == 4 * 1 + 3 + 2 &&
                npu::tile_compute_cycles(c, tile) == 13 + 13 + 9,
            "output-stationary waves including the edge K panel");
    const npu::Tile row{0, 0, 0, 1, 4};
    require(npu::panel_compute_cycles(c, row, 0) == 2 * 2 + 3 + 2, "one-row output waves");
    c.arch.compute.dataflow = npu::Dataflow::WeightStationary;
    // ws: ceil(2/2) * ceil(4/3) = 2 weight folds. Three rows hide the second
    // 2-cycle preload: preload 2, fold 3, last fold 3, skew 3, pipeline 2.
    require(npu::peak_macs_per_cycle(c) == 6 &&
                npu::panel_compute_cycles(c, tile, 0) == 2 + 3 + 3 + 3 + 2 &&
                npu::panel_compute_cycles(c, tile, 2) == 13 &&
                npu::tile_compute_cycles(c, tile) == 3 * 13,
            "weight-stationary folds including the edge K panel");
    // One row cannot hide the preload, so the first fold occupies 2 cycles.
    require(npu::panel_compute_cycles(c, row, 0) == 2 + 2 + 1 + 3 + 2,
            "a preload longer than the row stream is exposed");
    c.arch.compute.dataflow = npu::Dataflow::OutputStationary;
    rejects<std::invalid_argument>([&] { npu::panel_k(c, 3); }, "K block bounds");
    c.arch.compute.array_rows = 0;
    rejects<std::invalid_argument>([&] { npu::validate(c); }, "both array dimensions required");
    c.arch.compute.array_rows = 257;
    rejects<std::invalid_argument>([&] { npu::validate(c); }, "array bound");
    c.arch.compute.array_rows = c.arch.compute.array_cols = 0;
    c.arch.compute.dataflow = npu::Dataflow::WeightStationary;
    rejects_with([&] { npu::validate(c); },
                 "weight-stationary dataflow requires --array-rows and --array-cols");
    c.arch.compute.dataflow = npu::Dataflow::OutputStationary;
    c.mapping.tile_k = npu::max_dimension + 1;
    rejects_with([&] { npu::validate(c); }, "reduction tile dimension must be zero (full K) or "
                                            "in 1..1048576");
    c = npu::Config{};
    c.shape.m = 1; c.shape.n = 3; c.shape.k = 3;
    c.mapping.tile_m = 1; c.mapping.tile_n = 3; c.mapping.tile_k = 1; c.weights.bits = 4;
    require(npu::tile_read_bytes(c, {0, 0, 0, 1, 3}) == 21,
            "odd INT4 nibble is padded separately in each DMA panel");
    c.mapping.tile_k = 0;
    require(npu::tile_read_bytes(c, {0, 0, 0, 1, 3}) == 20,
            "full K uses only one INT4 pad nibble");
}

void check_random_edges() {
    std::uint32_t state = 0x361c1eU;
    const auto next = [&state]() {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        return state;
    };
    for (unsigned trial = 0; trial < 64; ++trial) {
        npu::Config c;
        c.shape.m = 1 + next() % 19;
        c.shape.n = 1 + next() % 17;
        c.shape.k = 1 + next() % 23;
        c.mapping.tile_m = 1 + next() % 11;
        c.mapping.tile_n = 1 + next() % 9;
        c.shape.seed = next();
        c.quant.input_zero = static_cast<std::int32_t>(next() % 256) - 128;
        c.quant.weight_zero = static_cast<std::int32_t>(next() % 256) - 128;
        c.quant.output_zero = static_cast<std::int32_t>(next() % 256) - 128;
        c.quant.multiplier = next() % 257;
        c.quant.shift = next() % 18;
        c.quant.relu = next() % 2 != 0;
        if (trial % 2) {
            for (std::uint64_t col = 0; col < c.shape.n; ++col) {
                c.quant.weight_zero_points.push_back(static_cast<std::int32_t>(next() % 256) - 128);
                c.quant.multipliers.push_back(next() % 257);
                c.quant.shifts.push_back(next() % 18);
            }
        }
        const auto p = npu::make_problem(c);
        const auto original_a = p.activations;
        const auto original_b = p.weights;
        const auto original_bias = p.bias;
        const auto reference = npu::reference(p, c.quant);
        auto tiles = npu::make_tiles(c);
        std::reverse(tiles.begin(), tiles.end());
        npu::Output tiled;
        for (const auto& tile : tiles) npu::compute_tile(p, c.quant, tile, tiled);
        require(tiled.accumulators == reference.accumulators &&
                    tiled.quantized == reference.quantized,
                "independent K-major tiled/reference agreement trial " + std::to_string(trial));
        require(npu::output_hash(tiled) == npu::output_hash(reference), "complete output hash agreement");
        require(p.activations == original_a && p.weights == original_b && p.bias == original_bias,
                "computing never mutates input fixtures");
    }
}

void check_overflow_and_shapes() {
    npu::Problem p;
    p.m = p.n = p.k = 1;
    p.activations = {127};
    p.weights = {127};
    p.bias = {std::numeric_limits<std::int32_t>::max()};
    npu::Requant q;
    q.input_zero = q.weight_zero = 0;
    rejects<std::overflow_error>([&] { npu::reference(p, q); }, "reference INT32 overflow");
    rejects<std::overflow_error>([&] {
        npu::Output output;
        npu::compute_tile(p, q, {0, 0, 0, 1, 1}, output);
    }, "tiled INT32 overflow");
    p.k = 2;
    p.activations = {127, 127};
    p.weights = {127, -127};
    require(npu::reference(p, q).accumulators[0] == std::numeric_limits<std::int32_t>::max(),
            "INT64 intermediate permits cancellation before final INT32 check");
    npu::Output tiled;
    npu::compute_tile(p, q, {0, 0, 0, 1, 1}, tiled);
    require(tiled.accumulators[0] == std::numeric_limits<std::int32_t>::max(),
            "tiled cancellation uses INT64");
    p.weights[0] = -127;
    p.weights[1] = -127;
    p.bias[0] = std::numeric_limits<std::int32_t>::min();
    rejects<std::overflow_error>([&] { npu::reference(p, q); }, "negative INT32 overflow");
    p.weights.pop_back();
    rejects<std::invalid_argument>([&] { npu::reference(p, q); }, "malformed matrix storage");
    p.weights.push_back(1);
    p.bias[0] = 0;
    rejects<std::invalid_argument>([&] {
        npu::Output output;
        npu::compute_tile(p, q, {0, UINT64_MAX, 0, 1, 1}, output);
    }, "tile bounds use subtraction without overflow");
    rejects<std::invalid_argument>([&] {
        npu::Output output{{1}, {}};
        npu::compute_tile(p, q, {0, 0, 0, 1, 1}, output);
    }, "malformed output storage");
}


// TOSA 1.0.2 apply_scale_32. Golden vectors are quant_semantics.md section 5;
// the remaining expectations come from TFLite MultiplyByQuantizedMultiplier
// evaluated with tosa shift = 31 - TFLite shift (tests/quantization_tests.py).
void check_tosa_rescale() {
    struct Vector { std::int32_t value, multiplier, shift, single, twice; };
    const Vector vectors[] = {
        {3, 1861152495, 32, 1, 2},
        {-6, 1073741824, 32, -1, -2},
        {-3, 1073741824, 31, -1, -1},
        {5038971, 2062684434, 33, 1209998, 1209999},
        // Ties: single rounding rounds half toward +infinity; double rounding
        // matches it up to shift 31 and rounds away from zero beyond.
        {3, 1 << 30, 31, 2, 2},
        {-1, 1 << 30, 31, 0, 0},
        {2, 1 << 30, 32, 1, 1},
        {-2, 1 << 30, 32, 0, -1},
        // Extremes of value and multiplier.
        {INT32_MIN, INT32_MAX, 62, -1, -1},
        {INT32_MAX, INT32_MAX, 62, 1, 1},
        {INT32_MIN, INT32_MAX, 33, -536870912, -536870912},
        {INT32_MAX, INT32_MAX, 40, 4194304, 4194304},
        {INT32_MIN, 1, 33, 0, 0},
        {INT32_MIN, 0, 62, 0, 0},
        {-2, 1, 2, 0, 0},
    };
    for (const auto& v : vectors) {
        const auto label = std::to_string(v.value) + "*" + std::to_string(v.multiplier) + ">>" +
                           std::to_string(v.shift);
        require(npu::apply_scale_32(v.value, v.multiplier, v.shift, false) == v.single,
                "single rounding " + label);
        require(npu::apply_scale_32(v.value, v.multiplier, v.shift, true) == v.twice,
                "double rounding " + label);
    }
    // The REQUIRE clauses of apply_scale_32.
    rejects_with([] { npu::apply_scale_32(0, -1, 31, false); }, "nonnegative");
    rejects_with([] { npu::apply_scale_32(0, 1, 1, false); }, "2..62");
    rejects_with([] { npu::apply_scale_32(0, 1, 63, true); }, "2..62");
    rejects<std::overflow_error>([] { npu::apply_scale_32(2, 1, 2, false); },
                                 "shift 2 takes values in [-2, 2)");
    rejects<std::overflow_error>([] { npu::apply_scale_32(-3, 1, 2, true); },
                                 "shift 2 lower input bound");
    rejects<std::overflow_error>([] { npu::apply_scale_32(INT32_MIN, 1, 31, false); },
                                 "shift 31 takes values in [-2^30, 2^30)");
    // -0.5 and 0.5 - 2^-32: single rounding sends both to 0.
    require(npu::apply_scale_32(INT32_MIN, 1, 32, false) == 0 &&
                npu::apply_scale_32(INT32_MAX, 1, 32, false) == 0 &&
                npu::apply_scale_32(INT32_MIN, 1, 32, true) == -1,
            "shift 32 accepts every INT32 value");

    // Through requantize: output zero point, ReLU and INT8 saturation follow
    // the scale exactly as in native mode.
    npu::Requant q;
    q.input_zero = q.weight_zero = q.output_zero = 0;
    for (const auto mode : {npu::RescaleMode::TosaSingle, npu::RescaleMode::TosaDouble}) {
        q.rescale = mode;
        const bool twice = mode == npu::RescaleMode::TosaDouble;
        for (const auto& v : vectors) {
            q.multiplier = v.multiplier;
            q.shift = v.shift;
            const auto expected = std::clamp<std::int32_t>(twice ? v.twice : v.single, -128, 127);
            require(npu::requantize(v.value, q) == expected, "TOSA requantize to INT8");
        }
    }
    // -1.5 from -6 * 4 / 2^4: native rounds the magnitude away from zero.
    q.multiplier = 4;
    q.shift = 4;
    q.rescale = npu::RescaleMode::Native;
    require(npu::requantize(-6, q) == -2, "native rounds the magnitude");
    q.rescale = npu::RescaleMode::TosaSingle;
    require(npu::requantize(-6, q) == -1, "TOSA single rounds half toward +infinity");
    q.output_zero = 9;
    q.relu = true;
    // 7 * 4 / 16 = 1.75 rounds to 2; shift 4 accepts inputs in [-8, 8) only.
    require(npu::requantize(-6, q) == 9 && npu::requantize(7, q) == 11, "TOSA with ReLU");
    rejects<std::overflow_error>([&] { npu::requantize(8, q); }, "shift 4 input range");
    q.relu = false;
    q.output_zero = 0;
    // Signed per-channel shifts across the whole TOSA range.
    q.multipliers = {1 << 30, 1 << 30, INT32_MAX};
    q.shifts = {31, 32, 62};
    require(npu::requantize(-3, q, 0) == -1 && npu::requantize(-2, q, 1) == 0 &&
                npu::requantize(INT32_MAX, q, 2) == 1,
            "per-channel TOSA shifts");
    q.rescale = npu::RescaleMode::TosaDouble;
    require(npu::requantize(-2, q, 1) == -1, "per-channel double rounding");
    q.shifts = {31, 1, 62};
    rejects_with([&] { npu::requantize(0, q, 1); },
                 "per-channel shifts must be in 2..62 for tosa-double rescale");
    q.shifts = {31, 32, 63};
    rejects_with([&] { npu::requantize(0, q, 2); }, "2..62");
    q.multipliers.clear();
    q.shifts.clear();
    q.multiplier = 1;
    q.shift = 1;
    rejects_with([&] { npu::requantize(0, q); },
                 "requantization shift must be in 2..62 for tosa-double rescale");
    q.shift = 40;
    q.rounding = npu::Rounding::HalfToEven;
    rejects_with([&] { npu::requantize(0, q); }, "rounding even applies to native rescale only");
    q.rounding = npu::Rounding::HalfAwayFromZero;

    // The reference reports an accumulator outside the apply_scale_32 range.
    npu::Problem p;
    p.m = p.n = p.k = 1;
    p.activations = {10};
    p.weights = {1};
    p.bias = {0};
    q.multiplier = 3;
    q.shift = 5;
    require(npu::reference(p, q).quantized == std::vector<std::int8_t>{1}, "30 / 32 rounds to 1");
    q.shift = 4;
    rejects<std::overflow_error>([&] { npu::reference(p, q); }, "10 is outside [-8, 8)");
}

// G blocks: a grouped GEMM equals G independent GEMMs placed side by side.
void check_grouped_gemm() {
    std::uint32_t state = 0x6e0u;
    const auto next = [&state]() {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        return state;
    };
    for (unsigned trial = 0; trial < 24; ++trial) {
        npu::Config c;
        c.shape.groups = 1 + next() % 6;
        c.shape.m = 1 + next() % 7;
        c.shape.n = trial % 4 == 0 ? 1 : 1 + next() % 5;  // n = 1: depthwise
        c.shape.k = trial % 4 == 0 ? 9 : 1 + next() % 8;
        c.shape.seed = next();
        c.mapping.tile_m = 1 + next() % 4;
        c.mapping.tile_n = 1 + next() % 3;
        c.mapping.tile_k = next() % 4;
        const auto channels = c.shape.groups * c.shape.n;
        c.quant.input_zero = static_cast<std::int32_t>(next() % 31) - 15;
        if (trial % 2) {
            c.quant.rescale = npu::RescaleMode::TosaDouble;
            for (std::uint64_t channel = 0; channel < channels; ++channel) {
                c.quant.weight_zero_points.push_back(static_cast<std::int32_t>(next() % 31) - 15);
                const auto multiplier = (1U << 30) + next() % (1U << 30);
                c.quant.multipliers.push_back(static_cast<std::int32_t>(multiplier));
                c.quant.shifts.push_back(static_cast<std::int32_t>(33 + next() % 10));
            }
        } else {
            c.quant.shift = static_cast<std::int32_t>(next() % 9);
        }
        const auto p = npu::make_problem(c);
        require(p.groups == c.shape.groups &&
                    p.activations.size() == c.shape.m * c.shape.groups * c.shape.k &&
                    p.weights.size() == c.shape.groups * c.shape.k * c.shape.n &&
                    p.bias.size() == channels,
                "grouped tensor sizes");
        const auto grouped = npu::reference(p, c.quant);
        for (std::uint64_t g = 0; g < c.shape.groups; ++g) {
            npu::Problem single;
            single.m = c.shape.m; single.n = c.shape.n; single.k = c.shape.k;
            for (std::uint64_t row = 0; row < c.shape.m; ++row)
                for (std::uint64_t k = 0; k < c.shape.k; ++k)
                    single.activations.push_back(p.activations[static_cast<std::size_t>(
                        row * c.shape.groups * c.shape.k + g * c.shape.k + k)]);
            const auto block = static_cast<std::ptrdiff_t>(g * c.shape.k * c.shape.n);
            single.weights.assign(p.weights.begin() + block,
                                  p.weights.begin() + block +
                                      static_cast<std::ptrdiff_t>(c.shape.k * c.shape.n));
            const auto first = static_cast<std::ptrdiff_t>(g * c.shape.n);
            const auto last = first + static_cast<std::ptrdiff_t>(c.shape.n);
            single.bias.assign(p.bias.begin() + first, p.bias.begin() + last);
            auto lane = c.quant;
            for (auto* vector : {&lane.weight_zero_points, &lane.multipliers, &lane.shifts})
                if (!vector->empty())
                    *vector = std::vector<std::int32_t>(vector->begin() + first,
                                                        vector->begin() + last);
            const auto part = npu::reference(single, lane);
            for (std::uint64_t row = 0; row < c.shape.m; ++row)
                for (std::uint64_t col = 0; col < c.shape.n; ++col) {
                    const auto at = static_cast<std::size_t>(row * channels + g * c.shape.n + col);
                    const auto from = static_cast<std::size_t>(row * c.shape.n + col);
                    require(grouped.accumulators[at] == part.accumulators[from] &&
                                grouped.quantized[at] == part.quantized[from],
                            "group " + std::to_string(g) + " equals its own GEMM, trial " +
                                std::to_string(trial));
                }
        }
        auto tiles = npu::make_tiles(c);
        require(tiles.size() == npu::tile_count(c), "tile count includes groups");
        for (std::size_t i = 0; i < tiles.size(); ++i) {
            const auto& t = tiles[i];
            require(t.id == i && t.group < c.shape.groups && t.col + t.cols <= c.shape.n,
                    "tiles stay inside one group");
            if (i) {
                const auto& before = tiles[i - 1];
                require(std::tie(before.group, before.row, before.col) <
                            std::tie(t.group, t.row, t.col),
                        "tile order is group-major, then row, then column");
            }
        }
        std::reverse(tiles.begin(), tiles.end());
        npu::Output tiled;
        for (const auto& tile : tiles) npu::compute_tile(p, c.quant, tile, tiled);
        require(tiled.accumulators == grouped.accumulators && tiled.quantized == grouped.quantized,
                "grouped tiled datapath equals the reference, trial " + std::to_string(trial));
        for (const auto encoding : {npu::WeightEncoding::Palette, npu::WeightEncoding::Sparse}) {
            auto e = c;
            e.weights.encoding = encoding;
            e.weights.palette_bits = 8;
            const auto panels = npu::encode_weights(e, p);
            require(panels.size() == c.shape.groups * npu::ceil_div(c.shape.n, c.mapping.tile_n) *
                                         npu::reduction_blocks(c),
                    "one encoded panel per group, column panel and K panel");
            npu::Output encoded;
            for (const auto& tile : npu::make_tiles(e)) {
                for (std::uint64_t block = 0; block < npu::reduction_blocks(e); ++block)
                    require(npu::weight_panel(e, panels, tile, block).group == tile.group,
                            "panel lookup is group-aware");
                npu::compute_tile_encoded(p, e, panels, tile, encoded);
            }
            require(encoded.accumulators == grouped.accumulators &&
                        encoded.quantized == grouped.quantized,
                    "grouped encoded datapath equals the reference");
        }
    }

    // Per-channel vectors cover G*N channels.
    npu::Config depthwise;
    depthwise.shape.groups = 8; depthwise.shape.m = 4; depthwise.shape.n = 1; depthwise.shape.k = 9;
    depthwise.quant.shifts.assign(8, 3);
    npu::validate(depthwise);
    depthwise.quant.shifts.assign(1, 3);
    rejects_with([&] { npu::validate(depthwise); }, "per-channel shift count must equal G*N");
    depthwise.quant.shifts.clear();
    npu::Problem wrong = npu::make_problem(npu::Config{});
    wrong.groups = 2;
    rejects_with([&] { npu::reference(wrong, npu::Requant{}); },
                 "activation vector size does not match M*G*K");
    npu::Output output;
    rejects_with([&] {
        npu::compute_tile(npu::make_problem(depthwise), npu::Requant{}, {0, 0, 0, 1, 1, 8}, output);
    }, "tile is outside the output matrix");
}

void check_limits() {
    const auto shape_error = [](const std::function<void(npu::Config&)>& mutation,
                                const std::string& expected) {
        npu::Config c;
        mutation(c);
        rejects_with([&] { npu::validate(c); }, expected);
    };
    shape_error([](auto& c) { c.shape.m = npu::max_dimension + 1; },
                "matrix dimensions must be in 1..1048576");
    shape_error([](auto& c) { c.shape.groups = npu::max_dimension + 1; },
                "groups must be in 1..1048576");
    shape_error([](auto& c) { c.shape.groups = 1024; c.shape.n = 1025; },
                "groups*N and groups*K must be at most 1048576");
    // Exactly 4e9 MACs is a legal bit-exact run; one more row is not.
    npu::Config functional;
    functional.shape.m = 62500; functional.shape.n = 64; functional.shape.k = 1000;
    require(npu::total_macs(functional.shape) == npu::max_functional_macs, "boundary workload");
    npu::validate(functional);
    ++functional.shape.m;
    rejects_with([&] { npu::validate(functional); },
                 "workload exceeds 4000000000 MACs for a bit-exact run; --timing-only allows up "
                 "to 10000000000000");
    functional.sim.timing_only = true;
    npu::validate(functional);
    rejects_with([&] { npu::make_problem(functional); }, "--timing-only runs generate no tensors");
    // The timing-only limit, with tiles large enough for the host budget.
    npu::Config timing;
    timing.sim.timing_only = true;
    timing.shape.m = 1000000; timing.shape.n = 1000; timing.shape.k = 10000;
    timing.mapping.tile_m = timing.mapping.tile_n = 1000;
    timing.arch.sram_bytes = UINT64_C(1) << 30;
    require(npu::total_macs(timing.shape) == npu::max_timing_only_macs, "timing-only boundary");
    npu::validate(timing);
    ++timing.shape.k;
    rejects_with([&] { npu::validate(timing); },
                 "workload exceeds 10000000000000 MACs, the --timing-only limit");
    // Largest dimensions without materialized tensors.
    npu::Config wide;
    wide.sim.timing_only = true;
    wide.shape.m = 1; wide.shape.n = npu::max_dimension; wide.shape.k = npu::max_dimension;
    wide.mapping.tile_n = 4096; wide.mapping.tile_k = 4096;
    wide.arch.sram_bytes = UINT64_C(1) << 30;
    npu::validate(wide);
    wide.weights.encoding = npu::WeightEncoding::Palette;
    rejects_with([&] { npu::validate(wide); }, "--timing-only supports dense weights only");
}

void check_validation() {
    npu::validate(npu::Config{});
    const auto bad = [](const std::function<void(npu::Config&)>& mutation) {
        npu::Config c;
        mutation(c);
        rejects([&] { npu::validate(c); }, "invalid configuration");
    };
    bad([](auto& c) { c.shape.m = 0; });
    bad([](auto& c) { c.shape.n = npu::max_dimension + 1; });
    bad([](auto& c) { c.shape.groups = 0; });
    bad([](auto& c) { c.shape.k = UINT64_MAX; });
    bad([](auto& c) { c.shape.m = c.shape.n = c.shape.k = 4096; });
    bad([](auto& c) { c.mapping.tile_m = 0; });
    bad([](auto& c) { c.mapping.tile_n = npu::max_dimension + 1; });
    bad([](auto& c) { c.arch.compute.macs_per_cycle = 0; });
    bad([](auto& c) { c.arch.memory.bytes_per_cycle = 0; });
    bad([](auto& c) { c.arch.memory.latency = UINT64_MAX; });
    bad([](auto& c) { c.arch.compute.pipeline_cycles = 1000000001; });
    bad([](auto& c) { c.mapping.buffer_slots = 0; });
    bad([](auto& c) { c.mapping.buffer_slots = 65; });
    bad([](auto& c) { c.arch.sram_bytes = 1; });
    bad([](auto& c) { c.arch.frequency_mhz = 0; });
    bad([](auto& c) { c.arch.frequency_mhz = 1e-300; });
    bad([](auto& c) { c.arch.frequency_mhz = std::numeric_limits<double>::quiet_NaN(); });
    bad([](auto& c) { c.arch.frequency_mhz = std::numeric_limits<double>::infinity(); });
    bad([](auto& c) { c.quant.output_zero = 128; });
    bad([](auto& c) {
        c.shape.m = c.shape.n = 4096; c.shape.k = 1;
        c.mapping.tile_m = c.mapping.tile_n = 1;
        c.arch.sram_bytes = UINT64_MAX;
    }); // Tile metadata alone exceeds the host memory safety budget.
    bad([](auto& c) {
        c.shape.m = c.shape.n = 1000; c.shape.k = 1;
        c.mapping.tile_m = c.mapping.tile_n = 1;
        c.sim.trace_path = "trace.csv";
    }); // Trace storage for 10^6 tiles alone exceeds the host safety budget.
    npu::Config untraced;
    untraced.shape.m = untraced.shape.n = 1000; untraced.shape.k = 1;
    untraced.mapping.tile_m = untraced.mapping.tile_n = 1;
    npu::validate(untraced); // Without a trace, 10^6 tile records fit.
    npu::Config c;
    c.arch.frequency_mhz = 1e-6;
    npu::validate(c); // A 1 Hz target remains representable and valid.
    c.arch.sram_bytes = npu::scratchpad_bytes(c);
    npu::validate(c);
    --c.arch.sram_bytes;
    rejects<std::invalid_argument>([&] { npu::validate(c); }, "exact SRAM capacity boundary");

    npu::Config slow;
    slow.shape.m = slow.shape.n = slow.shape.k = 1;
    slow.mapping.tile_m = slow.mapping.tile_n = slow.mapping.buffer_slots = 1;
    slow.arch.memory.latency = slow.arch.compute.pipeline_cycles = 1000000000;
    npu::validate(slow); // Event mode jumps directly across long stage waits.
    slow.sim.timing = npu::TimingMode::Cycle;
    rejects<std::invalid_argument>([&] { npu::validate(slow); },
                                   "cycle mode rejects impractical wait iteration count");
    slow.arch.compute.macs_per_cycle = slow.arch.memory.bytes_per_cycle = 1;
    slow.arch.memory.latency = 0;
    slow.arch.compute.pipeline_cycles = 9999992;
    npu::validate(slow); // Read6 + compute(1+9999992) + write1 = 10000000.
    ++slow.arch.compute.pipeline_cycles;
    rejects<std::invalid_argument>([&] { npu::validate(slow); },
                                   "cycle mode rejects one iteration beyond boundary");
    slow.arch.compute.pipeline_cycles = 0;
    npu::validate(slow); // The minimal mapping costs eight serial cycles.
}

}  // namespace

int main() {
    try {
        check_arithmetic();
        check_requantization();
        check_even_rounding();
        check_known_dense();
        check_per_channel_quantization();
        check_int4();
        check_fixture();
        check_tiles_and_timing();
        check_reduction_panels_and_systolic();
        check_random_edges();
        check_overflow_and_shapes();
        check_tosa_rescale();
        check_grouped_gemm();
        check_limits();
        check_validation();
        std::cout << "core tests: all checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "core tests FAILED: " << error.what() << '\n';
        return 1;
    }
}
