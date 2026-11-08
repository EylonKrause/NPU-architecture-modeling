#include "npu/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
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
    require(npu::requantize(std::numeric_limits<std::int32_t>::min(), q) == 17, "zero multiplier");
    q.output_zero = 0;
    q.multiplier = std::numeric_limits<std::int32_t>::max();
    q.shift = 0;
    require(npu::requantize(std::numeric_limits<std::int32_t>::min(), q) == -128,
            "negative extreme saturation without signed overflow");
    require(npu::requantize(std::numeric_limits<std::int32_t>::max(), q) == 127,
            "positive extreme saturation without signed overflow");
    q.shift = 31;
    require(npu::requantize(-1, q) == -1 && npu::requantize(1, q) == 1, "maximum shift boundary");
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
    // The tile at column 1 takes channel 1's parameters for its first column: per-channel
    // vectors are indexed by global channel.
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
            "per-channel shifts without per-channel multipliers; negative ties");
    q.multipliers = {0, INT32_MAX, 2};
    q.shifts = {0, 0, 31};
    require(npu::requantize(INT32_MIN, q, 0) == 5 && npu::requantize(INT32_MAX, q, 1) == 127 &&
                npu::requantize(INT32_MIN, q, 1) == -128 && npu::requantize(INT32_MIN, q, 2) == 3,
            "per-channel zero multiplier, extreme products and saturation");
    rejects<std::out_of_range>([&] { npu::requantize(1, q, 3); },
                               "per-channel requantization channel bounds");
    q.multipliers = {1};
    rejects<std::invalid_argument>([&] { npu::reference(p, q); },
                                   "per-channel parameter count is N");
    q.multipliers.clear();
    q.shifts = {0, 32, 0};
    rejects<std::invalid_argument>([&] { npu::reference(p, q); }, "per-channel shift exceeds 31");
    q.shifts.clear();
    q.weight_zero_points = {0, 128, 0};
    rejects<std::invalid_argument>([&] { npu::reference(p, q); }, "per-channel zero point is INT8");
}

void check_int4() {
    npu::Config c;
    c.shape.m = c.shape.n = c.shape.k = 2;
    c.weights.bits = 4;
    c.quant.weight_zero = 0;
    const auto p = npu::make_problem(c);
    require(p.weights == std::vector<std::int8_t>{-8, 7, -6, 1}, "INT4 fixture reaches -8 and 7");
    require(std::all_of(p.weights.begin(), p.weights.end(),
                        [](std::int8_t w) { return w >= -8 && w <= 7; }),
            "INT4 fixture legal weights");
    npu::Problem known;
    known.m = 1;
    known.n = 2;
    known.k = 2;
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
    require(p.weights == std::vector<std::int8_t>{13, -10, 0, 4}, "LCG weight golden fixture");
    require(p.bias == std::vector<std::int32_t>{39, 35}, "LCG bias golden fixture");
    c.shape.seed += UINT64_C(1) << 32;
    const auto wrap = npu::make_problem(c);
    require(p.activations == wrap.activations && p.weights == wrap.weights && p.bias == wrap.bias,
            "seed masks to low 32 bits");
    require(
        npu::output_hash({{-1, INT32_C(0x01020304)}, {-128, 127}}) == UINT64_C(3663678977158583484),
        "FNV-1a signed and little-endian golden fixture");
    require(npu::output_hash({{}, {}}) == UINT64_C(14695981039346656037), "FNV-1a empty offset");
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
    require(tiles[0].id == 0 && tiles[0].row == 0 && tiles[0].col == 0 && tiles[0].rows == 3 &&
                tiles[0].cols == 4,
            "first tile");
    require(tiles[3].id == 3 && tiles[3].row == 3 && tiles[3].col == 4 && tiles[3].rows == 2 &&
                tiles[3].cols == 3,
            "last edge tile");
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
    c.shape.m = 5;
    c.shape.n = 7;
    c.shape.k = 5;
    c.mapping.tile_m = 3;
    c.mapping.tile_n = 4;
    c.mapping.tile_k = 2;
    c.weights.bits = 4;
    c.arch.compute.macs_per_cycle = 7;
    c.arch.compute.pipeline_cycles = 2;
    c.quant.weight_zero_points.assign(7, 0);
    c.quant.multipliers.assign(7, 1);
    c.quant.shifts.assign(7, 0);
    const npu::Tile tile{0, 0, 0, 3, 4};
    npu::validate(c);
    require(npu::reduction_blocks(c) == 3 && npu::panel_k(c, 0) == 2 && npu::panel_k(c, 2) == 1,
            "K panels include final edge");
    require(npu::panel_read_bytes(c, tile, 0) == 74 && npu::panel_read_bytes(c, tile, 1) == 10 &&
                npu::panel_read_bytes(c, tile, 2) == 5 && npu::tile_read_bytes(c, tile) == 89,
            "INT4 panel traffic includes first-panel bias and quant metadata");
    // Per slot: 16 B bias + 48 B channel vectors, panel buffers of
    // 3*2 activations + 4 INT4 weight bytes, 5 B per accumulator and output.
    require(npu::slot_panel_buffers(c) == 2 && npu::scratchpad_bytes(c) == 2 * (64 + 2 * 10 + 60),
            "SRAM holds two K panels plus persistent accumulators and metadata per slot");
    c.mapping.panel_buffers = 1;
    require(npu::slot_panel_buffers(c) == 1 && npu::scratchpad_bytes(c) == 2 * (64 + 10 + 60),
            "one panel buffer per slot");
    for (const std::uint64_t buffers : {UINT64_C(0), UINT64_C(3)}) {
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
    c.arch.compute.array_rows = 2;
    c.arch.compute.array_cols = 3;
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
    rejects_with([&] { npu::validate(c); },
                 "reduction tile dimension must be zero (full K) or "
                 "in 1..1048576");
    c = npu::Config{};
    c.shape.m = 1;
    c.shape.n = 3;
    c.shape.k = 3;
    c.mapping.tile_m = 1;
    c.mapping.tile_n = 3;
    c.mapping.tile_k = 1;
    c.weights.bits = 4;
    require(npu::tile_read_bytes(c, {0, 0, 0, 1, 3}) == 21,
            "odd INT4 nibble is padded separately in each DMA panel");
    c.mapping.tile_k = 0;
    require(npu::tile_read_bytes(c, {0, 0, 0, 1, 3}) == 20, "full K uses only one INT4 pad nibble");
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
        c.quant.multiplier = static_cast<std::int32_t>(next() % 257);
        c.quant.shift = static_cast<std::int32_t>(next() % 18);
        c.quant.relu = next() % 2 != 0;
        if (trial % 2) {
            for (std::uint64_t col = 0; col < c.shape.n; ++col) {
                c.quant.weight_zero_points.push_back(static_cast<std::int32_t>(next() % 256) - 128);
                c.quant.multipliers.push_back(static_cast<std::int32_t>(next() % 257));
                c.quant.shifts.push_back(static_cast<std::int32_t>(next() % 18));
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
        require(
            tiled.accumulators == reference.accumulators && tiled.quantized == reference.quantized,
            "K-major tiles match the reference, trial " + std::to_string(trial));
        require(npu::output_hash(tiled) == npu::output_hash(reference),
                "tiled and reference hashes match");
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
    rejects<std::overflow_error>(
        [&] {
            npu::Output output;
            npu::compute_tile(p, q, {0, 0, 0, 1, 1}, output);
        },
        "tiled INT32 overflow");
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
    rejects<std::invalid_argument>(
        [&] {
            npu::Output output;
            npu::compute_tile(p, q, {0, UINT64_MAX, 0, 1, 1}, output);
        },
        "tile bounds use subtraction without overflow");
    rejects<std::invalid_argument>(
        [&] {
            npu::Output output{{1}, {}};
            npu::compute_tile(p, q, {0, 0, 0, 1, 1}, output);
        },
        "malformed output storage");
}

// TOSA 1.0.2 apply_scale_32. The first four vectors are the golden vectors of docs/numerics.md;
// the remaining expectations come from TFLite MultiplyByQuantizedMultiplier
// evaluated with tosa shift = 31 - TFLite shift (tests/quantization_tests.py).
void check_tosa_rescale() {
    struct Vector {
        std::int32_t value, multiplier, shift, single, twice;
    };
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
            single.m = c.shape.m;
            single.n = c.shape.n;
            single.k = c.shape.k;
            for (std::uint64_t row = 0; row < c.shape.m; ++row)
                for (std::uint64_t k = 0; k < c.shape.k; ++k)
                    single.activations.push_back(p.activations[static_cast<std::size_t>(
                        row * c.shape.groups * c.shape.k + g * c.shape.k + k)]);
            const auto block = static_cast<std::ptrdiff_t>(g * c.shape.k * c.shape.n);
            single.weights.assign(
                p.weights.begin() + block,
                p.weights.begin() + block + static_cast<std::ptrdiff_t>(c.shape.k * c.shape.n));
            const auto first = static_cast<std::ptrdiff_t>(g * c.shape.n);
            const auto last = first + static_cast<std::ptrdiff_t>(c.shape.n);
            single.bias.assign(p.bias.begin() + first, p.bias.begin() + last);
            auto lane = c.quant;
            for (auto* vector : {&lane.weight_zero_points, &lane.multipliers, &lane.shifts})
                if (!vector->empty())
                    *vector =
                        std::vector<std::int32_t>(vector->begin() + first, vector->begin() + last);
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
    depthwise.shape.groups = 8;
    depthwise.shape.m = 4;
    depthwise.shape.n = 1;
    depthwise.shape.k = 9;
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
    rejects_with(
        [&] {
            npu::compute_tile(npu::make_problem(depthwise), npu::Requant{}, {0, 0, 0, 1, 1, 8},
                              output);
        },
        "tile is outside the output matrix");
}

void check_decoder_overlap() {
    npu::Config c;
    c.shape.m = 4;
    c.shape.n = 8;
    c.shape.k = 16;
    c.mapping.tile_m = 4;
    c.mapping.tile_n = 8;
    c.mapping.tile_k = 8;
    c.arch.compute.macs_per_cycle = 16;
    c.arch.compute.pipeline_cycles = 3;
    c.quant.input_zero = c.quant.weight_zero = 0;
    c.weights.palette_bits = 2;
    c.weights.sparse_block = 4;
    // Column 0 holds 1 and every other weight is 0: one nonzero per K row.
    npu::Problem p;
    p.m = 4;
    p.n = 8;
    p.k = 16;
    p.activations.assign(64, 1);
    p.bias.assign(8, 0);
    for (std::uint64_t index = 0; index < 128; ++index)
        p.weights.push_back(static_cast<std::int8_t>(index % 8 == 0));
    const npu::Tile tile{0, 0, 0, 4, 8};
    // A K panel of 8: 4*8*8 MACs at 16 per cycle is 16 cycles.
    const auto panel_cycles = [&](npu::WeightEncoding encoding, std::uint64_t rate) {
        c.weights.encoding = encoding;
        c.arch.compute.weight_decode_values_per_cycle = rate;
        const auto panels = npu::encode_weights(c, p);
        return npu::encoded_panel_compute_cycles(c, tile, 1, npu::weight_panel(c, panels, tile, 1));
    };
    // Palette decodes 64 values: 4 cycles at 16 per cycle hide under the MACs.
    require(panel_cycles(npu::WeightEncoding::Palette, 16) == 16 + 3, "MAC-bound palette panel");
    require(npu::panel_compute_cycles(c, tile, 1) == 16 + 3, "palette bound equals dense");
    require(panel_cycles(npu::WeightEncoding::Palette, 1) == 64 + 3, "decode-bound palette panel");
    require(npu::panel_compute_cycles(c, tile, 1) == 64 + 3, "palette decode bound");
    // Sparse: 16 bitmap blocks, 8 of them kept (32 values), 8 nonzero
    // weights. At 2 per cycle decode is 8 + 16 cycles; MACs ceil(4*8/16).
    require(panel_cycles(npu::WeightEncoding::Sparse, 2) == 24 + 3, "decode-bound sparse panel");
    require(npu::panel_compute_cycles(c, tile, 1) == 32 + 8 + 3,
            "sparse bound decodes every value and block");
    require(panel_cycles(npu::WeightEncoding::Sparse, 64) == 2 + 3, "MAC-bound sparse panel");
    c.weights.encoding = npu::WeightEncoding::Dense;
    require(npu::panel_compute_cycles(c, tile, 1) == 16 + 3, "dense panel");
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
    shape_error(
        [](auto& c) {
            c.shape.groups = 1024;
            c.shape.n = 1025;
        },
        "groups*N and groups*K must be at most 1048576");
    // Exactly 4e9 MACs is a legal bit-exact run; one more row is not.
    npu::Config functional;
    functional.shape.m = 62500;
    functional.shape.n = 64;
    functional.shape.k = 1000;
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
    timing.shape.m = 1000000;
    timing.shape.n = 1000;
    timing.shape.k = 10000;
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
    wide.shape.m = 1;
    wide.shape.n = npu::max_dimension;
    wide.shape.k = npu::max_dimension;
    wide.mapping.tile_n = 4096;
    wide.mapping.tile_k = 4096;
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
        c.shape.m = c.shape.n = 4096;
        c.shape.k = 1;
        c.mapping.tile_m = c.mapping.tile_n = 1;
        c.arch.sram_bytes = UINT64_MAX;
    });  // Tile metadata alone exceeds the host memory safety budget.
    bad([](auto& c) {
        c.shape.m = c.shape.n = 1000;
        c.shape.k = 1;
        c.mapping.tile_m = c.mapping.tile_n = 1;
        c.sim.trace_path = "trace.csv";
    });  // Trace storage for 10^6 tiles alone exceeds the host safety budget.
    npu::Config untraced;
    untraced.shape.m = untraced.shape.n = 1000;
    untraced.shape.k = 1;
    untraced.mapping.tile_m = untraced.mapping.tile_n = 1;
    npu::validate(untraced);  // Without a trace, 10^6 tile records fit.
    npu::Config c;
    c.arch.frequency_mhz = 1e-6;
    npu::validate(c);  // 1 Hz is the lowest accepted clock.
    c.arch.sram_bytes = npu::scratchpad_bytes(c);
    npu::validate(c);
    --c.arch.sram_bytes;
    rejects<std::invalid_argument>([&] { npu::validate(c); }, "exact SRAM capacity boundary");

    npu::Config slow;
    slow.shape.m = slow.shape.n = slow.shape.k = 1;
    slow.mapping.tile_m = slow.mapping.tile_n = slow.mapping.buffer_slots = 1;
    slow.arch.memory.latency = slow.arch.compute.pipeline_cycles = 1000000000;
    npu::validate(slow);  // Event mode has no 10000000-cycle limit.
    slow.sim.timing = npu::TimingMode::Cycle;
    rejects<std::invalid_argument>([&] { npu::validate(slow); },
                                   "cycle mode rejects serial work above 10000000 cycles");
    slow.arch.compute.macs_per_cycle = slow.arch.memory.bytes_per_cycle = 1;
    slow.arch.memory.latency = 0;
    slow.arch.compute.pipeline_cycles = 9999992;
    npu::validate(slow);  // Read 6, compute 1 + 9999992, write 1: 10000000 cycles.
    ++slow.arch.compute.pipeline_cycles;
    rejects<std::invalid_argument>([&] { npu::validate(slow); },
                                   "cycle mode rejects 10000001 serial cycles");
    slow.arch.compute.pipeline_cycles = 0;
    npu::validate(slow);  // The minimal mapping costs eight serial cycles.
}

// next_burst against its limits, DramTimeline against an unmerged model, and
// the address map.
void check_memory_map() {
    npu::AxiConfig axi;
    axi.data_bytes = 16;
    axi.max_burst_beats = 16;
    const auto burst = [&](std::uint64_t address, std::uint64_t end, std::uint64_t bytes,
                           std::uint64_t beats, const char* what) {
        const auto b = npu::next_burst(axi, address, end);
        require(b.address == address && b.bytes == bytes && b.beats == beats, what);
    };
    burst(0x1000, 0x1000 + 1000, 256, 16, "aligned burst of max_burst_beats");
    burst(0x1008, 0x2000, 248, 16, "misaligned start: a partial first beat");
    burst(0x0FF0, 0x2000, 16, 1, "a 4 KiB boundary ends the burst");
    burst(0x100, 0x114, 20, 2, "a partial last beat");
    burst(0x105, 0x107, 2, 1, "two bytes inside one beat");
    axi.max_burst_beats = 256;
    burst(0, 0x10000, 4096, 256, "256 beats of 16 bytes fill a 4 KiB page");
    axi.data_bytes = 64;
    burst(0x20, 0x5000, 0xFE0, 64, "the page ends before 256 beats of 64 bytes");
    rejects_with([&] { npu::next_burst(axi, 5, 5); }, "burst address is outside its run");
    std::uint64_t state = 12345;
    const auto next = [&state](std::uint64_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (state >> 33) % bound;
    };
    for (int i = 0; i < 2000; ++i) {
        axi.data_bytes = UINT64_C(8) << next(4);
        axi.max_burst_beats = 1 + next(256);
        const auto start = next(20000), length = 1 + next(20000), end = start + length;
        std::uint64_t address = start, beats = 0;
        while (address < end) {
            const auto b = npu::next_burst(axi, address, end);
            require(b.bytes >= 1 && b.beats >= 1 && b.beats <= axi.max_burst_beats,
                    "burst size within the limits");
            require(address / 4096 == (address + b.bytes - 1) / 4096,
                    "no burst crosses a 4 KiB boundary");
            require(b.beats ==
                        npu::ceil_div(address + b.bytes, axi.data_bytes) - address / axi.data_bytes,
                    "beats count the slots touched");
            beats += b.beats;
            address += b.bytes;
        }
        require(address == end, "bursts tile the run");
        require(beats == npu::ceil_div(end, axi.data_bytes) - start / axi.data_bytes,
                "bursts after the first start beat aligned");
    }

    npu::DramTimeline dram;
    require(dram.reserve(100, 50) == 100, "empty timeline");
    require(dram.reserve(0, 100) == 0, "a gap of exactly the size fits");
    require(dram.intervals() == 1, "touching reservations merge");
    require(dram.reserve(0, 10) == 150, "full up to 150");
    require(dram.reserve(400, 10) == 400, "a later reservation leaves a hole");
    require(dram.intervals() == 2, "[0, 160) and [400, 410)");
    require(dram.reserve(120, 240) == 160, "the hole [160, 400) holds 240 bytes");
    require(dram.intervals() == 1, "filling the hole joins both sides");
    require(dram.reserve(155, 1) == 410, "no hole remains before 410");
    require(dram.reserve(50, 0) == 50, "an empty reservation takes nothing");
    require(dram.reserve(500, 20) == 500 && dram.reserve(600, 5) == 600, "two more holes");
    require(dram.intervals() == 3, "[0, 411), [500, 520) and [600, 605)");
    dram.release(520);
    require(dram.intervals() == 1, "release drops intervals ending by then");
    require(dram.reserve(0, 5) == 0, "released time is free again");
    require(dram.reserve(590, 10) == 590 && dram.intervals() == 2, "joins the next interval");
    // Against a timeline that keeps every reservation apart and forgets none.
    struct Apart {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> busy;
        std::uint64_t reserve(std::uint64_t begin, std::uint64_t bytes) {
            if (!bytes) return begin;
            for (bool moved = true; moved;) {
                moved = false;
                for (const auto& [low, high] : busy)
                    if (low < begin + bytes && begin < high) {
                        begin = high;
                        moved = true;
                    }
            }
            busy.emplace_back(begin, begin + bytes);
            return begin;
        }
    };
    std::uint64_t seed = 5;
    const auto pick = [&seed](std::uint64_t range) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return (seed >> 33) % range;
    };
    for (int round = 0; round < 50; ++round) {
        npu::DramTimeline merged;
        Apart apart;
        std::uint64_t now = 0;
        for (int step = 0; step < 200; ++step) {
            const auto earliest = now + pick(300);
            const auto bytes = pick(40);
            require(merged.reserve(earliest, bytes) == apart.reserve(earliest, bytes),
                    "merging intervals leaves every gap in place");
            if (pick(4) == 0) {
                now += pick(50);
                merged.release(now);
            }
        }
    }

    npu::Config c;
    c.shape = {5, 7, 10, 3, 1};
    c.mapping.tile_m = 2;
    c.mapping.tile_n = 3;
    c.mapping.tile_k = 4;
    c.weights.bits = 4;
    std::uint64_t cursor = 0;
    for (std::uint64_t group = 0; group < 3; ++group)
        for (std::uint64_t col = 0; col < 7; col += 3)
            for (std::uint64_t block = 0; block < 3; ++block) {
                const auto cols = std::min<std::uint64_t>(3, 7 - col);
                require(npu::dense_weight_offset(c, {0, 0, col, 2, cols, group}, block) == cursor,
                        "dense panels pack back to back in load order");
                cursor += npu::ceil_div(npu::panel_k(c, block) * cols * 4, 8);
            }
    std::vector<npu::AddressRun> runs;
    for (const auto& tile : npu::make_tiles(c)) {
        for (std::uint64_t block = 0; block < 3; ++block) {
            const auto weights = npu::ceil_div(npu::panel_k(c, block) * tile.cols * 4, 8);
            npu::panel_read_runs(c, tile, block, 77, weights, runs);
            std::uint64_t total = 0;
            for (const auto& run : runs) total += run.length;
            require(total == npu::panel_read_bytes(c, tile, block), "runs carry the panel bytes");
            require(runs.size() == tile.rows + 1 + (block == 0), "activation rows, weights, bias");
            require(runs[0].base ==
                            npu::activation_base + tile.row * 30 + tile.group * 10 + block * 4 &&
                        runs[0].length == npu::panel_k(c, block),
                    "activation rows are G*K apart");
            require(runs[tile.rows].base == npu::weight_base + 77, "weight panel at its offset");
            if (block == 0)
                require(runs.back().base == npu::param_base + (tile.group * 7 + tile.col) * 4 &&
                            runs.back().length == 4 * tile.cols,
                        "one bias record per channel");
            npu::panel_read_runs(c, tile, block, 77, 0, runs);
            require(runs.size() == tile.rows + (block == 0), "a cached panel skips the weights");
        }
        npu::tile_store_runs(c, tile, runs);
        require(runs.size() == tile.rows && runs[0].length == tile.cols &&
                    runs[0].base == npu::output_base + tile.row * 21 + tile.group * 7 + tile.col &&
                    runs.back().base == runs[0].base + (tile.rows - 1) * 21,
                "output rows are G*N apart");
    }
    require(npu::param_record_bytes(c) == 4, "bias only");
    c.quant.multipliers.assign(21, 1);
    c.quant.shifts.assign(21, 7);
    require(npu::param_record_bytes(c) == 12, "bias, multiplier and shift");

    npu::Config m;
    require(npu::peak_memory_bytes_per_cycle(m) == 16, "serial peak is bytes_per_cycle");
    m.arch.memory.model = npu::MemoryModel::Axi;
    m.arch.memory.axi.data_bytes = 32;
    require(npu::peak_memory_bytes_per_cycle(m) == 64, "two AXI channels");
    m.arch.memory.dram_bytes_per_cycle = 40;
    require(npu::peak_memory_bytes_per_cycle(m) == 40, "DRAM caps the port");
    m.arch.memory.dram_bytes_per_cycle = 100;
    require(npu::peak_memory_bytes_per_cycle(m) == 64, "a faster DRAM does not");
    npu::validate(m);
    using Mutation = std::function<void(npu::AxiConfig&, npu::MemoryConfig&)>;
    const auto bad = [](const Mutation& mutation, const std::string& message) {
        npu::Config config;
        config.arch.memory.model = npu::MemoryModel::Axi;
        mutation(config.arch.memory.axi, config.arch.memory);
        rejects_with([&] { npu::validate(config); }, message);
    };
    bad([](auto& a, auto&) { a.data_bytes = 12; }, "AXI data bytes must be 8, 16, 32 or 64");
    bad([](auto& a, auto&) { a.data_bytes = 0; }, "AXI data bytes must be 8, 16, 32 or 64");
    bad([](auto& a, auto&) { a.max_burst_beats = 0; }, "AXI burst beats must be in 1..256");
    bad([](auto& a, auto&) { a.max_burst_beats = 257; }, "AXI burst beats must be in 1..256");
    bad([](auto& a, auto&) { a.read_outstanding = 0; }, "AXI outstanding bursts must be in 1..64");
    bad([](auto& a, auto&) { a.write_outstanding = 65; }, "AXI outstanding bursts must be");
    bad([](auto& a, auto&) { a.write_latency = 1000000001; }, "AXI latencies must be at most");
    bad([](auto&, auto& memory) { memory.dram_bytes_per_cycle = 1000000001; },
        "DRAM bytes per cycle must be at most 1000000000");
    // Every burst pays the latency in the bound: 1e9 cycles times 1e9 bytes
    // per cycle leaves 64 bits.
    bad(
        [](auto& a, auto& memory) {
            a.read_latency = 1000000000;
            memory.dram_bytes_per_cycle = 1000000000;
        },
        "simulation cycle bound times DRAM bytes per cycle must fit in 64 bits");
}

// Replays one core's weight-cache lookups in load order, as the loader makes
// them: an LRU over (channel panel, block) keys holding whole panels, dense or
// the encoded buffers in `panels`.
std::uint64_t replay_weight_reads(const npu::Config& c, const std::vector<npu::Tile>& tiles,
                                  const std::vector<npu::EncodedWeightPanel>& panels) {
    struct Line {
        std::uint64_t key, bytes;
    };
    std::vector<Line> lru;  // Most recent first.
    std::uint64_t cached = 0, reads = 0;
    const auto cache = c.arch.weight_cache_bytes;
    const auto blocks = npu::reduction_blocks(c);
    for (const auto& tile : tiles) {
        for (std::uint64_t block = 0; block < blocks; ++block) {
            const auto bytes = panels.empty() ?
                                   (npu::panel_k(c, block) * tile.cols * c.weights.bits + 7) / 8 :
                                   npu::weight_panel(c, panels, tile, block).buffer.size();
            const auto key = (tile.group * c.shape.n + tile.col) * blocks + block;
            const auto hit = std::find_if(lru.begin(), lru.end(),
                                          [key](const Line& line) { return line.key == key; });
            if (cache && hit != lru.end()) {
                const auto line = *hit;
                lru.erase(hit);
                lru.insert(lru.begin(), line);
                continue;
            }
            reads += bytes;
            if (!cache || bytes > cache) continue;
            while (cached + bytes > cache) {
                cached -= lru.back().bytes;
                lru.pop_back();
            }
            lru.insert(lru.begin(), {key, bytes});
            cached += bytes;
        }
    }
    return reads;
}

void check_cluster_plan() {
    // Units split into contiguous runs whose sizes differ by at most one.
    for (std::uint64_t units = 1; units <= 40; ++units) {
        for (std::uint64_t cores = 1; cores <= 9; ++cores) {
            npu::ClusterPlan plan;
            plan.cores = cores;
            plan.units = units;
            std::uint64_t next = 0;
            for (std::uint64_t core = 0; core < cores; ++core) {
                const auto count = plan.unit_count(core);
                require(plan.first_unit(core) == next, "runs are contiguous");
                require(count == units / cores || count == units / cores + 1, "balanced runs");
                require(core == 0 || count <= plan.unit_count(core - 1), "longer runs first");
                for (auto unit = next; unit < next + count; ++unit)
                    require(plan.unit_core(unit) == core, "a unit belongs to its run");
                next += count;
            }
            require(next == units, "every unit has a core");
        }
    }

    npu::Config c;
    c.arch.memory.model = npu::MemoryModel::Axi;
    c.shape = {10, 7, 9, 2, 1};
    c.mapping.tile_m = 3;
    c.mapping.tile_n = 2;
    c.arch.cores = 2;
    require(npu::cluster_plan(c, npu::Partition::M).split == npu::ClusterSplit::Groups &&
                npu::cluster_plan(c, npu::Partition::M).units == 2,
            "groups >= cores go to cores whole");
    c.arch.cores = 3;
    const auto rows = npu::cluster_plan(c, npu::Partition::M);
    const auto columns = npu::cluster_plan(c, npu::Partition::N);
    require(rows.split == npu::ClusterSplit::Rows && rows.units == 2 * 4, "row units");
    require(columns.split == npu::ClusterSplit::Columns && columns.units == 2 * 4, "column units");
    require(npu::to_string(npu::ClusterSplit::Columns) == "columns", "split names");
    rejects_with([&] { npu::cluster_plan(c, npu::Partition::Auto); }, "splits by m or n");

    // The closed-form estimate against a replay of every core's tiles: exact
    // read, weight and write bytes, MACs and compute cycles per core, dense or
    // timed from the encoded panels.
    std::uint64_t state = 99;
    const auto draw = [&state](std::uint64_t low, std::uint64_t high) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return low + (state >> 33) % (high - low + 1);
    };
    for (int round = 0; round < 400; ++round) {
        npu::Config r;
        r.arch.memory.model = npu::MemoryModel::Axi;
        r.shape = {draw(1, 30), draw(1, 30), draw(1, 40), draw(1, 4), 1};
        r.mapping.tile_m = draw(1, 9);
        r.mapping.tile_n = draw(1, 9);
        r.mapping.tile_k = draw(0, 1) ? 0 : draw(1, r.shape.k);
        r.arch.cores = draw(1, 9);
        r.arch.weight_cache_bytes = draw(0, 2) ? std::uint64_t{0} : draw(1, 600);
        r.weights.bits = draw(0, 1) ? 4 : 8;
        r.arch.compute.macs_per_cycle = draw(1, 64);
        r.arch.sram_bytes = 1ULL << 26;
        const auto encoding = draw(0, 2);
        r.weights.encoding = encoding == 0 ? npu::WeightEncoding::Dense :
                             encoding == 1 ? npu::WeightEncoding::Palette :
                                             npu::WeightEncoding::Sparse;
        r.weights.palette_bits = 8;
        r.weights.sparse_block = draw(1, 8);
        r.arch.compute.weight_decode_values_per_cycle = draw(1, 64);
        npu::validate(r);
        std::vector<npu::EncodedWeightPanel> panels;
        if (r.weights.encoding != npu::WeightEncoding::Dense) {
            auto problem = npu::make_problem(r);
            // Sparse: set three weights in four to the weight zero point so whole blocks drop.
            if (r.weights.encoding == npu::WeightEncoding::Sparse)
                for (auto& weight : problem.weights)
                    if (draw(0, 3)) weight = static_cast<std::int8_t>(r.quant.weight_zero);
            panels = npu::encode_weights(r, problem);
        }
        const auto costs =
            panels.empty() ? npu::dense_column_costs(r) : npu::encoded_column_costs(r, panels);
        const auto tiles = npu::make_tiles(r);
        for (const auto partition : {npu::Partition::M, npu::Partition::N}) {
            const auto plan = npu::cluster_plan(r, partition);
            const auto estimate = npu::estimate_cluster(r, plan, costs);
            std::vector<std::vector<npu::Tile>> assigned(plan.cores);
            for (const auto& tile : tiles)
                assigned[plan.unit_core(npu::tile_unit(r, plan, tile))].push_back(tile);
            std::uint64_t reads = 0;
            for (std::uint64_t core = 0; core < plan.cores; ++core) {
                const auto& e = estimate.cores[core];
                std::uint64_t activations = 0, params = 0, writes = 0, macs = 0, cycles = 0;
                for (const auto& tile : assigned[core]) {
                    activations += tile.rows * r.shape.k;
                    params += tile.cols * npu::param_record_bytes(r);
                    writes += npu::tile_write_bytes(tile);
                    macs += tile.rows * tile.cols * r.shape.k;
                    if (panels.empty()) cycles += npu::tile_compute_cycles(r, tile);
                    for (std::uint64_t block = 0;
                         !panels.empty() && block < npu::reduction_blocks(r); ++block)
                        cycles += npu::encoded_panel_compute_cycles(
                            r, tile, block, npu::weight_panel(r, panels, tile, block));
                }
                const auto weights = replay_weight_reads(r, assigned[core], panels);
                const auto label =
                    " (round " + std::to_string(round) + ", core " + std::to_string(core) + ")";
                require(e.tiles == assigned[core].size(), "tiles per core" + label);
                require(e.weight_bytes == weights, "weight-cache closed form" + label);
                require(e.read_bytes == activations + weights + params, "read bytes" + label);
                require(e.write_bytes == writes && e.macs == macs, "writes and MACs" + label);
                require(e.compute_cycles == cycles, "compute cycles" + label);
                reads += e.read_bytes;
            }
            require(estimate.read_bytes == reads, "estimate totals");
        }
    }

    // auto_partition: the smaller bound wins, then fewer bytes, then n.
    npu::ClusterEstimate m, n;
    m.bound_cycles = n.bound_cycles = 10;
    m.read_bytes = n.read_bytes = 5;
    require(npu::auto_partition(m, n) == npu::Partition::N, "ties go to n");
    m.read_bytes = 4;
    require(npu::auto_partition(m, n) == npu::Partition::M, "then fewer bytes");
    m.bound_cycles = 11;
    require(npu::auto_partition(m, n) == npu::Partition::N, "the bound decides first");

    npu::Config p;
    p.arch.memory.model = npu::MemoryModel::Axi;
    p.arch.memory.axi.data_bytes = 32;
    p.arch.cores = 4;
    require(npu::peak_memory_bytes_per_cycle(p) == 256, "every core's two channels");
    p.arch.memory.dram_bytes_per_cycle = 100;
    require(npu::peak_memory_bytes_per_cycle(p) == 100, "DRAM caps the cluster");
    npu::validate(p);
    p.arch.cores = 0;
    rejects_with([&] { npu::validate(p); }, "cores must be in 1..64");
    p.arch.cores = 65;
    rejects_with([&] { npu::validate(p); }, "cores must be in 1..64");
    p.arch.cores = 2;
    p.arch.memory.model = npu::MemoryModel::Serial;
    rejects_with([&] { npu::validate(p); }, "more than one core requires --memory-model axi");
}

// A deterministic draw in [low, high].
struct Draw {
    std::uint64_t state;
    std::uint64_t operator()(std::uint64_t low, std::uint64_t high) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return low + (state >> 33) % (high - low + 1);
    }
};

// A small random convolution whose geometry is valid: G groups of C_g input
// channels, N outputs per group, tile_k a whole number of kernel rows.
npu::Config random_conv(Draw& draw, bool depthwise) {
    while (true) {
        npu::Config c;
        auto& v = c.shape.conv;
        v.input_h = draw(1, 7);
        v.input_w = draw(1, 7);
        v.kernel_h = draw(1, 3);
        v.kernel_w = draw(1, 3);
        v.stride_h = draw(1, 3);
        v.stride_w = draw(1, 3);
        v.dilation_h = draw(1, 2);
        v.dilation_w = draw(1, 2);
        v.pad_top = draw(0, 2);
        v.pad_left = draw(0, 2);
        v.pad_bottom = draw(0, 2);
        v.pad_right = draw(0, 2);
        const auto group_channels = draw(1, 3);
        c.shape.groups = depthwise ? draw(2, 6) : draw(1, 3);
        v.input_c = c.shape.groups * group_channels;
        c.shape.n = depthwise ? 1 : draw(1, 3);
        const auto reach_h = v.dilation_h * (v.kernel_h - 1) + 1;
        const auto reach_w = v.dilation_w * (v.kernel_w - 1) + 1;
        if (reach_h > v.input_h + v.pad_top + v.pad_bottom ||
            reach_w > v.input_w + v.pad_left + v.pad_right)
            continue;
        const auto out_h = (v.input_h + v.pad_top + v.pad_bottom - reach_h) / v.stride_h + 1;
        const auto out_w = (v.input_w + v.pad_left + v.pad_right - reach_w) / v.stride_w + 1;
        c.shape.m = out_h * out_w;
        c.shape.k = v.kernel_h * v.kernel_w * group_channels;
        c.shape.seed = draw(1, 1000);
        c.mapping.tile_m = draw(1, c.shape.m);
        c.mapping.tile_n = draw(1, 4);
        c.mapping.tile_k = draw(0, v.kernel_h) * v.kernel_w * group_channels;
        c.quant.input_zero = static_cast<std::int32_t>(draw(0, 20)) - 10;
        c.arch.sram_bytes = 1ULL << 24;
        if (depthwise) c.arch.compute.depthwise_macs_per_cycle = draw(1, 40);
        return c;
    }
}

// Every input pixel the output pixels [first, first + count) read over kernel
// rows [ky_first, ky_first + ky_count), by enumeration of every tap.
std::map<std::uint64_t, std::set<std::uint64_t>> needed_pixels(const npu::Config& c,
                                                               std::uint64_t first,
                                                               std::uint64_t count,
                                                               std::uint64_t ky_first,
                                                               std::uint64_t ky_count) {
    const auto& v = c.shape.conv;
    const auto out_w = npu::conv_geometry(c.shape).output_w;
    std::map<std::uint64_t, std::set<std::uint64_t>> pixels;
    for (auto m = first; m < first + count; ++m)
        for (auto ky = ky_first; ky < ky_first + ky_count; ++ky)
            for (std::uint64_t kx = 0; kx < v.kernel_w; ++kx) {
                const auto y =
                    static_cast<std::int64_t>(m / out_w * v.stride_h + ky * v.dilation_h) -
                    static_cast<std::int64_t>(v.pad_top);
                const auto x =
                    static_cast<std::int64_t>(m % out_w * v.stride_w + kx * v.dilation_w) -
                    static_cast<std::int64_t>(v.pad_left);
                if (y >= 0 && x >= 0 && y < static_cast<std::int64_t>(v.input_h) &&
                    x < static_cast<std::int64_t>(v.input_w))
                    pixels[static_cast<std::uint64_t>(y)].insert(static_cast<std::uint64_t>(x));
            }
    return pixels;
}

// Direct convolution over the NHWC input, without the im2col operand.
std::vector<std::int32_t> direct_conv(const npu::Config& c, const npu::Problem& p) {
    const auto& v = c.shape.conv;
    const auto g = npu::conv_geometry(c.shape);
    const auto channels = c.shape.groups * c.shape.n;
    std::vector<std::int32_t> result;
    for (std::uint64_t oy = 0; oy < g.output_h; ++oy)
        for (std::uint64_t ox = 0; ox < g.output_w; ++ox)
            for (std::uint64_t channel = 0; channel < channels; ++channel) {
                const auto group = channel / c.shape.n, out = channel % c.shape.n;
                std::int64_t sum = p.bias[channel];
                for (std::uint64_t ky = 0; ky < v.kernel_h; ++ky)
                    for (std::uint64_t kx = 0; kx < v.kernel_w; ++kx) {
                        const auto y =
                            static_cast<std::int64_t>(oy * v.stride_h + ky * v.dilation_h) -
                            static_cast<std::int64_t>(v.pad_top);
                        const auto x =
                            static_cast<std::int64_t>(ox * v.stride_w + kx * v.dilation_w) -
                            static_cast<std::int64_t>(v.pad_left);
                        // Padding holds the input zero point and adds nothing.
                        if (y < 0 || x < 0 || y >= static_cast<std::int64_t>(v.input_h) ||
                            x >= static_cast<std::int64_t>(v.input_w))
                            continue;
                        for (std::uint64_t ci = 0; ci < g.group_channels; ++ci) {
                            const auto in = p.input[(static_cast<std::uint64_t>(y) * v.input_w +
                                                     static_cast<std::uint64_t>(x)) *
                                                        v.input_c +
                                                    group * g.group_channels + ci];
                            const auto kk = (ky * v.kernel_w + kx) * g.group_channels + ci;
                            const auto w = p.weights[(group * c.shape.k + kk) * c.shape.n + out];
                            sum += static_cast<std::int64_t>(in - c.quant.input_zero) *
                                   (w - c.quant.weight_zero);
                        }
                    }
                result.push_back(static_cast<std::int32_t>(sum));
            }
    return result;
}

void check_convolution_geometry() {
    // MobileNetV2 block_1_depthwise with TF SAME padding (0, 1, 0, 1).
    npu::Config dw;
    dw.shape.m = 3136;
    dw.shape.n = 1;
    dw.shape.k = 9;
    dw.shape.groups = 96;
    dw.shape.conv = {112, 112, 96, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1};
    const auto same = npu::conv_geometry(dw.shape);
    require(same.output_h == 56 && same.output_w == 56 && same.group_channels == 1 &&
                same.kernel_row_k == 3,
            "TF SAME stride 2 on 112 gives 56");
    dw.shape.conv = {112, 112, 96, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1};
    require(npu::conv_geometry(dw.shape).output_w == 56, "pad 1 stride 2 on 112 gives 56");
    // ResNet-50 layer1 3x3: 56 x 56 x 64, pad 1.
    npu::Config res;
    res.shape.m = 3136;
    res.shape.n = 64;
    res.shape.k = 576;
    res.shape.conv = {56, 56, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1};
    const auto res_geometry = npu::conv_geometry(res.shape);
    require(res_geometry.kernel_row_k == 192 && res_geometry.output_h == 56, "res2 geometry");
    res.mapping.tile_k = 384;
    require(npu::reduction_blocks(res) == 2 && npu::conv_kernel_rows(res, 1).first == 2 &&
                npu::conv_kernel_rows(res, 1).count == 1,
            "a panel of two kernel rows, then the last row");
    res.mapping.tile_k = 0;
    require(npu::conv_kernel_rows(res, 0).count == 3, "tile_k 0 takes every kernel row");
    require(!npu::implicit_conv(npu::Config{}.shape) &&
                npu::conv_geometry(npu::Config{}.shape).output_h == 0,
            "the default is a plain GEMM");

    const auto conv_error = [&res](const std::function<void(npu::Config&)>& mutation,
                                   const std::string& expected) {
        auto c = res;
        mutation(c);
        rejects_with([&] { npu::validate(c); }, expected);
    };
    conv_error([](auto& c) { c.shape.m = 3135; },
               "M must equal output height x width = 56 x 56 = 3136, got 3135");
    conv_error([](auto& c) { c.shape.k = 575; },
               "K must equal kernel_h x kernel_w x input_c / groups = 576, got 575");
    conv_error(
        [](auto& c) {
            c.shape.groups = 3;
            c.shape.k = 192;
        },
        "input channels 64 must be a multiple of groups 3");
    conv_error([](auto& c) { c.shape.conv.input_w = 0; },
               "input height, width and channels must all be in 1..1048576");
    conv_error([](auto& c) { c.shape.conv.stride_h = 0; },
               "kernel, stride and dilation must be in 1..1048576");
    conv_error([](auto& c) { c.shape.conv.dilation_w = 40; },
               "kernel width extent 81 exceeds "
               "the padded input 58");
    conv_error([](auto& c) { c.mapping.tile_k = 256; },
               "--tile-k must be 0 or a multiple of kernel_w x input_c / groups = 192 (whole "
               "kernel rows), got 256");
    npu::Config loose;
    loose.shape.conv.kernel_h = 3;
    rejects_with([&] { npu::validate(loose); }, "need the input extents --conv-input-h");
    // The NHWC input counts against the host storage of a bit-exact run: a
    // 256 MiB input with a stride-16 1x1 kernel exceeds it, timing alone does not.
    npu::Config big;
    big.shape.conv = {16384, 16384, 1, 1, 1, 16, 16, 1, 1, 0, 0, 0, 0};
    big.shape.m = 1024 * 1024;
    big.shape.n = 1;
    big.shape.k = 1;
    rejects_with([&] { npu::validate(big); },
                 "estimated simulator working storage exceeds 256 MiB");
    big.sim.timing_only = true;
    npu::validate(big);
}

void check_convolution_footprint() {
    Draw draw{77};
    std::vector<npu::InputSpan> spans;
    std::vector<npu::AddressRun> runs;
    for (int round = 0; round < 300; ++round) {
        const bool depthwise = round % 3 == 0;
        auto c = random_conv(draw, depthwise);
        npu::validate(c);
        require(npu::channel_wise(c) == depthwise, "a depthwise draw runs channel-wise");
        const auto& v = c.shape.conv;
        const auto g = npu::conv_geometry(c.shape);
        const auto label = " (round " + std::to_string(round) + ")";
        std::uint64_t largest = 0;
        for (const auto& tile : npu::make_tiles(c)) {
            const auto first_group = depthwise ? tile.col : tile.group;
            const auto groups = depthwise ? tile.cols : 1;
            const auto channels = groups * g.group_channels;
            for (std::uint64_t block = 0; block < npu::reduction_blocks(c); ++block) {
                const auto rows = npu::conv_kernel_rows(c, block);
                const auto pixels = needed_pixels(c, tile.row, tile.rows, rows.first, rows.count);
                npu::conv_input_spans(c, tile.row, tile.rows, block, spans);
                require(spans.size() == pixels.size(), "one span per needed input row" + label);
                std::uint64_t width = 0;
                std::vector<npu::AddressRun> expected;
                for (std::size_t i = 0; i < spans.size(); ++i) {
                    const auto entry = std::next(pixels.begin(), static_cast<std::ptrdiff_t>(i));
                    require(spans[i].row == entry->first &&
                                spans[i].first == *entry->second.begin() &&
                                spans[i].last == *entry->second.rbegin(),
                            "a span is the hull of its row's needed columns" + label);
                    width += spans[i].last - spans[i].first + 1;
                    const auto row_base =
                        npu::activation_base + spans[i].row * v.input_w * v.input_c;
                    if (channels == v.input_c) {
                        expected.push_back({row_base + spans[i].first * v.input_c,
                                            (spans[i].last - spans[i].first + 1) * v.input_c});
                    } else {
                        for (auto x = spans[i].first; x <= spans[i].last; ++x)
                            expected.push_back(
                                {row_base + x * v.input_c + first_group * g.group_channels,
                                 channels});
                    }
                }
                require(npu::activation_panel_bytes(c, tile, block) == width * channels,
                        "panel activation bytes are the footprint" + label);
                const auto widest = std::min(c.mapping.tile_n, npu::grid_columns(c));
                largest = std::max(largest, width * (depthwise ? widest : 1) * g.group_channels);
                npu::panel_read_runs(c, tile, block, 0, 0, runs);
                if (block == 0) runs.pop_back();  // the bias records
                require(runs.size() == expected.size(), "activation run count" + label);
                for (std::size_t i = 0; i < runs.size(); ++i)
                    require(
                        runs[i].base == expected[i].base && runs[i].length == expected[i].length,
                        "activation run " + std::to_string(i) + label);
                npu::panel_read_runs(c, tile, block, 0, 0, runs, false);
                require(runs.size() == (block == 0 ? 1U : 0U), "a cached panel reads no input");
            }
        }
        // A panel buffer holds the largest footprint of any tile and the
        // deepest weight panel.
        const auto rows = std::min(c.mapping.tile_m, c.shape.m);
        const auto cols = std::min(c.mapping.tile_n, npu::grid_columns(c));
        const auto depth = std::min(c.shape.k, c.mapping.tile_k ? c.mapping.tile_k : c.shape.k);
        const auto buffers = std::min<std::uint64_t>(2, npu::reduction_blocks(c));
        require(npu::scratchpad_bytes(c) ==
                    2 * (4 * cols + buffers * (largest + depth * cols) + 5 * rows * cols),
                "the slots hold the largest input footprint" + label);
        // The operand is the im2col view of the generated input; the
        // reference and the tiled datapaths agree with a direct convolution.
        const auto p = npu::make_problem(c);
        require(p.input.size() == v.input_h * v.input_w * v.input_c &&
                    p.activations == npu::im2col(c, p.input),
                "the operand is built from the NHWC input" + label);
        const auto expected = npu::reference(p, c.quant);
        require(expected.accumulators == direct_conv(c, p),
                "the im2col GEMM equals a direct convolution" + label);
        npu::Output tiled;
        for (const auto& tile : npu::make_tiles(c)) {
            if (depthwise) npu::compute_channel_tile(p, c.quant, tile, tiled);
            else npu::compute_tile(p, c.quant, tile, tiled);
        }
        require(
            tiled.accumulators == expected.accumulators && tiled.quantized == expected.quantized,
            "tiles reproduce the reference" + label);
    }
}

void check_depthwise_datapath() {
    npu::Config c;
    c.shape.m = 7;
    c.shape.n = 1;
    c.shape.k = 9;
    c.shape.groups = 10;
    c.shape.seed = 5;
    c.mapping.tile_m = 3;
    c.mapping.tile_n = 4;
    c.arch.compute.pipeline_cycles = 2;
    require(!npu::channel_wise(c) && npu::grid_groups(c) == 10 && npu::grid_columns(c) == 1,
            "rate 0 keeps grouped GEMMs");
    c.arch.compute.depthwise_macs_per_cycle = 8;
    require(npu::channel_wise(c) && npu::grid_groups(c) == 1 && npu::grid_columns(c) == 10 &&
                npu::peak_macs_per_cycle(c) == 8,
            "the depthwise datapath takes the problem");
    const auto tiles = npu::make_tiles(c);
    require(tiles.size() == 9 && npu::tile_count(c) == 9, "three row tiles of three channel tiles");
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        const auto& t = tiles[i];
        require(t.group == 0 && t.row == i / 3 * 3 && t.col == i % 3 * 4 &&
                    t.rows == std::min<std::uint64_t>(3, 7 - t.row) &&
                    t.cols == std::min<std::uint64_t>(4, 10 - t.col),
                "row tile outer, channel tile inner");
    }
    std::vector<npu::AddressRun> runs;
    for (const auto tile_k : {UINT64_C(0), UINT64_C(4)}) {
        c.mapping.tile_k = tile_k;
        std::uint64_t cursor = 0;
        for (std::uint64_t col = 0; col < 10; col += 4)
            for (std::uint64_t block = 0; block < npu::reduction_blocks(c); ++block) {
                const auto cols = std::min<std::uint64_t>(4, 10 - col);
                require(npu::dense_weight_offset(c, {0, 0, col, 3, cols, 0}, block) == cursor,
                        "channel-tile weight panels pack in load order");
                cursor += npu::panel_k(c, block) * cols;
            }
        for (const auto& t : tiles) {
            for (std::uint64_t block = 0; block < npu::reduction_blocks(c); ++block) {
                const auto depth = npu::panel_k(c, block);
                require(npu::panel_compute_cycles(c, t, block) ==
                            npu::ceil_div(t.rows * t.cols * depth, 8) + 2,
                        "ceil(M_t * C_t * K_p / rate) + pipeline");
                require(npu::activation_panel_bytes(c, t, block) == t.rows * t.cols * depth,
                        "M_t x C_t x K_p activation bytes");
                npu::panel_read_runs(c, t, block, 5, t.cols * depth, runs);
                const auto per_row = tile_k ? t.cols : 1;
                require(runs.size() == t.rows * per_row + 1 + (block == 0), "panel run count");
                for (std::uint64_t r = 0; r < t.rows; ++r)
                    for (std::uint64_t j = 0; j < per_row; ++j) {
                        const auto& run = runs[r * per_row + j];
                        const auto base =
                            npu::activation_base + (t.row + r) * 90 + (t.col + j) * 9 + block * 4;
                        require(run.base == base && run.length == (tile_k ? depth : t.cols * 9),
                                "one run per row over all of K, else one per channel");
                    }
                require(runs[t.rows * per_row].base == npu::weight_base + 5 &&
                            runs[t.rows * per_row].length == t.cols * depth,
                        "the weight panel holds C_t x K_p");
                if (block == 0)
                    require(runs.back().base == npu::param_base + t.col * 4 &&
                                runs.back().length == t.cols * 4,
                            "one bias record per channel");
            }
            npu::tile_store_runs(c, t, runs);
            require(runs.size() == t.rows &&
                        runs[0].base == npu::output_base + t.row * 10 + t.col &&
                        runs[0].length == t.cols,
                    "a store writes M_t rows of C_t channels");
        }
        // Two slots of C_t-wide panels: bias, the panel buffers (one per
        // reduction panel, at most two), sums and outputs.
        const std::uint64_t depth = tile_k ? 4 : 9, buffers = tile_k ? 2 : 1;
        require(npu::scratchpad_bytes(c) ==
                    2 * (4 * 4 + buffers * (3 * 4 * depth + 4 * depth) + 5 * 3 * 4),
                "scratchpad of channel tiles");
    }
    c.mapping.tile_k = 0;
    const auto p = npu::make_problem(c);
    const auto expected = npu::reference(p, c.quant);
    npu::Output output;
    for (auto t = tiles.rbegin(); t != tiles.rend(); ++t)
        npu::compute_channel_tile(p, c.quant, *t, output);
    require(output.accumulators == expected.accumulators && output.quantized == expected.quantized,
            "channel tiles reproduce the reference");
    rejects_with([&] { npu::compute_channel_tile(p, c.quant, {0, 0, 8, 1, 3, 0}, output); },
                 "tile is outside the output matrix");
    // The activation cache keys channel tiles apart; GEMM column tiles of one
    // row tile share a key.
    require(npu::activation_cache_key(c, tiles[0], 0) != npu::activation_cache_key(c, tiles[1], 0),
            "channel tiles never share activation panels");
    npu::Config gemm;
    gemm.shape.m = 6;
    gemm.shape.n = 9;
    gemm.shape.k = 8;
    gemm.shape.groups = 2;
    gemm.mapping.tile_m = 3;
    gemm.mapping.tile_n = 4;
    gemm.mapping.tile_k = 4;
    const auto key = [&gemm](std::uint64_t row, std::uint64_t col, std::uint64_t group,
                             std::uint64_t block) {
        return npu::activation_cache_key(gemm, {0, row, col, 3, 1, group}, block);
    };
    require(key(0, 0, 0, 0) == key(0, 4, 0, 0) && key(0, 0, 0, 0) != key(3, 0, 0, 0) &&
                key(0, 0, 0, 0) != key(0, 0, 1, 0) && key(0, 0, 0, 0) != key(0, 0, 0, 1),
            "a GEMM key is (group, row tile, block)");
    gemm.arch.activation_cache_bytes = 100;
    require(
        npu::scratchpad_bytes(gemm) ==
            [&gemm] {
                auto none = gemm;
                none.arch.activation_cache_bytes = 0;
                return npu::scratchpad_bytes(none);
            }() +
                100,
        "the activation cache counts in SRAM");
    gemm.arch.activation_cache_bytes = 1000000001;
    rejects_with([&] { npu::validate(gemm); }, "activation cache size must be at most");

    auto encoded = c;
    encoded.weights.encoding = npu::WeightEncoding::Palette;
    rejects_with([&] { npu::validate(encoded); }, "the depthwise datapath reads dense weights");
    encoded.shape.n = 2;
    npu::validate(encoded);
    require(!npu::channel_wise(encoded), "N > 1 stays a grouped GEMM");
    c.arch.compute.depthwise_macs_per_cycle = 1000000001;
    rejects_with([&] { npu::validate(c); }, "depthwise MACs per cycle must be at most");
}

// Replays one core's activation-cache lookups in load order: an LRU over
// activation_cache_key lines of activation_panel_bytes, returning the bytes
// read and counting the hits.
std::uint64_t replay_activation_reads(const npu::Config& c, const std::vector<npu::Tile>& tiles,
                                      std::uint64_t& hits) {
    struct Line {
        std::uint64_t key, bytes;
    };
    std::vector<Line> lru;  // Most recent first.
    std::uint64_t cached = 0, reads = 0;
    const auto cache = c.arch.activation_cache_bytes;
    for (const auto& tile : tiles) {
        for (std::uint64_t block = 0; block < npu::reduction_blocks(c); ++block) {
            const auto bytes = npu::activation_panel_bytes(c, tile, block);
            const auto key = npu::activation_cache_key(c, tile, block);
            const auto hit = std::find_if(lru.begin(), lru.end(),
                                          [key](const Line& line) { return line.key == key; });
            if (cache && hit != lru.end()) {
                const auto line = *hit;
                lru.erase(hit);
                lru.insert(lru.begin(), line);
                ++hits;
                continue;
            }
            reads += bytes;
            if (!cache || bytes > cache) continue;
            while (cached + bytes > cache) {
                cached -= lru.back().bytes;
                lru.pop_back();
            }
            lru.insert(lru.begin(), {key, bytes});
            cached += bytes;
        }
    }
    return reads;
}

void check_activation_estimate() {
    // The closed-form activation reads against a replay of every core's
    // activation cache, for GEMMs, convolutions and the depthwise datapath.
    Draw draw{4242};
    std::uint64_t total_hits = 0;
    for (int round = 0; round < 400; ++round) {
        npu::Config r;
        const auto kind = round % 4;  // GEMM, convolution, depthwise GEMM, depthwise conv
        if (kind == 1 || kind == 3) {
            r = random_conv(draw, kind == 3);
        } else {
            r.shape = {draw(1, 30), kind == 2 ? 1 : draw(1, 30), draw(1, 30), draw(2, 6), 1};
            if (kind == 0) r.shape.groups = draw(1, 4);
            r.mapping.tile_m = draw(1, 9);
            r.mapping.tile_n = draw(1, 9);
            r.mapping.tile_k = draw(0, 1) ? 0 : draw(1, r.shape.k);
            if (kind == 2) r.arch.compute.depthwise_macs_per_cycle = draw(1, 32);
        }
        r.arch.memory.model = npu::MemoryModel::Axi;
        r.arch.cores = draw(1, 5);
        r.arch.activation_cache_bytes = draw(0, 3) ? draw(1, 400) : std::uint64_t{0};
        r.arch.weight_cache_bytes = draw(0, 1) ? std::uint64_t{0} : draw(1, 300);
        r.arch.sram_bytes = 1ULL << 26;
        npu::validate(r);
        const auto costs = npu::dense_column_costs(r);
        const auto tiles = npu::make_tiles(r);
        for (const auto partition : {npu::Partition::M, npu::Partition::N}) {
            const auto plan = npu::cluster_plan(r, partition);
            const auto estimate = npu::estimate_cluster(r, plan, costs);
            const auto assigned = npu::make_core_tiles(r, plan);
            for (std::uint64_t core = 0; core < plan.cores; ++core) {
                const auto& e = estimate.cores[core];
                std::uint64_t params = 0, cycles = 0, macs = 0;
                for (const auto& tile : assigned[core]) {
                    params += tile.cols * npu::param_record_bytes(r);
                    cycles += npu::tile_compute_cycles(r, tile);
                    macs += tile.rows * tile.cols * r.shape.k;
                }
                const auto activations = replay_activation_reads(r, assigned[core], total_hits);
                const auto weights = replay_weight_reads(r, assigned[core], {});
                const auto label =
                    " (round " + std::to_string(round) + ", core " + std::to_string(core) + ")";
                require(e.weight_bytes == weights, "weight bytes" + label);
                require(e.read_bytes == activations + weights + params,
                        "activation-cache closed form" + label);
                require(e.compute_cycles == cycles && e.macs == macs, "compute" + label);
                require(e.tiles == assigned[core].size(), "tiles" + label);
            }
        }
    }
    require(total_hits > 0, "the rounds exercise activation-cache hits");
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
        check_decoder_overlap();
        check_limits();
        check_validation();
        check_memory_map();
        check_cluster_plan();
        check_convolution_geometry();
        check_convolution_footprint();
        check_depthwise_datapath();
        check_activation_estimate();
        std::cout << "core tests: all checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "core tests FAILED: " << error.what() << '\n';
        return 1;
    }
}
