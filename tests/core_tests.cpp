#include "npu/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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
    npu::Quantization q;
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
    q.shift = 0;
    q.multiplier = UINT32_MAX;
    rejects<std::invalid_argument>([&] { npu::requantize(0, q); }, "multiplier exceeds INT32");
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

void check_known_dense() {
    npu::Problem p;
    p.m = p.n = 2;
    p.k = 3;
    p.activations = {1, 2, 3, -1, 0, 2};
    p.weights = {4, -2, 5, 1, -3, 2};
    p.bias = {7, -5};
    npu::Quantization q;
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

void check_fixture() {
    npu::Config c;
    c.m = 2;
    c.n = 2;
    c.k = 2;
    const auto p = npu::make_problem(c);
    require(p.activations == std::vector<std::int8_t>{-9, 6, -5, 5},
            "LCG activation golden fixture");
    require(p.weights == std::vector<std::int8_t>{13, -10, 0, 4},
            "LCG weight golden fixture");
    require(p.bias == std::vector<std::int32_t>{39, 35}, "LCG bias golden fixture");
    c.seed += UINT64_C(1) << 32;
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
    c.m = 5;
    c.n = 7;
    c.k = 3;
    c.tile_m = 3;
    c.tile_n = 4;
    c.macs_per_cycle = 7;
    c.pipeline_cycles = 2;
    c.buffer_slots = 2;
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
                ++coverage[static_cast<std::size_t>(r * c.n + col)];
    require(std::all_of(coverage.begin(), coverage.end(), [](unsigned n) { return n == 1; }),
            "tiles cover every output exactly once");
    c.tile_m = c.tile_n = 4096;
    require(npu::make_tiles(c).size() == 1, "oversized mapping handles one edge tile");
    require(npu::scratchpad_bytes(c) == 2 * (15 + 21 + 28 + 5 * 35),
            "oversized mapping allocates actual extents");
}

void check_random_edges() {
    std::uint32_t state = 0x361c1eU;
    const auto next = [&state]() {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        return state;
    };
    for (unsigned trial = 0; trial < 64; ++trial) {
        npu::Config c;
        c.m = 1 + next() % 19;
        c.n = 1 + next() % 17;
        c.k = 1 + next() % 23;
        c.tile_m = 1 + next() % 11;
        c.tile_n = 1 + next() % 9;
        c.seed = next();
        c.quant.input_zero = static_cast<std::int32_t>(next() % 256) - 128;
        c.quant.weight_zero = static_cast<std::int32_t>(next() % 256) - 128;
        c.quant.output_zero = static_cast<std::int32_t>(next() % 256) - 128;
        c.quant.multiplier = next() % 257;
        c.quant.shift = next() % 18;
        c.quant.relu = next() % 2 != 0;
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
    npu::Quantization q;
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

void check_validation() {
    npu::validate(npu::Config{});
    const auto bad = [](const std::function<void(npu::Config&)>& mutation) {
        npu::Config c;
        mutation(c);
        rejects([&] { npu::validate(c); }, "invalid configuration");
    };
    bad([](auto& c) { c.m = 0; });
    bad([](auto& c) { c.n = 4097; });
    bad([](auto& c) { c.k = UINT64_MAX; });
    bad([](auto& c) { c.m = c.n = c.k = 4096; });
    bad([](auto& c) { c.tile_m = 0; });
    bad([](auto& c) { c.tile_n = 4097; });
    bad([](auto& c) { c.macs_per_cycle = 0; });
    bad([](auto& c) { c.memory_bytes_per_cycle = 0; });
    bad([](auto& c) { c.memory_latency = UINT64_MAX; });
    bad([](auto& c) { c.pipeline_cycles = 1000000001; });
    bad([](auto& c) { c.buffer_slots = 0; });
    bad([](auto& c) { c.buffer_slots = 65; });
    bad([](auto& c) { c.sram_bytes = 1; });
    bad([](auto& c) { c.frequency_mhz = 0; });
    bad([](auto& c) { c.frequency_mhz = 1e-300; });
    bad([](auto& c) { c.frequency_mhz = std::numeric_limits<double>::quiet_NaN(); });
    bad([](auto& c) { c.frequency_mhz = std::numeric_limits<double>::infinity(); });
    bad([](auto& c) { c.timing = "rtl"; });
    bad([](auto& c) { c.quant.output_zero = 128; });
    bad([](auto& c) {
        c.m = c.n = 4096; c.k = 1;
        c.tile_m = c.tile_n = 1;
        c.sram_bytes = UINT64_MAX;
    }); // Tile metadata alone exceeds the host memory safety budget.
    bad([](auto& c) {
        c.m = c.n = 1000; c.k = 1;
        c.tile_m = c.tile_n = 1;
    }); // Trace/metadata storage alone exceeds the host safety budget.
    npu::Config c;
    c.frequency_mhz = 1e-6;
    npu::validate(c); // A 1 Hz target remains representable and valid.
    c.sram_bytes = npu::scratchpad_bytes(c);
    npu::validate(c);
    --c.sram_bytes;
    rejects<std::invalid_argument>([&] { npu::validate(c); }, "exact SRAM capacity boundary");

    npu::Config slow;
    slow.m = slow.n = slow.k = slow.tile_m = slow.tile_n = slow.buffer_slots = 1;
    slow.memory_latency = slow.pipeline_cycles = 1000000000;
    npu::validate(slow); // Event mode jumps directly across long stage waits.
    slow.timing = "cycle";
    rejects<std::invalid_argument>([&] { npu::validate(slow); },
                                   "cycle mode rejects impractical wait iteration count");
    slow.macs_per_cycle = slow.memory_bytes_per_cycle = 1;
    slow.memory_latency = 0;
    slow.pipeline_cycles = 9999992;
    npu::validate(slow); // Read6 + compute(1+9999992) + write1 = 10000000.
    ++slow.pipeline_cycles;
    rejects<std::invalid_argument>([&] { npu::validate(slow); },
                                   "cycle mode rejects one iteration beyond boundary");
    slow.pipeline_cycles = 0;
    npu::validate(slow); // The minimal mapping costs eight serial cycles.
}

}  // namespace

int main() {
    try {
        check_arithmetic();
        check_requantization();
        check_known_dense();
        check_fixture();
        check_tiles_and_timing();
        check_random_edges();
        check_overflow_and_shapes();
        check_validation();
        std::cout << "core tests: all checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "core tests FAILED: " << error.what() << '\n';
        return 1;
    }
}
