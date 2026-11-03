#include "npu/core.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Operation>
void rejects(Operation operation, const char* message) {
    bool threw = false;
    try { operation(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}

void independent_storage(const std::string& codec) {
    npu::Config c;
    c.m = 3; c.n = 3; c.k = 5;
    c.tile_m = 2; c.tile_n = 2; c.tile_k = 3;
    c.weight_encoding = codec;
    c.palette_bits = 2; c.sparse_block = 3;
    c.quant.weight_zero_points = {1, -2, 3};
    npu::Problem p;
    p.m = c.m; p.n = c.n; p.k = c.k;
    p.activations = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15};
    p.weights = {1, -2, 3, 4, -2, 3, 1, -5, 3, 1, -2, 6, 1, -2, 3};
    p.bias = {19, -27, 31};
    const auto expected = npu::reference(p, c.quant);
    const auto encoded = npu::encode_weights(c, p);
    require(encoded.size() == 4, "encoding contains each column/K panel exactly once");
    // A simulated datapath that secretly reads the original expanded weights
    // will now fail. The encoded buffer is the numerical source of truth.
    std::fill(p.weights.begin(), p.weights.end(), static_cast<std::int8_t>(7));
    require(npu::reference(p, c.quant).accumulators != expected.accumulators,
            "expanded weight mutation must materially change the scalar result");
    npu::Output actual;
    const auto tiles = npu::make_tiles(c);
    for (const auto& tile : tiles) npu::compute_tile_encoded(p, c, encoded, tile, actual);
    require(actual.accumulators == expected.accumulators && actual.quantized == expected.quantized,
            "encoded datapath must be independent of expanded reference weights");

    auto corrupted = encoded;
    corrupted.front().buffer.clear();
    rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
            "truncated encoded header must be rejected");
    corrupted = encoded;
    corrupted.front().buffer[3] = 1;
    rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
            "nonzero reserved header must be rejected");
    corrupted = encoded;
    corrupted.front().buffer.push_back(0);
    rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
            "extra payload bytes must be rejected");
    corrupted = encoded;
    corrupted.front().buffer[2] = 7;
    rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
            "incorrect codec bit width must be rejected");
    corrupted = encoded;
    ++corrupted.front().k_start;
    rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
            "misaligned encoded panel must be rejected");
    corrupted = encoded;
    if (codec == "sparse") {
        ++corrupted.front().retained_values;
        rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
                "sparse retained count must match bitmap and payload");
    } else {
        corrupted.front().buffer[0] = 0;
        corrupted.front().buffer[1] = 0;
        rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
                "empty palette must be rejected");
    }
}

void complete_int8_palette() {
    npu::Config c;
    c.m = 1; c.n = 1; c.k = 256;
    c.tile_m = 1; c.tile_n = 1;
    c.weight_encoding = "palette"; c.palette_bits = 8;
    auto p = npu::make_problem(c);
    for (int value = -128; value <= 127; ++value)
        p.weights.at(static_cast<std::size_t>(value + 128)) = static_cast<std::int8_t>(value);
    const auto encoded = npu::encode_weights(c, p);
    require(encoded.front().buffer[0] == 0 && encoded.front().buffer[1] == 1,
            "256-entry LUT count must occupy two header bytes");
    require(encoded.front().buffer.size() == 4 + 256 + 256,
            "full INT8 palette must account for header, LUT and packed indices");
    npu::Output actual;
    npu::compute_tile_encoded(p, c, encoded, npu::make_tiles(c).front(), actual);
    require(actual.accumulators == npu::reference(p, c.quant).accumulators,
            "all signed INT8 values must round-trip through LUT");
}
}  // namespace

int main() {
    try {
        independent_storage("palette");
        independent_storage("sparse");
        complete_int8_palette();
        std::cout << "Compressed storage, corruption guards and full INT8 LUT verified\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
