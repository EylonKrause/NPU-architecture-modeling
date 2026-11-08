#include "npu/core.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Operation>
void rejects(Operation operation, const char* message) {
    bool threw = false;
    try {
        operation();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

void encoded_panels_only(npu::WeightEncoding codec) {
    npu::Config c;
    c.shape.m = 3;
    c.shape.n = 3;
    c.shape.k = 5;
    c.mapping.tile_m = 2;
    c.mapping.tile_n = 2;
    c.mapping.tile_k = 3;
    c.weights.encoding = codec;
    c.weights.palette_bits = 2;
    c.weights.sparse_block = 3;
    c.quant.weight_zero_points = {1, -2, 3};
    npu::Problem p;
    p.m = c.shape.m;
    p.n = c.shape.n;
    p.k = c.shape.k;
    p.activations = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15};
    p.weights = {1, -2, 3, 4, -2, 3, 1, -5, 3, 1, -2, 6, 1, -2, 3};
    p.bias = {19, -27, 31};
    const auto expected = npu::reference(p, c.quant);
    const auto encoded = npu::encode_weights(c, p);
    require(encoded.size() == 4, "one encoded panel per column panel and K panel");
    // Overwrite the expanded weights: the reference result changes and the encoded
    // datapath's must not.
    std::fill(p.weights.begin(), p.weights.end(), static_cast<std::int8_t>(7));
    require(npu::reference(p, c.quant).accumulators != expected.accumulators,
            "overwritten weights change the reference result");
    npu::Output actual;
    const auto tiles = npu::make_tiles(c);
    for (const auto& tile : tiles) npu::compute_tile_encoded(p, c, encoded, tile, actual);
    require(actual.accumulators == expected.accumulators && actual.quantized == expected.quantized,
            "encoded datapath reads only the encoded panels");

    auto corrupted = encoded;
    corrupted.front().buffer.clear();
    rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
            "truncated header");
    corrupted = encoded;
    corrupted.front().buffer[3] = 1;
    rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
            "nonzero reserved header byte");
    corrupted = encoded;
    corrupted.front().buffer.push_back(0);
    rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
            "trailing payload bytes");
    corrupted = encoded;
    corrupted.front().buffer[2] = 7;
    rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
            "wrong codec bit width");
    corrupted = encoded;
    ++corrupted.front().k_start;
    rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
            "misaligned panel");
    corrupted = encoded;
    if (codec == npu::WeightEncoding::Sparse) {
        ++corrupted.front().retained_values;
        rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
                "retained count disagrees with the bitmap");
    } else {
        corrupted.front().buffer[0] = 0;
        corrupted.front().buffer[1] = 0;
        rejects([&] { npu::compute_tile_encoded(p, c, corrupted, tiles.front(), actual); },
                "empty palette");
    }
}

void complete_int8_palette() {
    npu::Config c;
    c.shape.m = 1;
    c.shape.n = 1;
    c.shape.k = 256;
    c.mapping.tile_m = 1;
    c.mapping.tile_n = 1;
    c.weights.encoding = npu::WeightEncoding::Palette;
    c.weights.palette_bits = 8;
    auto p = npu::make_problem(c);
    for (int value = -128; value <= 127; ++value)
        p.weights.at(static_cast<std::size_t>(value + 128)) = static_cast<std::int8_t>(value);
    const auto encoded = npu::encode_weights(c, p);
    require(encoded.front().buffer[0] == 0 && encoded.front().buffer[1] == 1,
            "256-entry LUT count in two header bytes");
    require(encoded.front().buffer.size() == 4 + 256 + 256,
            "256-value palette: 4-byte header, 256-byte LUT, 256 packed indices");
    npu::Output actual;
    npu::compute_tile_encoded(p, c, encoded, npu::make_tiles(c).front(), actual);
    require(actual.accumulators == npu::reference(p, c.quant).accumulators,
            "every signed INT8 value round-trips through the LUT");
}
}  // namespace

int main() {
    try {
        encoded_panels_only(npu::WeightEncoding::Palette);
        encoded_panels_only(npu::WeightEncoding::Sparse);
        complete_int8_palette();
        std::cout << "compression core tests: all checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "compression core tests FAILED: " << error.what() << '\n';
        return 1;
    }
}
