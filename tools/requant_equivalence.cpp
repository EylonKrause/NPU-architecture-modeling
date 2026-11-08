// Equivalence of TOSA 1.0.2 apply_scale_32 with the TFLite, gemmlowp and CMSIS-NN
// requantization bodies, and the golden vectors of docs/numerics.md sections 2.4 and 2.6.
// The library bodies are transcribed from the commits listed in docs/numerics.md section 7,
// with C casts written as static_cast and assertions as require(); apply_scale_32 follows
// the TOSA pseudo-code.
//
//   g++ -O2 -std=c++17 tools/requant_equivalence.cpp -o requant_equivalence
//   ./requant_equivalence [CASES]
//
// CASES defaults to 20,000,000. Each case draws, from std::mt19937_64 seeded with 20261107,
// a multiplier uniform in [2^30, 2^31), a TFLite shift L uniform in -31..7 (TOSA shift
// 31 - L in 24..62) and an accumulator uniform in the apply_scale_32 input range of that
// shift, which is also the documented input range of every library body. The exit status
// is 1 when a golden vector differs or a comparison expected to agree has a mismatch.
#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>

namespace {

void require(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "requant_equivalence: domain check failed: %s\n", what);
        std::exit(2);
    }
}

// TOSA 1.0.2 pseudocode/library/scale_helpers.tosac.
std::int32_t apply_scale_32(std::int32_t value, std::int32_t multiplier, int shift,
                            bool double_round) {
    require(multiplier >= 0, "apply_scale_32 multiplier");
    require(2 <= shift && shift <= 62, "apply_scale_32 shift");
    const std::int64_t half = std::int64_t{1} << (shift - 1);
    require(value >= -half && value < half, "apply_scale_32 value");
    std::int64_t round = half;
    if (double_round && shift > 31) round += value >= 0 ? (1 << 30) : -(1 << 30);
    const std::int64_t result = (static_cast<std::int64_t>(value) * multiplier + round) >> shift;
    return static_cast<std::int32_t>(result);
}

namespace gemmlowp {

// fixedpoint.h.
std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a, std::int32_t b) {
    bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
    std::int64_t a_64(a);
    std::int64_t b_64(b);
    std::int64_t ab_64 = a_64 * b_64;
    std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
    std::int32_t ab_x2_high32 = static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
    return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}

// fixedpoint.h, with the std::int32_t forms of Dup, BitAnd, Add, ShiftRight,
// MaskIfLessThan and MaskIfGreaterThan.
std::int32_t RoundingDivideByPOT(std::int32_t x, int exponent) {
    require(exponent >= 0, "RoundingDivideByPOT exponent");
    require(exponent <= 31, "RoundingDivideByPOT exponent");
    const std::int32_t mask = static_cast<std::int32_t>((1ll << exponent) - 1);
    const std::int32_t zero = 0;
    const std::int32_t one = 1;
    const std::int32_t remainder = x & mask;
    const std::int32_t threshold = (mask >> 1) + ((x < zero ? ~0 : 0) & one);
    return (x >> exponent) + ((remainder > threshold ? ~0 : 0) & one);
}

}  // namespace gemmlowp

namespace tflite {

// common.cc, TFLITE_SINGLE_ROUNDING build.
std::int32_t MultiplyByQuantizedMultiplierSingle(std::int32_t x, std::int32_t quantized_multiplier,
                                                 int shift) {
    require(quantized_multiplier >= 0, "TFLite multiplier");
    require(shift >= -31 && shift <= 30, "TFLite shift");

    const std::int64_t total_shift = 31 - shift;
    const std::int64_t round = static_cast<std::int64_t>(1) << (total_shift - 1);
    std::int64_t result = x * static_cast<std::int64_t>(quantized_multiplier) + round;
    result = result >> total_shift;

    require(result >= std::numeric_limits<std::int32_t>::min() &&
                result <= std::numeric_limits<std::int32_t>::max(),
            "TFLite result");
    return static_cast<std::int32_t>(result);
}

// common.cc, default build.
std::int32_t MultiplyByQuantizedMultiplierDouble(std::int32_t x, std::int32_t quantized_multiplier,
                                                 int shift) {
    using gemmlowp::RoundingDivideByPOT;
    using gemmlowp::SaturatingRoundingDoublingHighMul;
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    return RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier),
        right_shift);
}

}  // namespace tflite

namespace cmsis_nn {

// arm_nnsupportfunctions.h.
constexpr std::int32_t LEFT_SHIFT(std::int32_t shift) { return shift > 0 ? shift : 0; }
constexpr std::int32_t RIGHT_SHIFT(std::int32_t shift) { return shift > 0 ? 0 : -shift; }

// C branch. The arm_nn_long_long union with word.low = 1 << 30 and word.high = 0 is the
// 64-bit value 1 << 30.
std::int32_t arm_nn_doubling_high_mult_no_sat(std::int32_t m1, std::int32_t m2) {
    std::int32_t result = 0;
    std::int64_t mult = std::int64_t{1} << 30;
    mult = mult + static_cast<std::int64_t>(m1) * m2;
    result = static_cast<std::int32_t>(mult >> 31);
    return result;
}

// C branch. At exponent 31 the published (1 << exponent) - 1 overflows int; the mask is
// formed in 64 bits, which gives the 0x7fffffff that the C expression yields on
// two's-complement targets.
std::int32_t arm_nn_divide_by_power_of_two(const std::int32_t dividend,
                                           const std::int32_t exponent) {
    std::int32_t result = 0;
    const auto remainder_mask = static_cast<std::int32_t>((std::int64_t{1} << exponent) - 1);
    std::int32_t remainder = remainder_mask & dividend;

    result = dividend >> exponent;

    std::int32_t threshold = remainder_mask >> 1;
    if (result < 0) {
        threshold++;
    }
    if (remainder > threshold) {
        result++;
    }

    return result;
}

// arm_nn_requantize, CMSIS_NN_USE_SINGLE_ROUNDING branch.
std::int32_t arm_nn_requantize_single(const std::int32_t val, const std::int32_t multiplier,
                                      const std::int32_t shift) {
    const std::int64_t total_shift = 31 - shift;
    const std::int64_t new_val = val * static_cast<std::int64_t>(multiplier);

    std::int64_t result = new_val >> (total_shift - 1);
    result = (result + 1) >> 1;

    return static_cast<std::int32_t>(result);
}

// arm_nn_requantize, default C branch.
std::int32_t arm_nn_requantize_double(const std::int32_t val, const std::int32_t multiplier,
                                      const std::int32_t shift) {
    return arm_nn_divide_by_power_of_two(
        arm_nn_doubling_high_mult_no_sat(val * (1 << LEFT_SHIFT(shift)), multiplier),
        RIGHT_SHIFT(shift));
}

}  // namespace cmsis_nn

// Uniform in [low, high] from one 64-bit draw. std::uniform_int_distribution is not
// specified across standard libraries; this is, so every platform draws the same cases.
std::int64_t draw(std::mt19937_64& engine, std::int64_t low, std::int64_t high) {
    const auto span = static_cast<std::uint64_t>(high - low) + 1;
    return low + static_cast<std::int64_t>(engine() % span);
}

struct Results {
    std::int32_t tosa_single, tosa_double, tflite_single, tflite_double, cmsis_single, cmsis_double;
};

Results evaluate(std::int32_t x, std::int32_t multiplier, int tflite_shift) {
    const int tosa_shift = 31 - tflite_shift;
    return {apply_scale_32(x, multiplier, tosa_shift, false),
            apply_scale_32(x, multiplier, tosa_shift, true),
            tflite::MultiplyByQuantizedMultiplierSingle(x, multiplier, tflite_shift),
            tflite::MultiplyByQuantizedMultiplierDouble(x, multiplier, tflite_shift),
            cmsis_nn::arm_nn_requantize_single(x, multiplier, tflite_shift),
            cmsis_nn::arm_nn_requantize_double(x, multiplier, tflite_shift)};
}

struct Golden {
    std::int32_t x, multiplier;
    int tflite_shift;
    std::int32_t single, twice;
};

// docs/numerics.md section 2.6.
constexpr Golden golden[] = {
    {3, 1861152495, -1, 1, 2},
    {-6, 1073741824, -1, -1, -2},
    {-3, 1073741824, 0, -1, -1},
    {5038971, 2062684434, -2, 1209998, 1209999},
};

}  // namespace

int main(int argc, char** argv) {
    std::uint64_t cases = 20000000;
    if (argc > 1) cases = std::strtoull(argv[1], nullptr, 10);
    if (argc > 2 || cases == 0) {
        std::fprintf(stderr, "usage: requant_equivalence [CASES]\n");
        return 2;
    }

    int failures = 0;
    for (const auto& g : golden) {
        const auto r = evaluate(g.x, g.multiplier, g.tflite_shift);
        if (r.tosa_single != g.single || r.tflite_single != g.single ||
            r.cmsis_single != g.single || r.tosa_double != g.twice || r.tflite_double != g.twice ||
            r.cmsis_double != g.twice) {
            std::printf("golden vector x=%" PRId32 " multiplier=%" PRId32 " shift=%d differs\n",
                        g.x, g.multiplier, g.tflite_shift);
            ++failures;
        }
    }

    std::mt19937_64 engine(20261107);
    std::uint64_t single_tflite = 0, double_tflite = 0, single_cmsis = 0, double_cmsis = 0;
    std::uint64_t single_double = 0, right_shift_cases = 0;
    for (std::uint64_t i = 0; i < cases; ++i) {
        const auto multiplier = static_cast<std::int32_t>(draw(engine, 1ll << 30, (1ll << 31) - 1));
        const auto tflite_shift = static_cast<int>(draw(engine, -31, 7));
        const std::int64_t half = std::int64_t{1} << (30 - tflite_shift);
        const auto x = static_cast<std::int32_t>(
            draw(engine, std::max<std::int64_t>(-half, std::numeric_limits<std::int32_t>::min()),
                 std::min<std::int64_t>(half - 1, std::numeric_limits<std::int32_t>::max())));
        const auto r = evaluate(x, multiplier, tflite_shift);
        single_tflite += r.tosa_single != r.tflite_single;
        double_tflite += r.tosa_double != r.tflite_double;
        single_cmsis += r.cmsis_single != r.tflite_single;
        double_cmsis += r.cmsis_double != r.tflite_double;
        single_double += r.tflite_single != r.tflite_double;
        right_shift_cases += tflite_shift < 0;
    }
    failures += single_tflite != 0 || double_tflite != 0 || single_cmsis != 0 || double_cmsis != 0;

    std::printf("cases %" PRIu64 ", seed 20261107\n", cases);
    std::printf("TOSA SINGLE_ROUND against TFLite TFLITE_SINGLE_ROUNDING: %" PRIu64 "\n",
                single_tflite);
    std::printf("TOSA DOUBLE_ROUND against TFLite default (gemmlowp): %" PRIu64 ", %" PRIu64
                " right-shift cases\n",
                double_tflite, right_shift_cases);
    std::printf("CMSIS-NN CMSIS_NN_USE_SINGLE_ROUNDING against TFLite single rounding: %" PRIu64
                "\n",
                single_cmsis);
    std::printf("CMSIS-NN default against TFLite default: %" PRIu64 "\n", double_cmsis);
    std::printf("TFLite single rounding against TFLite default: %" PRIu64 " (%.2f%%)\n",
                single_double,
                100.0 * static_cast<double>(single_double) / static_cast<double>(cases));
    return failures ? 1 : 0;
}
