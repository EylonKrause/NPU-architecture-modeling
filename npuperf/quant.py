"""Integer requantization oracle, written independently of the C++ datapath.

Three rescale modes, matching npu-sim --rescale:

native       sign-magnitude rounding: round(|acc * M| / 2^shift) with ties away from zero
             (``rounding="away"``) or to even, then the sign; M <= INT32_MAX, shift 0..31.
tosa-single  TOSA 1.0.2 RESCALE apply_scale_32 with SINGLE_ROUND.
tosa-double  TOSA 1.0.2 RESCALE apply_scale_32 with DOUBLE_ROUND.

The TOSA modes are computed through the TFLite formulation, so the oracle checks the
simulator, which follows the TOSA pseudo-code, against a second published definition:
tosa-single == TFLite MultiplyByQuantizedMultiplier built with TFLITE_SINGLE_ROUNDING,
tosa-double == TFLite's default (gemmlowp SaturatingRoundingDoublingHighMul followed by
RoundingDivideByPOT), with ``tflite_shift = 31 - tosa_shift``. Every mode then adds the
output zero point, applies ReLU at the output zero point when requested and saturates to INT8.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
RESCALE_MODES = ("native", "tosa-single", "tosa-double")


def saturate(value: int, low: int = INT8_MIN, high: int = INT8_MAX) -> int:
    return max(low, min(high, value))


def _int32(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not INT32_MIN <= value <= INT32_MAX:
        raise ValueError(f"{name} must be a signed 32-bit integer")
    return value


def round_div(value: int, denominator: int, rounding: str = "away") -> int:
    """Nearest-integer ``value / denominator`` on the magnitude; ties away from zero or to even."""
    if rounding not in ("away", "even"):
        raise ValueError("rounding must be away or even")
    quotient, remainder = divmod(abs(value), denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator
                                       and (rounding == "away" or quotient % 2)):
        quotient += 1
    return -quotient if value < 0 else quotient


def rescale_native(value: int, multiplier: int, shift: int, rounding: str = "away") -> int:
    _int32(value, "accumulator")
    if not 0 <= multiplier <= INT32_MAX:
        raise ValueError("native multiplier must be in 0..INT32_MAX")
    if not 0 <= shift <= 31:
        raise ValueError("native shift must be in 0..31")
    return round_div(value * multiplier, 1 << shift, rounding)


# gemmlowp fixedpoint.h, written for Python integers with C++ int32/int64 semantics.

def _truncating_div(numerator: int, denominator: int) -> int:
    quotient = abs(numerator) // abs(denominator)
    return quotient if (numerator >= 0) == (denominator > 0) else -quotient


def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    if a == b == INT32_MIN:
        return INT32_MAX
    product = a * b
    nudge = (1 << 30) if product >= 0 else 1 - (1 << 30)
    return _truncating_div(product + nudge, 1 << 31)


def rounding_divide_by_pot(x: int, exponent: int) -> int:
    if not 0 <= exponent <= 31:
        raise ValueError("RoundingDivideByPOT exponent must be in 0..31")
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def tflite_double_rounding(x: int, quantized_multiplier: int, shift: int) -> int:
    """TFLite MultiplyByQuantizedMultiplier, default (gemmlowp) build; shift < 0 is right."""
    left, right = (shift, 0) if shift > 0 else (0, -shift)
    shifted = _int32(x * (1 << left), "left-shifted accumulator")
    return rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(shifted, quantized_multiplier), right)


def tflite_single_rounding(x: int, quantized_multiplier: int, shift: int) -> int:
    """TFLite MultiplyByQuantizedMultiplier built with TFLITE_SINGLE_ROUNDING."""
    if not -31 <= shift <= 30:
        raise ValueError("TFLite shift must be in -31..30")
    total = 31 - shift
    return (x * quantized_multiplier + (1 << (total - 1))) >> total


def rescale_tosa(value: int, multiplier: int, shift: int, double_round: bool = False) -> int:
    """TOSA apply_scale_32 semantics (including its REQUIRE checks) via the TFLite forms."""
    _int32(value, "accumulator")
    if not 0 <= multiplier <= INT32_MAX:
        raise ValueError("TOSA multiplier must be in 0..INT32_MAX")
    if not 2 <= shift <= 62:
        raise ValueError("TOSA shift must be in 2..62")
    if not -(1 << (shift - 1)) <= value < (1 << (shift - 1)):
        raise ValueError("accumulator is outside the TOSA range for this shift")
    tflite_shift = 31 - shift
    if double_round:
        return tflite_double_rounding(value, multiplier, tflite_shift)
    return tflite_single_rounding(value, multiplier, tflite_shift)


def rescale(value: int, multiplier: int, shift: int, mode: str = "native",
            rounding: str = "away") -> int:
    if mode == "native":
        return rescale_native(value, multiplier, shift, rounding)
    if mode in ("tosa-single", "tosa-double"):
        return rescale_tosa(value, multiplier, shift, mode == "tosa-double")
    raise ValueError(f"rescale mode must be one of {', '.join(RESCALE_MODES)}")


def requantize(value: int, q: Mapping[str, Any]) -> int:
    """INT32 accumulator -> INT8 output for one channel's parameters ``q``."""
    scaled = rescale(value, q["multiplier"], q["shift"], q.get("rescale", "native"),
                     q.get("rounding", "away"))
    scaled += q.get("output_zero", 0)
    if q.get("relu", False):
        scaled = max(scaled, q.get("output_zero", 0))
    return saturate(scaled)


def quantize_multiplier(scale: float) -> tuple[int, int]:
    """TFLite QuantizeMultiplier: float scale -> (Q31 multiplier, TOSA shift)."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    if scale == 0:
        return 0, 31
    fraction, exponent = math.frexp(scale)
    scaled = fraction * (1 << 31)
    fixed = int(math.floor(scaled + 0.5))  # std::round: halves away from zero, scaled > 0
    if fixed == 1 << 31:
        fixed //= 2
        exponent += 1
    if exponent < -31:
        return 0, 31
    shift = 31 - exponent
    if not 2 <= shift <= 62:
        raise ValueError(f"scale {scale!r} is outside the TOSA 32-bit rescale range")
    return fixed, shift


def gemm_reference(activations: Sequence[int], weights: Sequence[int], bias: Sequence[int],
                   m: int, n: int, k: int, *, input_zero: int = 0,
                   weight_zero: int | Sequence[int] = 0, quant: Mapping[str, Any] | None = None,
                   lanes: Sequence[Mapping[str, Any]] | None = None,
                   groups: int = 1) -> tuple[list[int], list[int]]:
    """Scalar (grouped) GEMM: activations M x G*K, weights G blocks of K x N, bias G*N.

    ``quant`` gives the layer parameters; ``lanes`` (one per output channel, G*N) override
    multiplier/shift/weight_zero per channel. Returns (INT32 accumulators, INT8 outputs),
    both row-major M x G*N.
    """
    channels = groups * n
    if len(activations) != m * groups * k or len(weights) != groups * k * n \
            or len(bias) != channels:
        raise ValueError("reference tensor sizes do not match the GEMM shape")
    zeros = list(weight_zero) if isinstance(weight_zero, Sequence) else [weight_zero] * channels
    if len(zeros) != channels:
        raise ValueError("weight zero points must have one entry per output channel")
    params = [{**(quant or {}), **(lanes[c] if lanes else {})} for c in range(channels)]
    if lanes:
        zeros = [lane.get("weight_zero", zeros[c]) for c, lane in enumerate(lanes)]
    accumulators: list[int] = []
    outputs: list[int] = []
    row_stride = groups * k
    for row in range(m):
        for channel in range(channels):
            group, col = divmod(channel, n)
            total = bias[channel]
            a_base = row * row_stride + group * k
            w_base = group * k * n + col
            for r in range(k):
                total += (activations[a_base + r] - input_zero) * (weights[w_base + r * n]
                                                                  - zeros[channel])
            _int32(total, "accumulator")
            accumulators.append(total)
            outputs.append(requantize(total, params[channel]) if quant is not None or lanes
                           else saturate(total))
    return accumulators, outputs
