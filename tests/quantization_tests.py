#!/usr/bin/env python3
"""Integer oracles for requantization, per-channel scales, INT4 weights
and grouped GEMM.

The TOSA rescale modes are checked against TFLite MultiplyByQuantizedMultiplier
and CMSIS-NN arm_nn_requantize, each written here from its own source (TFLite
common.cc, gemmlowp fixedpoint.h, CMSIS-NN arm_nnsupportfunctions.h) independently
of TOSA apply_scale_32. A TFLite shift is a left-shift exponent; the simulator takes
the TOSA right shift, tosa_shift = 31 - tflite_shift.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

BINARY: Path
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


def c_divide(a: int, b: int) -> int:
    """C++ integer division, which truncates toward zero."""
    quotient = abs(a) // abs(b)
    return quotient if (a >= 0) == (b > 0) else -quotient


# gemmlowp fixedpoint.h @ 13d5770: the VQRDMULH computation.
def saturating_rounding_doubling_high_mul(a: int, b: int) -> int:
    overflow = a == b == INT32_MIN
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else 1 - (1 << 30)
    return INT32_MAX if overflow else c_divide(ab + nudge, 1 << 31)


# gemmlowp fixedpoint.h: rounding arithmetic right shift, ties away from zero.
def rounding_divide_by_pot(x: int, exponent: int) -> int:
    assert 0 <= exponent <= 31
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exponent) + (1 if remainder > threshold else 0)


# TFLite common.cc, default build (double rounding).
def tflite_double(x: int, quantized_multiplier: int, shift: int) -> int:
    left_shift = shift if shift > 0 else 0
    right_shift = 0 if shift > 0 else -shift
    shifted = x * (1 << left_shift)
    # C++ wraps here; the TOSA input range keeps every tested value exact.
    assert INT32_MIN <= shifted <= INT32_MAX
    return rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(shifted, quantized_multiplier), right_shift
    )


# TFLite common.cc, TFLITE_SINGLE_ROUNDING build.
def tflite_single(x: int, quantized_multiplier: int, shift: int) -> int:
    assert quantized_multiplier >= 0 and -31 <= shift <= 30
    total_shift = 31 - shift
    result = (x * quantized_multiplier + (1 << (total_shift - 1))) >> total_shift
    assert INT32_MIN <= result <= INT32_MAX
    return result


# CMSIS-NN arm_nn_requantize, CMSIS_NN_USE_SINGLE_ROUNDING branch.
def cmsis_single(val: int, multiplier: int, shift: int) -> int:
    total_shift = 31 - shift
    result = (val * multiplier) >> (total_shift - 1)
    return (result + 1) >> 1


# CMSIS-NN arm_nn_requantize, default C branch.
def cmsis_double(val: int, multiplier: int, shift: int) -> int:
    def doubling_high_mult_no_sat(m1: int, m2: int) -> int:
        return ((1 << 30) + m1 * m2) >> 31

    def divide_by_power_of_two(dividend: int, exponent: int) -> int:
        remainder_mask = (1 << exponent) - 1
        remainder = remainder_mask & dividend
        result = dividend >> exponent
        threshold = remainder_mask >> 1
        if result < 0:
            threshold += 1
        if remainder > threshold:
            result += 1
        return result

    left, right = (shift, 0) if shift > 0 else (0, -shift)
    return divide_by_power_of_two(doubling_high_mult_no_sat(val * (1 << left), multiplier), right)


def tosa_input_range(tosa_shift: int) -> tuple[int, int]:
    """The apply_scale_32 REQUIRE on the input: [-2^(s-1), 2^(s-1))."""
    return -(1 << (tosa_shift - 1)), 1 << (tosa_shift - 1)


def rescale(mode: str, value: int, multiplier: int, tosa_shift: int) -> int:
    """TFLite formulation of a simulator rescale mode, before the output zero point."""
    tflite_shift = 31 - tosa_shift
    if mode == "tosa-single":
        return tflite_single(value, multiplier, tflite_shift)
    return tflite_double(value, multiplier, tflite_shift)


def native(value: int, multiplier: int, shift: int, rounding: str) -> int:
    """Native mode: round the magnitude of value * multiplier, then restore the sign."""
    product = value * multiplier
    magnitude, remainder = divmod(abs(product), 1 << shift)
    if remainder * 2 > (1 << shift) or (
        remainder * 2 == (1 << shift) and (rounding == "away" or magnitude % 2)
    ):
        magnitude += 1
    return -magnitude if product < 0 else magnitude


def finish(scaled: int, output_zero: int, relu: bool) -> int:
    scaled += output_zero
    if relu:
        scaled = max(scaled, output_zero)
    return max(-128, min(127, scaled))


def oracle(
    a,
    w,
    bias,
    m,
    n,
    k,
    input_zero,
    zeros,
    multipliers,
    shifts,
    output_zero,
    relu,
    rounding="away",
    groups=1,
    mode="native",
):
    """Grouped GEMM: activations M x (G*K), weights G blocks of K x N, outputs M x (G*N)."""
    accumulators, outputs = [], []
    for row in range(m):
        for group in range(groups):
            for col in range(n):
                channel = group * n + col
                acc = bias[channel] + sum(
                    (a[row * groups * k + group * k + inner] - input_zero)
                    * (w[(group * k + inner) * n + col] - zeros[channel])
                    for inner in range(k)
                )
                if not INT32_MIN <= acc <= INT32_MAX:
                    raise OverflowError("final accumulator does not fit INT32")
                accumulators.append(acc)
                if mode == "native":
                    scaled = native(acc, multipliers[channel], shifts[channel], rounding)
                else:
                    scaled = rescale(mode, acc, multipliers[channel], shifts[channel])
                outputs.append(finish(scaled, output_zero, relu))
    return accumulators, outputs


def run_tensor(
    a,
    w,
    bias,
    m,
    n,
    k,
    *,
    zeros,
    multipliers,
    shifts,
    groups=1,
    bits=8,
    input_zero=0,
    output_zero=0,
    relu=False,
    rounding="away",
    rescale_mode="native",
    extra=(),
):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "tensors.npu"
        path.write_text(
            "NPU1 "
            + " ".join(map(str, (m, n, k)))
            + "\n"
            + "\n".join(" ".join(map(str, values)) for values in (a, w, bias))
            + "\n"
        )
        quant_path = Path(temporary) / "quantization.npuq"
        # The vectors are written as given, so short ones exercise the truncation check.
        quant_path.write_text(
            f"NPUQ1 {groups * n}\n"
            + "\n".join(
                " ".join(map(str, row)) for row in zip(zeros, multipliers, shifts, strict=False)
            )
            + "\n"
        )
        command = [
            str(BINARY),
            "--input",
            str(path),
            "--dump-output",
            "--weight-bits",
            str(bits),
            "--input-zero",
            str(input_zero),
            "--weight-zero",
            "0",
            "--output-zero",
            str(output_zero),
            "--quantization-file",
            str(quant_path),
            "--rounding",
            rounding,
            "--rescale",
            rescale_mode,
            "--groups",
            str(groups),
            *extra,
        ]
        if relu:
            command.append("--relu")
        return subprocess.run(command, capture_output=True, text=True, timeout=60)


def run_rescale(accumulators, multipliers, shifts, mode, *, output_zero=0, relu=False):
    """One channel per case: M = K = 1 with zero activations, so each accumulator is its bias."""
    count = len(accumulators)
    return run_tensor(
        [0],
        [0] * count,
        accumulators,
        1,
        count,
        1,
        zeros=[0] * count,
        multipliers=multipliers,
        shifts=shifts,
        output_zero=output_zero,
        relu=relu,
        rescale_mode=mode,
        extra=["--tile-n", "256"],
    )


# Golden vectors of docs/numerics.md: x, multiplier, TFLite shift, single and double rounding.
PUBLISHED_VECTORS = [
    (3, 1861152495, -1, 1, 2),
    (-6, 1073741824, -1, -1, -2),
    (-3, 1073741824, 0, -1, -1),
    (5038971, 2062684434, -2, 1209998, 1209999),
]


class QuantizationChecks(unittest.TestCase):
    def assert_rejected(self, proc, fragment=""):
        self.assertEqual(proc.returncode, 2, proc.stderr)
        lines = proc.stderr.strip().splitlines()
        self.assertTrue(lines and lines[-1].startswith("npu-sim: "), proc.stderr)
        self.assertIn(fragment, lines[-1])

    def result(self, proc):
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertIs(result["numerical_match"], True)
        self.assertEqual(result["numerics"], "bit-exact")
        return result

    def test_known_asymmetric_channels(self):
        a, w, bias = [1, 3, -2, 4], [2, -3, 7, -4, 6, -8], [5, -7, 3]
        proc = run_tensor(
            a,
            w,
            bias,
            2,
            3,
            2,
            zeros=[1, -2, 3],
            multipliers=[1, 3, 2],
            shifts=[1, 2, 0],
            input_zero=1,
            output_zero=5,
            extra=["--tile-m", "1", "--tile-n", "1"],
        )
        result = self.result(proc)
        self.assertEqual(result["accumulators"], [-5, 9, -19, -13, 20, -42])
        self.assertEqual(result["outputs"], [2, 12, -33, -2, 20, -79])

    def test_manual_nearest_even_channels(self):
        proc = run_tensor(
            [1, -1],
            [1, 3, 5, 2, 6, 10],
            [0] * 6,
            2,
            6,
            1,
            zeros=[0] * 6,
            multipliers=[1] * 6,
            shifts=[1, 1, 1, 2, 2, 2],
            rounding="even",
            output_zero=7,
            extra=["--tile-m", "1", "--tile-n", "2"],
        )
        result = self.result(proc)
        self.assertEqual(result["accumulators"], [1, 3, 5, 2, 6, 10, -1, -3, -5, -2, -6, -10])
        self.assertEqual(result["outputs"], [7, 9, 9, 7, 9, 9, 7, 5, 5, 7, 5, 5])

    def test_reject_unsupported_rounding(self):
        proc = run_tensor(
            [1], [1], [0], 1, 1, 1, zeros=[0], multipliers=[1], shifts=[1], rounding="floor"
        )
        self.assert_rejected(proc, "--rounding: expected away|even, got 'floor'")
        proc = run_tensor(
            [1], [1], [0], 1, 1, 1, zeros=[0], multipliers=[1], shifts=[2], rescale_mode="tosa"
        )
        self.assert_rejected(proc, "--rescale: expected native|tosa-single|tosa-double")

    def test_random_per_channel_gemms_match_the_oracle(self):
        rng = random.Random(361)
        for trial in range(18):
            m, n, k = rng.randint(1, 6), rng.randint(1, 7), rng.randint(1, 11)
            bits = 4 if trial % 2 else 8
            low, high = (-8, 7) if bits == 4 else (-128, 127)
            a = [rng.randint(-128, 127) for _ in range(m * k)]
            w = [rng.randint(low, high) for _ in range(k * n)]
            bias = [rng.randint(-10000, 10000) for _ in range(n)]
            zeros = [rng.randint(low, high) for _ in range(n)]
            multipliers = [rng.choice([0, 1, 3, 127, (1 << 31) - 1]) for _ in range(n)]
            shifts = [rng.choice([0, 1, 7, 16, 31]) for _ in range(n)]
            input_zero, output_zero, relu = (
                rng.randint(-128, 127),
                rng.randint(-128, 127),
                bool(trial % 3),
            )
            rounding = "even" if trial % 3 == 1 else "away"
            expected = oracle(
                a,
                w,
                bias,
                m,
                n,
                k,
                input_zero,
                zeros,
                multipliers,
                shifts,
                output_zero,
                relu,
                rounding,
            )
            with self.subTest(trial=trial, bits=bits):
                proc = run_tensor(
                    a,
                    w,
                    bias,
                    m,
                    n,
                    k,
                    zeros=zeros,
                    multipliers=multipliers,
                    shifts=shifts,
                    bits=bits,
                    input_zero=input_zero,
                    output_zero=output_zero,
                    relu=relu,
                    rounding=rounding,
                    extra=[
                        "--tile-m",
                        "2",
                        "--tile-n",
                        "3",
                        "--tile-k",
                        str(1 + trial % 4),
                        "--array-rows",
                        str(1 + trial % 3),
                        "--array-cols",
                        str(1 + trial % 4),
                    ],
                )
                result = self.result(proc)
                self.assertEqual((result["accumulators"], result["outputs"]), expected)

    def test_int4_negative_extremes_and_odd_packing(self):
        proc = run_tensor(
            [127, -128],
            [-8, 7, 0, 7, -8, -1],
            [1, -1, 0],
            1,
            3,
            2,
            zeros=[0, 0, 0],
            multipliers=[1, 1, 1],
            shifts=[0, 0, 0],
            bits=4,
            extra=["--tile-n", "1", "--tile-k", "1"],
        )
        result = self.result(proc)
        self.assertEqual(result["accumulators"], [-1911, 1912, 128])
        self.assertEqual(result["outputs"], [-128, 127, 127])

    def test_reject_bad_channels_and_int4_storage(self):
        for zeros, multipliers, shifts, weight, fragment in [
            ([0], [1, 1], [0, 0], 0, "truncated quantization parameters"),
            ([0, 0], [1], [0, 0], 0, "truncated quantization parameters"),
            ([0, 0], [1, 1], [32, 0], 0, "per-channel shifts must be in 0..31 for native"),
            ([0, 0], [1, 1], [-1, 0], 0, "per-channel shifts must be in 0..31 for native"),
            ([0, 0], [-1, 1], [0, 0], 0, "per-channel multipliers must be in 0..2147483647"),
            ([0, 8], [1, 1], [0, 0], 0, "INT4 per-channel weight zero points"),
            ([0, 0], [1, 1], [0, 0], 8, "outside configured signed precision"),
            ([0, 0], [1, 1], [0, 0], -9, "outside configured signed precision"),
        ]:
            with self.subTest(zeros=zeros, multipliers=multipliers, shifts=shifts, weight=weight):
                proc = run_tensor(
                    [1],
                    [weight, 0],
                    [0, 0],
                    1,
                    2,
                    1,
                    zeros=zeros,
                    multipliers=multipliers,
                    shifts=shifts,
                    bits=4,
                )
                self.assert_rejected(proc, fragment)

    def test_formulations_reproduce_published_vectors(self):
        for x, multiplier, shift, single, double in PUBLISHED_VECTORS:
            with self.subTest(x=x, multiplier=multiplier, shift=shift):
                self.assertEqual(tflite_single(x, multiplier, shift), single)
                self.assertEqual(cmsis_single(x, multiplier, shift), single)
                self.assertEqual(tflite_double(x, multiplier, shift), double)
                self.assertEqual(cmsis_double(x, multiplier, shift), double)
                self.assertEqual(rescale("tosa-single", x, multiplier, 31 - shift), single)
                self.assertEqual(rescale("tosa-double", x, multiplier, 31 - shift), double)

    def test_published_vectors_through_the_simulator(self):
        accumulators = [x for x, *_ in PUBLISHED_VECTORS]
        multipliers = [multiplier for _, multiplier, *_ in PUBLISHED_VECTORS]
        shifts = [31 - shift for _, _, shift, *_ in PUBLISHED_VECTORS]
        for mode, column in (("tosa-single", 3), ("tosa-double", 4)):
            with self.subTest(mode=mode):
                result = self.result(run_rescale(accumulators, multipliers, shifts, mode))
                self.assertEqual(result["accumulators"], accumulators)
                # 1209998 and 1209999 saturate to INT8 127.
                expected = [max(-128, min(127, row[column])) for row in PUBLISHED_VECTORS]
                self.assertEqual(result["outputs"], expected)
                self.assertEqual(result["config"]["quant"]["rescale"], mode)

    def test_negative_ties_separate_the_three_modes(self):
        # -6 * 4 / 2^4 and -6 * 2^30 / 2^32 are both -1.5.
        cases = ([-6, -6], [4, 1 << 30], [4, 32])
        expected = {"tosa-single": [-1, -1], "tosa-double": [-1, -2]}
        for mode, outputs in expected.items():
            with self.subTest(mode=mode):
                self.assertEqual(self.result(run_rescale(*cases, mode))["outputs"], outputs)
                self.assertEqual(
                    [rescale(mode, *case) for case in zip(*cases, strict=True)], outputs
                )
        self.assertEqual(self.result(run_rescale([-6], [4], [4], "native"))["outputs"], [-2])

    def test_tosa_modes_match_tflite_and_cmsis(self):
        rng = random.Random(20261107)

        def normalized():
            return rng.randrange(1 << 30, 1 << 31)

        def in_range_case():
            # Result magnitude near the INT8 range.
            magnitude = rng.randrange(1, 1 << rng.randint(1, 31))
            acc = magnitude if rng.random() < 0.5 else -magnitude
            multiplier = normalized()
            shift = (abs(acc) * multiplier).bit_length() - 7 + rng.randint(-1, 2)
            return acc, multiplier, max(2, min(62, shift))

        def tie_case():
            # acc * 2^30 / 2^(31 + e) = acc / 2^(e + 1) is an exact half for an
            # odd multiple of 2^e; e > 0 puts the shift past 31, where double
            # rounding differs.
            exponent = rng.randint(0, 6)
            acc = (2 * rng.randint(0, 120) + 1) << exponent
            return (acc if rng.random() < 0.5 else -acc), 1 << 30, 31 + exponent

        def extreme_case():
            acc = rng.choice(
                [
                    INT32_MIN,
                    INT32_MAX,
                    INT32_MIN + 1,
                    0,
                    1,
                    -1,
                    (1 << rng.randint(0, 30)) + rng.choice([-1, 0, 1]),
                    -(1 << rng.randint(0, 31)) + rng.choice([-1, 0, 1]),
                ]
            )
            acc = max(INT32_MIN, min(INT32_MAX, acc))
            multiplier = rng.choice([0, 1, INT32_MAX, 1 << 30, normalized()])
            return acc, multiplier, rng.randint(33, 62)

        def small_shift_case():
            # TFLite left shifts (tosa shift 2..30) need inputs inside the TOSA range.
            shift = rng.randint(2, 30)
            low, high = tosa_input_range(shift)
            acc = rng.randrange(max(low, -1000), min(high, 1000))
            return acc, rng.choice([0, 1, 3, 255, 65535, normalized()]), shift

        makers = [in_range_case, tie_case, extreme_case, small_shift_case]
        cases = [makers[index % len(makers)]() for index in range(2400)]
        for acc, multiplier, shift in cases:
            low, high = tosa_input_range(shift)
            self.assertTrue(low <= acc < high)
            # TFLite and CMSIS-NN agree in both rounding modes.
            tfl = 31 - shift
            self.assertEqual(
                tflite_single(acc, multiplier, tfl), cmsis_single(acc, multiplier, tfl)
            )
            self.assertEqual(
                tflite_double(acc, multiplier, tfl), cmsis_double(acc, multiplier, tfl)
            )
        differ = sum(
            rescale("tosa-single", *case) != rescale("tosa-double", *case) for case in cases
        )
        self.assertGreater(differ, 50, "the sample must separate single from double rounding")
        unsaturated = sum(-128 <= rescale("tosa-double", *case) <= 127 for case in cases)
        self.assertGreater(unsaturated, len(cases) // 2)
        for mode in ("tosa-single", "tosa-double"):
            for chunk, (output_zero, relu) in enumerate([(0, False), (-37, True), (100, False)]):
                part = cases[chunk::3]
                with self.subTest(mode=mode, output_zero=output_zero, relu=relu):
                    accumulators, multipliers, shifts = (
                        list(column) for column in zip(*part, strict=True)
                    )
                    result = self.result(
                        run_rescale(
                            accumulators,
                            multipliers,
                            shifts,
                            mode,
                            output_zero=output_zero,
                            relu=relu,
                        )
                    )
                    expected = [finish(rescale(mode, *case), output_zero, relu) for case in part]
                    self.assertEqual(result["accumulators"], accumulators)
                    self.assertEqual(result["outputs"], expected)

    def test_tosa_rejects_parameters_and_inputs_outside_its_ranges(self):
        for shifts, fragment in (
            ([1], "per-channel shifts must be in 2..62 for tosa-single"),
            ([63], "per-channel shifts must be in 2..62 for tosa-single"),
        ):
            with self.subTest(shifts=shifts):
                self.assert_rejected(run_rescale([0], [1], shifts, "tosa-single"), fragment)
        # Shift 2 accepts inputs in [-2, 2) only.
        edges = self.result(run_rescale([-2, 1], [1, 1], [2, 2], "tosa-single"))
        self.assertEqual(
            edges["outputs"], [rescale("tosa-single", -2, 1, 2), rescale("tosa-single", 1, 1, 2)]
        )
        for acc in (2, -3):
            with self.subTest(acc=acc):
                self.assert_rejected(
                    run_rescale([acc], [1], [2], "tosa-double"),
                    f"accumulator {acc} is outside [-2, 2), the apply_scale_32 "
                    "input range for shift 2",
                )
        self.assert_rejected(
            run_rescale([INT32_MIN], [1], [31], "tosa-single"), "outside [-1073741824, 1073741824)"
        )
        self.assert_rejected(
            run_rescale([0], [-1], [40], "tosa-single"),
            "per-channel multipliers must be in 0..2147483647",
        )
        proc = subprocess.run(
            [str(BINARY), "--rescale", "tosa-single", "--rounding", "even", "--shift", "40"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assert_rejected(proc, "rounding even applies to native rescale only")
        proc = subprocess.run(
            [str(BINARY), "--rescale", "tosa-double", "--shift", "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assert_rejected(proc, "requantization shift must be in 2..62 for tosa-double rescale")

    def test_grouped_and_depthwise_oracle(self):
        rng = random.Random(1907)
        cases = [
            dict(groups=g, m=rng.randint(1, 5), n=rng.randint(1, 4), k=rng.randint(1, 7))
            for g in (2, 3, 5)
        ]
        # Depthwise 3x3: N is the channel multiplier, K the window.
        cases += [dict(groups=16, m=6, n=1, k=9), dict(groups=4, m=3, n=2, k=9)]
        for index, case in enumerate(cases):
            groups, m, n, k = case["groups"], case["m"], case["n"], case["k"]
            channels = groups * n
            mode = ("native", "tosa-single", "tosa-double")[index % 3]
            a = [rng.randint(-128, 127) for _ in range(m * groups * k)]
            w = [rng.randint(-128, 127) for _ in range(groups * k * n)]
            bias = [rng.randint(-5000, 5000) for _ in range(channels)]
            zeros = [rng.randint(-128, 127) for _ in range(channels)]
            if mode == "native":
                multipliers = [rng.randint(0, 255) for _ in range(channels)]
                shifts = [rng.randint(8, 20) for _ in range(channels)]
            else:
                multipliers = [rng.randrange(1 << 30, 1 << 31) for _ in range(channels)]
                shifts = [rng.randint(38, 45) for _ in range(channels)]
            input_zero, output_zero = rng.randint(-128, 127), rng.randint(-20, 20)
            relu = bool(index % 2)
            expected = oracle(
                a,
                w,
                bias,
                m,
                n,
                k,
                input_zero,
                zeros,
                multipliers,
                shifts,
                output_zero,
                relu,
                groups=groups,
                mode=mode,
            )
            for tile_m, tile_n, tile_k in ((1, 1, 2), (2, 3, 0), (8, 8, 4)):
                with self.subTest(case=case, mode=mode, tile=(tile_m, tile_n, tile_k)):
                    proc = run_tensor(
                        a,
                        w,
                        bias,
                        m,
                        n,
                        k,
                        zeros=zeros,
                        multipliers=multipliers,
                        shifts=shifts,
                        groups=groups,
                        input_zero=input_zero,
                        output_zero=output_zero,
                        relu=relu,
                        rescale_mode=mode,
                        extra=[
                            "--tile-m",
                            str(tile_m),
                            "--tile-n",
                            str(tile_n),
                            "--tile-k",
                            str(tile_k),
                        ],
                    )
                    result = self.result(proc)
                    self.assertEqual((result["accumulators"], result["outputs"]), expected)
                    self.assertEqual(result["config"]["shape"]["groups"], groups)
                    self.assertEqual(result["macs"], m * channels * k)

    def test_grouped_inputs_must_cover_every_group(self):
        a, w, bias = [1, 2, 3, 4], [1, -1, 2, -2], [0, 0]
        # G = 2, N = 1, K = 2: two channels need two NPUQ1 rows.
        ok = run_tensor(
            a, w, bias, 1, 1, 2, zeros=[0, 0], multipliers=[1, 1], shifts=[0, 0], groups=2
        )
        self.assertEqual(self.result(ok)["accumulators"], [1 * 1 + 2 * -1, 3 * 2 + 4 * -2])
        with tempfile.TemporaryDirectory() as temporary:
            quant = Path(temporary) / "q.npuq"
            quant.write_text("NPUQ1 1\n0 1 0\n")
            tensors = Path(temporary) / "t.npu"
            tensors.write_text("NPU1 1 1 2\n1 2 3 4\n1 -1 2 -2\n0 0\n")
            proc = subprocess.run(
                [
                    str(BINARY),
                    "--input",
                    str(tensors),
                    "--groups",
                    "2",
                    "--quantization-file",
                    str(quant),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assert_rejected(proc, "NPUQ1 G*N")
            # One group's worth of tensors is truncated input for G = 2.
            tensors.write_text("NPU1 1 1 2\n1 2\n1 -1\n0\n")
            proc = subprocess.run(
                [str(BINARY), "--input", str(tensors), "--groups", "2"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assert_rejected(proc, "truncated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
