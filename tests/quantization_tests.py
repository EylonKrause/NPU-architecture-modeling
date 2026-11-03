#!/usr/bin/env python3
"""Independent integer oracle for per-channel scales and signed INT4 weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

BINARY: Path


def oracle(a, w, bias, m, n, k, input_zero, zeros, multipliers, shifts, output_zero, relu, rounding="away"):
    accumulators, outputs = [], []
    for row in range(m):
        for col in range(n):
            acc = bias[col] + sum((a[row * k + inner] - input_zero) *
                                 (w[inner * n + col] - zeros[col]) for inner in range(k))
            if not -(1 << 31) <= acc < (1 << 31):
                raise OverflowError("final accumulator does not fit INT32")
            accumulators.append(acc)
            product = acc * multipliers[col]
            shift = shifts[col]
            divisor = 1 << shift
            magnitude, remainder = divmod(abs(product), divisor)
            if remainder * 2 > divisor or (remainder * 2 == divisor and
                    (rounding == "away" or magnitude % 2)):
                magnitude += 1
            scaled = (-magnitude if product < 0 else magnitude) + output_zero
            if relu:
                scaled = max(scaled, output_zero)
            outputs.append(max(-128, min(127, scaled)))
    return accumulators, outputs


def run_tensor(a, w, bias, m, n, k, *, zeros, multipliers, shifts,
               bits=8, input_zero=0, output_zero=0, relu=False, rounding="away", extra=()):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "tensors.npu"
        path.write_text("NPU1 " + " ".join(map(str, (m, n, k))) + "\n" +
                        "\n".join(" ".join(map(str, values)) for values in (a, w, bias)) + "\n")
        quant_path = Path(temporary) / "quantization.npuq"
        # Keep malformed lengths malformed: the CLI must reject truncated files.
        quant_path.write_text(f"NPUQ1 {n}\n" + "\n".join(
            " ".join(map(str, row)) for row in zip(zeros, multipliers, shifts)) + "\n")
        command = [str(BINARY), "--input", str(path), "--dump-output", "--weight-bits", str(bits),
                   "--input-zero", str(input_zero), "--weight-zero", "0", "--output-zero", str(output_zero),
                   "--quantization-file", str(quant_path), "--rounding", rounding, *extra]
        if relu:
            command.append("--relu")
        return subprocess.run(command, capture_output=True, text=True, timeout=30)


class QuantizationChecks(unittest.TestCase):
    def test_known_asymmetric_channels(self):
        a, w, bias = [1, 3, -2, 4], [2, -3, 7, -4, 6, -8], [5, -7, 3]
        proc = run_tensor(a, w, bias, 2, 3, 2, zeros=[1, -2, 3], multipliers=[1, 3, 2],
                          shifts=[1, 2, 0], input_zero=1, output_zero=5,
                          extra=["--tile-m", "1", "--tile-n", "1"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["accumulators"], [-5, 9, -19, -13, 20, -42])
        self.assertEqual(result["outputs"], [2, 12, -33, -2, 20, -79])

    def test_manual_nearest_even_channels(self):
        proc = run_tensor([1, -1], [1, 3, 5, 2, 6, 10], [0] * 6, 2, 6, 1,
                          zeros=[0] * 6, multipliers=[1] * 6,
                          shifts=[1, 1, 1, 2, 2, 2], rounding="even", output_zero=7,
                          extra=["--tile-m", "1", "--tile-n", "2"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["accumulators"], [1, 3, 5, 2, 6, 10, -1, -3, -5, -2, -6, -10])
        self.assertEqual(result["outputs"], [7, 9, 9, 7, 9, 9, 7, 5, 5, 7, 5, 5])

    def test_reject_unsupported_rounding(self):
        proc = run_tensor([1], [1], [0], 1, 1, 1, zeros=[0], multipliers=[1],
                          shifts=[1], rounding="floor")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("rounding", proc.stderr)

    def test_random_independent_oracle(self):
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
            input_zero, output_zero, relu = rng.randint(-128, 127), rng.randint(-128, 127), bool(trial % 3)
            rounding = "even" if trial % 3 == 1 else "away"
            expected = oracle(a, w, bias, m, n, k, input_zero, zeros, multipliers, shifts, output_zero, relu, rounding)
            with self.subTest(trial=trial, bits=bits):
                proc = run_tensor(a, w, bias, m, n, k, zeros=zeros, multipliers=multipliers,
                                  shifts=shifts, bits=bits, input_zero=input_zero,
                                  output_zero=output_zero, relu=relu, rounding=rounding,
                                  extra=["--tile-m", "2", "--tile-n", "3", "--tile-k", str(1 + trial % 4),
                                         "--array-rows", str(1 + trial % 3),
                                         "--array-cols", str(1 + trial % 4)])
                self.assertEqual(proc.returncode, 0, proc.stderr)
                result = json.loads(proc.stdout)
                self.assertEqual((result["accumulators"], result["outputs"]), expected)

    def test_int4_negative_extremes_and_odd_packing(self):
        proc = run_tensor([127, -128], [-8, 7, 0, 7, -8, -1], [1, -1, 0], 1, 3, 2,
                          zeros=[0, 0, 0], multipliers=[1, 1, 1], shifts=[0, 0, 0], bits=4,
                          extra=["--tile-n", "1", "--tile-k", "1"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result["accumulators"], [-1911, 1912, 128])
        self.assertEqual(result["outputs"], [-128, 127, 127])

    def test_reject_bad_channels_and_int4_storage(self):
        for zeros, multipliers, shifts, weight in [
            ([0], [1, 1], [0, 0], 0),
            ([0, 0], [1], [0, 0], 0),
            ([0, 0], [1, 1], [32, 0], 0),
            ([0, 8], [1, 1], [0, 0], 0),
            ([0, 0], [1, 1], [0, 0], 8),
            ([0, 0], [1, 1], [0, 0], -9),
        ]:
            with self.subTest(zeros=zeros, multipliers=multipliers, shifts=shifts, weight=weight):
                proc = run_tensor([1], [weight, 0], [0, 0], 1, 2, 1, zeros=zeros,
                                  multipliers=multipliers, shifts=shifts, bits=4)
                self.assertNotEqual(proc.returncode, 0)
                self.assertTrue(proc.stderr.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
