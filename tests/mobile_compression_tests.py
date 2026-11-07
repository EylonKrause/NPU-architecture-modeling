#!/usr/bin/env python3
"""Independent compression, zero skipping, traffic and SystemC timing checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

import quantization_tests as quant

BINARY: Path


def panels(weights, n, k, tile_n, tile_k):
    for col in range(0, n, tile_n):
        cols = min(tile_n, n - col)
        for start in range(0, k, tile_k or k):
            count = min(tile_k or k, k - start)
            yield col, cols, start, [weights[inner * n + channel]
                for inner in range(start, start + count)
                for channel in range(col, col + cols)]


def encoded_sizes(weights, zeros, n, k, tile_n, tile_k, codec, bits=8, palette_bits=2,
                  sparse_block=4, decode_rate=16):
    records = []
    for col, cols, start, values in panels(weights, n, k, tile_n, tile_k):
        count = len(values)
        nonzero = sum(value != zeros[col + index % cols] for index, value in enumerate(values))
        if codec == "palette":
            size = 4 + len(set(values)) + (count * palette_bits + 7) // 8
            retained = count
            decode = (retained + decode_rate - 1) // decode_rate
        else:
            groups = [values[index:index + sparse_block] for index in range(0, count, sparse_block)]
            retained = 0
            for group_id, group in enumerate(groups):
                if any(value != zeros[col + (group_id * sparse_block + index) % cols]
                       for index, value in enumerate(group)):
                    retained += len(group)
            size = 4 + (len(groups) + 7) // 8 + (retained * bits + 7) // 8
            decode = (len(groups) + decode_rate - 1) // decode_rate + \
                     (retained + decode_rate - 1) // decode_rate
        records.append((size, nonzero, retained, decode, count))
    return records


class CompressionChecks(unittest.TestCase):
    def run_case(self, a, w, bias, m, n, k, *, codec, bits=8, palette_bits=2,
                 sparse_block=4, tile_m=2, tile_n=3, tile_k=3, zeros=None,
                 decode_rate=16, extra=(), rounding="away"):
        zeros = [0] * n if zeros is None else zeros
        kwargs = dict(zeros=zeros, multipliers=[1] * n, shifts=[3] * n,
                      bits=bits, input_zero=-3, output_zero=5, rounding=rounding)
        flags = ["--tile-m", str(tile_m), "--tile-n", str(tile_n), "--tile-k", str(tile_k),
                 "--weight-encoding", codec, "--palette-bits", str(palette_bits),
                 "--sparse-block", str(sparse_block), "--weight-decode-values-per-cycle", str(decode_rate),
                 *extra]
        proc = quant.run_tensor(a, w, bias, m, n, k, **kwargs, extra=flags)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        accumulators, outputs = quant.oracle(a, w, bias, m, n, k, -3, zeros,
                                            [1] * n, [3] * n, 5, False, rounding)
        self.assertEqual(result["accumulators"], accumulators)
        self.assertEqual(result["outputs"], outputs)
        self.assertTrue(result["numerical_match"])
        self.assertEqual(result["macs"], m * n * k)
        self.assertEqual(result["executed_macs"] + result["skipped_macs"], m * n * k)
        return result

    def test_every_palette_width_random_independent_oracle(self):
        rng = random.Random(122)
        for width in (1, 2, 4, 8):
            for source_bits in (4, 8):
                for trial in range(3):
                    m, n, k = 3 + trial, 7, 8 + trial
                    alphabet = rng.sample(range(-8, 8) if source_bits == 4 else range(-128, 128),
                                          min(1 << width, 16 if source_bits == 4 else 31))
                    a = [rng.randrange(-128, 128) for _ in range(m * k)]
                    w = [rng.choice(alphabet) for _ in range(n * k)]
                    bias = [rng.randrange(-99, 99) for _ in range(n)]
                    with self.subTest(width=width, bits=source_bits, trial=trial):
                        result = self.run_case(a, w, bias, m, n, k, codec="palette", bits=source_bits,
                                               palette_bits=width, tile_n=3, tile_k=4, rounding="even")
                        records = encoded_sizes(w, [0] * n, n, k, 3, 4, "palette", bits=source_bits,
                                                palette_bits=width)
                        self.assertEqual(result["encoded_weight_storage_bytes"], sum(x[0] for x in records))
                        self.assertEqual(result["skipped_macs"], 0)
                        self.assertEqual(result["weight_decode_cycles"], ((m + 1) // 2) * sum(x[3] for x in records))
                        self.assertEqual(result["decoded_weight_values"], ((m + 1) // 2) * n * k)

    def test_sparse_asymmetric_zero_points_and_odd_int4_panels(self):
        rng = random.Random(421)
        for bits in (4, 8):
            for block in (1, 3, 4, 17):
                m, n, k = 5, 7, 11
                zeros = [-3, 1, 4, -2, 0, 5, -1]
                a = [rng.randint(-128, 127) for _ in range(m * k)]
                w = [zeros[index % n] if index % 7 != 3 else rng.randint(-8, 7)
                     for index in range(k * n)]
                bias = [1, -2, 3, -4, 5, -6, 7]
                with self.subTest(bits=bits, block=block):
                    result = self.run_case(a, w, bias, m, n, k, codec="sparse", bits=bits,
                        sparse_block=block, tile_n=3, tile_k=5, zeros=zeros)
                    records = encoded_sizes(w, zeros, n, k, 3, 5, "sparse", bits=bits, sparse_block=block)
                    self.assertEqual(result["encoded_weight_storage_bytes"], sum(x[0] for x in records))
                    self.assertEqual(result["executed_macs"], m * sum(x[1] for x in records))
                    self.assertEqual(result["weight_decode_cycles"], 3 * sum(x[3] for x in records))
                    self.assertEqual(result["decoded_weight_values"], 3 * sum(x[2] for x in records))

    def test_exact_serial_timing_and_metadata_charged(self):
        for codec in ("palette", "sparse"):
            m, n, k = 1, 4, 8
            a, w, bias = [1] * k, [0] * 16 + [1] * 16, [0] * n
            result = self.run_case(a, w, bias, m, n, k, codec=codec, palette_bits=1,
                sparse_block=8, tile_m=1, tile_n=n, tile_k=0, decode_rate=5,
                extra=["--memory-bytes-per-cycle", "1", "--memory-latency", "0",
                       "--pipeline-cycles", "0", "--macs-per-cycle", "8"])
            record, = encoded_sizes(w, [0] * n, n, k, n, 0, codec, palette_bits=1,
                                     sparse_block=8, decode_rate=5)
            size, nonzero, retained, decode, _ = record
            # Input quantization-file carries three INT32 vectors plus bias.
            read = k + size + 16 * n
            # The decoder overlaps the MACs: the panel takes the longer of the two.
            compute = max(((n * k if codec == "palette" else nonzero) + 7) // 8, decode)
            self.assertEqual(result["read_bytes"], read)
            self.assertEqual(result["weight_read_bytes"], size)
            self.assertEqual(result["cycles"], read + compute + n)
            self.assertEqual(result["compute_busy_cycles"], compute)

    def test_decoder_overlaps_mac_work(self):
        # Every panel takes max(MAC cycles, decode cycles) + pipeline. A slow
        # decoder bounds the panel; a fast one costs nothing over dense.
        rng = random.Random(77)
        m, n, k, tile_m, tile_n, tile_k = 5, 7, 11, 2, 3, 4
        a = [rng.randint(-128, 127) for _ in range(m * k)]
        w = [rng.choice([0, 0, 0, 1, -1, 2]) for _ in range(k * n)]
        bias = [rng.randint(-50, 50) for _ in range(n)]
        flags = ["--pipeline-cycles", "3", "--macs-per-cycle", "8"]
        dense = self.run_case(a, w, bias, m, n, k, codec="dense", tile_m=tile_m, tile_n=tile_n,
                              tile_k=tile_k, extra=flags)
        for codec in ("palette", "sparse"):
            for rate in (1, 3, 1000):
                with self.subTest(codec=codec, rate=rate):
                    result = self.run_case(a, w, bias, m, n, k, codec=codec, tile_m=tile_m,
                                           tile_n=tile_n, tile_k=tile_k, decode_rate=rate,
                                           extra=flags)
                    records = encoded_sizes(w, [0] * n, n, k, tile_n, tile_k, codec,
                                            decode_rate=rate)
                    overlapped = summed = 0
                    for row in range(0, m, tile_m):
                        rows = min(tile_m, m - row)
                        for _, nonzero, _, decode, count in records:
                            macs = (rows * (count if codec == "palette" else nonzero) + 7) // 8
                            overlapped += max(macs, decode) + 3
                            summed += macs + decode + 3
                    self.assertEqual(result["compute_busy_cycles"], overlapped)
                    self.assertLess(result["compute_busy_cycles"], summed)
                    if codec == "palette" and rate == 1000:
                        self.assertEqual(result["compute_busy_cycles"],
                                         dense["compute_busy_cycles"])

    def test_all_zero_centered_sparse_keeps_bias_without_macs(self):
        m, n, k = 3, 3, 7
        zeros = [3, -2, 5]
        result = self.run_case([7] * (m * k), zeros * k, [10, -15, 21], m, n, k,
            codec="sparse", zeros=zeros, tile_n=n, tile_k=0, tile_m=m, sparse_block=4,
            extra=["--pipeline-cycles", "0"])
        self.assertEqual(result["executed_macs"], 0)
        self.assertEqual(result["skipped_macs"], m * n * k)
        self.assertEqual(result["decoded_weight_values"], 0)
        self.assertEqual(result["encoded_weight_storage_bytes"], 5)
        self.assertEqual(result["compute_busy_cycles"], 1)

    def test_cache_stores_encoded_bytes_and_repeats_decoder_work(self):
        m, n, k = 5, 4, 8
        a, w, bias = [1] * (m * k), [0, 1] * (n * k // 2), [0] * n
        base = self.run_case(a, w, bias, m, n, k, codec="palette", palette_bits=1,
                             tile_m=1, tile_n=n, tile_k=0)
        size = 4 + 2 + (n * k + 7) // 8
        cached = self.run_case(a, w, bias, m, n, k, codec="palette", palette_bits=1,
            tile_m=1, tile_n=n, tile_k=0, extra=["--weight-cache-bytes", str(size)])
        self.assertEqual(cached["weight_cache_hits"], m - 1)
        self.assertEqual(cached["weight_cache_misses"], 1)
        self.assertEqual(cached["weight_read_bytes"], size)
        self.assertEqual(cached["peak_cached_weight_bytes"], size)
        self.assertEqual(base["read_bytes"] - cached["read_bytes"], (m - 1) * size)
        self.assertEqual(base["weight_decode_cycles"], cached["weight_decode_cycles"])
        self.assertEqual(cached["weight_bytes_saved"], (m - 1) * size)

    def test_event_cycle_progress_live_equivalence(self):
        a, w, bias, m, n, k = [1, 2, -3] * 3, [0, 1, 0, 0, 2] * 3, [2] * 5, 3, 5, 3
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.npu"
            path.write_text(f"NPU1 {m} {n} {k}\n" + "\n".join(
                " ".join(map(str, v)) for v in (a, w, bias)) + "\n")
            for codec in ("palette", "sparse"):
                flags = [str(BINARY), "--input", str(path), "--input-zero", "0", "--weight-zero", "0",
                         "--weight-encoding", codec, "--palette-bits", "2", "--sparse-block", "2",
                         "--tile-k", "2", "--tile-n", "3", "--tile-m", "1"]
                event = json.loads(subprocess.check_output(flags, text=True))
                cycle = json.loads(subprocess.check_output([*flags, "--timing", "cycle"], text=True))
                for key in ("cycles", "output_hash", "read_bytes", "compute_busy_cycles", "executed_macs",
                            "skipped_macs", "weight_decode_cycles", "weight_read_bytes"):
                    self.assertEqual(event[key], cycle[key])
                progress = Path(temporary) / "progress.jsonl"
                sliced = json.loads(subprocess.check_output([*flags, "--progress-jsonl", str(progress),
                                                            "--slice-cycles", "7"], text=True))
                self.assertEqual(sliced["cycles"], event["cycles"])
                self.assertEqual(json.loads(progress.read_text().splitlines()[-1])["output_hash"],
                                 str(event["output_hash"]))
                live = subprocess.Popen([*flags, "--live-stdio"], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    snapshot = json.loads(live.stdout.readline())
                    while not snapshot["completed"]:
                        live.stdin.write("advance 3\n"); live.stdin.flush()
                        snapshot = json.loads(live.stdout.readline())
                    self.assertEqual(snapshot["cycles"], event["cycles"])
                    self.assertEqual(snapshot["output_hash"], str(event["output_hash"]))
                    self.assertEqual(snapshot["skipped_macs"], event["skipped_macs"])
                    live.stdin.write("stop\n"); live.stdin.flush()
                    live.communicate(timeout=10)
                    self.assertEqual(live.returncode, 0)
                finally:
                    if live.poll() is None:
                        live.kill(); live.communicate()

    def test_compression_can_expand_and_must_not_fake_speedups(self):
        # Eight distinct INT8 weights have no redundancy. LUT overhead expands
        # palette storage, and decoder cycles are explicitly visible.
        result = self.run_case([1] * 8, list(range(8)), [0], 1, 1, 8,
                               codec="palette", palette_bits=8, tile_n=1, tile_k=0)
        self.assertGreater(result["encoded_weight_storage_bytes"], result["dense_weight_storage_bytes"])
        self.assertGreater(result["weight_decode_cycles"], 0)

    def test_invalid_flags_and_lossless_capacity_rejected(self):
        for flags, fragment in [(["--weight-encoding", "zip"], "encoding"),
                (["--palette-bits", "3"], "palette"), (["--sparse-block", "0"], "sparse"),
                (["--sparse-block", "4097"], "sparse"),
                (["--weight-decode-values-per-cycle", "0"], "decode"),
                (["--weight-encoding", "sparse", "--array-rows", "2", "--array-cols", "2"], "systolic"),
                (["--weight-encoding", "palette", "--palette-bits", "1"], "capacity")]:
            with self.subTest(flags=flags):
                proc = subprocess.run([str(BINARY), *flags], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(fragment, proc.stderr)

    def test_palette_systolic_timing_preserves_geometry(self):
        result = self.run_case([1] * 10, [0, 1] * 10, [0] * 4, 2, 4, 5,
            codec="palette", palette_bits=1, tile_m=2, tile_n=4, tile_k=0,
            extra=["--array-rows", "2", "--array-cols", "2", "--pipeline-cycles", "3"])
        # Two 2x2 output waves of K = 5 back to back and skew 2 + 2 - 2 once
        # hide the LUT decoder's ceil(20 / 16) cycles; then pipeline 3.
        self.assertEqual(result["compute_busy_cycles"], max(2 * 5 + 2, 2) + 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    quant.BINARY = BINARY
    unittest.main(argv=[__file__, *remaining])
