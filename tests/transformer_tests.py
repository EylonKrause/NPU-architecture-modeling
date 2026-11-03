#!/usr/bin/env python3
"""Independent scalar decoder oracle and black-box SystemC contract tests."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

BINARY: Path


def f32(value: float) -> float:
    return struct.unpack('<f', struct.pack('<f', value))[0]


def word(value: float) -> int:
    return struct.unpack('<I', struct.pack('<f', value))[0]


def unword(value: int) -> float:
    return struct.unpack('<f', struct.pack('<I', value))[0]


def stored(value: float, precision: str) -> float:
    value = f32(value)
    if precision == 'fp32':
        return value
    if precision == 'fp16':
        try:
            return struct.unpack('<e', struct.pack('<e', value))[0]
        except OverflowError:
            return math.copysign(math.inf, value)
    # Independent nearest-neighbour definition, rather than the C++ bit-add
    # implementation. Float64 can represent all binary32/bfloat16 distances.
    if not math.isfinite(value) or value == 0:
        return value
    sign = -1 if value < 0 else 1
    positive = abs(value)
    lower_word = word(positive) >> 16
    low = unword(lower_word << 16)
    high = (math.ldexp(1.0, 128) if lower_word == 0x7f7f
            else unword((lower_word + 1) << 16))
    distance_low, distance_high = positive - low, high - positive
    pick_high = distance_high < distance_low or (distance_high == distance_low and lower_word & 1)
    rounded = high if pick_high else low
    if rounded == math.ldexp(1.0, 128):
        rounded = math.inf
    return math.copysign(rounded, sign)


def add(lhs: float, rhs: float) -> float:
    return f32(lhs + rhs)


def mul(lhs: float, rhs: float) -> float:
    return f32(lhs * rhs)


def reference(*, dim=16, heads=4, kv_heads=2, ff_dim=32, tokens=6, seed=1,
              precision='fp16') -> list[list[float]]:
    """Full-sequence tensor path; all K/V computed before a causal masked read.

    This deliberately differs from the simulator's incremental cache algorithm.
    Every reduction and materialization has an explicit binary32/storage cast.
    """
    cast = lambda x: stored(x, precision)
    head_dim, kv_dim = dim // heads, dim // heads * kv_heads

    def weights(rows, cols, matrix):
        return [[cast(((row * 17 + col * 13 + seed % 101 + matrix * 7) % 31 - 15) / 64)
                 for col in range(cols)] for row in range(rows)]

    matrices = [weights(dim, dim, 1), weights(dim, kv_dim, 2), weights(dim, kv_dim, 3),
                weights(dim, dim, 4), weights(dim, ff_dim, 5), weights(dim, ff_dim, 6),
                weights(ff_dim, dim, 7)]

    def dense(vector, matrix):
        output = []
        for col in range(len(matrix[0])):
            accumulated = 0.0
            for row, element in enumerate(vector):
                accumulated = add(accumulated, mul(element, matrix[row][col]))
            output.append(cast(accumulated))
        return output

    def norm(vector, which):
        square = 0.0
        for element in vector:
            square = add(square, mul(element, element))
        inv = f32(1 / f32(math.sqrt(add(f32(square / dim), f32(1e-5)))))
        gamma = [cast(1 + ((i * 3 + seed % 101 + which * 11) % 9 - 4) / 32)
                 for i in range(dim)]
        return [cast(mul(mul(x, inv), g)) for x, g in zip(vector, gamma)]

    def rotate(vector, count, position):
        result = list(vector)
        for head in range(count):
            for pair in range(0, head_dim, 2):
                angle = position / (10000 ** (pair / head_dim))
                co, si = f32(math.cos(angle)), f32(math.sin(angle))
                at = head * head_dim + pair
                left, right = vector[at:at + 2]
                result[at] = cast(add(mul(left, co), -mul(right, si)))
                result[at + 1] = cast(add(mul(left, si), mul(right, co)))
        return result

    inputs = [[cast(((pos * 19 + col * 7 + seed % 97) % 37 - 18) / 16)
               for col in range(dim)] for pos in range(tokens)]
    normalized = [norm(vector, 1) for vector in inputs]
    queries = [rotate(dense(vector, matrices[0]), heads, pos) for pos, vector in enumerate(normalized)]
    keys = [rotate(dense(vector, matrices[1]), kv_heads, pos) for pos, vector in enumerate(normalized)]
    values = [dense(vector, matrices[2]) for vector in normalized]
    scale = f32(1 / f32(math.sqrt(head_dim)))
    outputs = []
    for position, query in enumerate(queries):
        context = []
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            scores = []
            for past in range(position + 1):  # Explicit causal mask over full tensors.
                score = 0.0
                for lane in range(head_dim):
                    score = add(score, mul(query[head * head_dim + lane], keys[past][kv_head * head_dim + lane]))
                scores.append(mul(score, scale))
            maximum = max(scores)
            exponentials = [f32(math.exp(add(score, -maximum))) for score in scores]
            denominator = 0.0
            for value in exponentials:
                denominator = add(denominator, value)
            probabilities = [f32(value / denominator) for value in exponentials]
            for lane in range(head_dim):
                accumulated = 0.0
                for past, probability in enumerate(probabilities):
                    accumulated = add(accumulated, mul(probability, values[past][kv_head * head_dim + lane]))
                context.append(cast(accumulated))
        first = [cast(add(lhs, rhs)) for lhs, rhs in zip(inputs[position], dense(context, matrices[3]))]
        normalized = norm(first, 2)
        gate, up = dense(normalized, matrices[4]), dense(normalized, matrices[5])
        activated = [cast(mul(f32(g / add(1.0, f32(math.exp(-g)))), u)) for g, u in zip(gate, up)]
        outputs.append([cast(add(lhs, rhs)) for lhs, rhs in zip(first, dense(activated, matrices[6]))])
    return outputs


def run(*arguments: str) -> dict:
    completed = subprocess.run([str(BINARY), *arguments], capture_output=True, text=True, timeout=30)
    if completed.returncode:
        raise AssertionError(f'{completed.stdout}\n{completed.stderr}')
    return json.loads(completed.stdout)


def round_words(words: list[int], precision: str) -> list[int]:
    return run('--precision', precision, '--round-f32-bits', ','.join(map(str, words)))['rounded_f32_bits']


class TransformerTests(unittest.TestCase):
    def assert_sequence(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for row, (result, oracle) in enumerate(zip(actual, expected)):
            self.assertEqual(len(result), len(oracle))
            for col, (left, right) in enumerate(zip(result, oracle)):
                self.assertTrue(math.isfinite(left))
                self.assertAlmostEqual(left, right, delta=3e-6 * max(1, abs(right)),
                                       msg=f'output[{row}][{col}]')

    def test_full_independent_float_oracle_all_precisions_and_head_groups(self):
        for precision in ('fp32', 'fp16', 'bf16'):
            for kv_heads in (1, 2, 4):
                with self.subTest(precision=precision, kv_heads=kv_heads):
                    result = run('--precision', precision, '--kv-heads', str(kv_heads), '--dump-output')
                    self.assert_sequence(result['outputs'], reference(precision=precision, kv_heads=kv_heads))
                    self.assertTrue(result['partition_match'])
                    self.assertTrue(result['causal'])
                    self.assertTrue(result['rope'])
                    self.assertTrue(result['fp32_accumulation'])

    def test_nondefault_shape_and_unsigned_seed(self):
        result = run('--dim', '24', '--heads', '3', '--kv-heads', '1', '--ff-dim', '19',
                     '--prefill', '2', '--decode', '3', '--seed', '4294967295', '--precision', 'fp32', '--dump-output')
        expected = reference(dim=24, heads=3, kv_heads=1, ff_dim=19, tokens=5,
                             seed=4294967295, precision='fp32')
        self.assert_sequence(result['outputs'], expected)

    def test_prefill_incremental_partitions_and_no_future_leakage(self):
        whole = run('--prefill', '8', '--decode', '0', '--dump-output')
        for prefill in (1, 3, 6):
            with self.subTest(prefill=prefill):
                chunked = run('--prefill', str(prefill), '--decode', str(8 - prefill), '--dump-output')
                self.assertEqual(chunked['outputs'], whole['outputs'])
                self.assertEqual(chunked['requests_completed'], 1 + 8 - prefill)
        prefix = run('--prefill', '2', '--decode', '0', '--dump-output')
        self.assertEqual(prefix['outputs'], whole['outputs'][:2])

    def test_bounded_cache_atomic_rejection_and_reset_replay(self):
        result = run('--context', '6', '--verify-lifecycle', '--dump-output')
        self.assertTrue(result['lifecycle_verified'])
        self.assertTrue(result['transaction_rollback_verified'])
        self.assertEqual(result['cache_resets'], 1)
        self.assertEqual(result['rejected_requests'], 5)
        self.assertEqual(result['rejected_work_tokens'], 1)
        self.assertGreater(result['rejected_work_cycles'], 0)
        self.assertEqual(result['tokens_completed'], result['tokens_executed'])
        self.assertEqual(result['kv_live_bytes'], result['kv_capacity_bytes'])
        self.assertGreater(result['simulation_cycles'], result['workload_cycles'])
        self.assert_sequence(result['outputs'], reference())

    def test_traffic_macs_sram_and_cycle_accounting(self):
        report = run()
        n, d, f, kv, h = 6, 16, 32, 8, 4
        width = 2
        # Q/O plus K/V and three FFN projections, then QK and probability*V.
        weights_per_token = 2 * d * d + 2 * d * kv + 3 * d * f
        macs = n * weights_per_token + 2 * d * sum(range(1, n + 1))
        self.assertEqual(report['tensor_macs'], macs)
        self.assertEqual(report['dma_read_bytes'], n * (weights_per_token + 3 * d) * width)
        self.assertEqual(report['dma_write_bytes'], n * d * width)
        self.assertEqual(report['dma_transactions'], n * 9 + 2 * (1 + 3))
        self.assertEqual(report['kv_write_bytes'], 2 * n * kv * width)
        self.assertEqual(report['kv_read_bytes'], 2 * kv * width * sum(range(1, n + 1)))
        self.assertEqual(report['kv_capacity_bytes'], 2 * 32 * kv * width)
        self.assertEqual(report['scratch_bytes'], 4 * (12 * d + 3 * f + 2 * h * 32))
        self.assertEqual(report['workload_cycles'], sum(report[field] for field in ('tensor_cycles', 'vector_cycles', 'dma_cycles')))
        self.assertEqual(report['workload_cycles'], report['prefill_cycles'] + report['decode_cycles'])
        self.assertEqual(report['simulation_cycles'], report['workload_cycles'])
        slow = run('--tensor-macs-per-cycle', '1')
        self.assertGreater(slow['workload_cycles'], report['workload_cycles'])
        self.assertEqual(slow['tensor_macs'], report['tensor_macs'])
        fp32 = run('--precision', 'fp32')
        self.assertEqual(fp32['dma_read_bytes'], 2 * report['dma_read_bytes'])
        self.assertEqual(fp32['kv_capacity_bytes'], 2 * report['kv_capacity_bytes'])

    def test_kv_head_reduction_reduces_cache_and_work(self):
        multi = run('--kv-heads', '4')
        grouped = run('--kv-heads', '1')
        self.assertEqual(multi['kv_capacity_bytes'], 4 * grouped['kv_capacity_bytes'])
        self.assertGreater(multi['tensor_macs'], grouped['tensor_macs'])
        self.assertGreater(multi['dma_read_bytes'], grouped['dma_read_bytes'])

    def test_native_systemc_waveform(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / 'transformer'
            result = run('--vcd', str(base))
            trace = base.with_suffix('.vcd').read_text()
            for signal in ('decoder_stage', 'decoder_busy', 'tokens_completed', 'kv_tokens'):
                self.assertIn(signal, trace)
            self.assertGreater(trace.count('#'), result['tokens_completed'] * 15)
            self.assertIn('$timescale', trace)

    def test_progress_records_actual_stage_and_token_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'progress.jsonl'
            result = run('--progress-jsonl', str(path))
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertGreater(len(events), result['tokens_completed'] * 15)
            cycles = [event['cycles'] for event in events]
            self.assertEqual(cycles, sorted(cycles))
            tokens = [event for event in events if event['event'] == 'token_executed']
            self.assertEqual([event['tokens_executed'] for event in tokens], list(range(1, 7)))
            self.assertEqual([event['tokens_completed'] for event in tokens], [0, 0, 0, 3, 4, 5])
            commits = [event for event in events if event['event'] == 'request_commit']
            self.assertEqual([event['tokens_completed'] for event in commits], [3, 4, 5, 6])
            self.assertEqual([event['kv_tokens'] for event in commits], [3, 4, 5, 6])
            self.assertEqual(events[-1]['cycles'], result['workload_cycles'])
            for field in ('tensor_macs', 'vector_ops', 'dma_read_bytes', 'dma_write_bytes'):
                self.assertEqual(events[-1][field], result[field])

    def test_failed_candidate_executes_without_publishing_a_committed_token(self):
        for precision in ('fp32', 'fp16', 'bf16'):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'progress.jsonl'
                result = run('--precision', precision, '--verify-lifecycle', '--progress-jsonl', str(path))
                self.assertTrue(result['transaction_rollback_verified'])
                self.assertEqual(result['rejected_work_tokens'], 1)
                events = [json.loads(line) for line in path.read_text().splitlines()]
                executed = [e for e in events if e['event'] == 'token_executed']
                self.assertEqual(len(executed), 13)
                self.assertEqual(executed[6]['tokens_executed'], 7)
                self.assertEqual(executed[6]['tokens_completed'], 6)
                self.assertEqual(executed[6]['kv_tokens'], 0)
                commits = [e for e in events if e['event'] == 'request_commit']
                self.assertEqual([e['tokens_completed'] for e in commits], [3, 4, 5, 6, 12])
        # The capacity-one boundary cannot hold a successful prefix plus a bad
        # token. It still checks finite numerical rejection without commitment.
        one = run('--context', '1', '--prefill', '1', '--decode', '0', '--verify-lifecycle')
        self.assertTrue(one['transaction_rollback_verified'])
        self.assertEqual(one['rejected_work_tokens'], 0)

    def test_aliased_artifact_destinations_fail_before_truncating_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            victim = directory / 'wave.vcd'
            sentinel = 'existing file must survive rejected output aliases\n'
            victim.write_text(sentinel)
            pairs = [(directory / 'wave', victim), (victim, victim),
                     (directory / 'wave', directory / '.' / 'wave.vcd')]
            if os.name == 'posix':
                linked = directory / 'alias.jsonl'
                linked.symlink_to(victim)
                pairs.append((directory / 'wave', linked))
                hardlink = directory / 'hardlink.jsonl'
                os.link(victim, hardlink)
                pairs.append((directory / 'wave', hardlink))
                parent = directory / 'linked-directory'
                parent.symlink_to(directory, target_is_directory=True)
                pairs.append((directory / 'wave', parent / 'wave.vcd'))
            for vcd, progress in pairs:
                with self.subTest(vcd=vcd, progress=progress):
                    result = subprocess.run([str(BINARY), '--vcd', str(vcd), '--progress-jsonl', str(progress)],
                                            capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('output paths conflict', result.stderr)
                    self.assertEqual(victim.read_text(), sentinel)

    def test_fp16_all_finite_encodings_roundtrip(self):
        words = [word(struct.unpack('<e', struct.pack('<H', half))[0])
                 for half in range(65536) if half & 0x7c00 != 0x7c00]
        for offset in range(0, len(words), 4096):
            batch = words[offset:offset + 4096]
            self.assertEqual(round_words(batch, 'fp16'), batch)

    def test_storage_rounding_ties_subnormals_overflow_and_random_binary32(self):
        rng = random.Random(927)
        words = [0, 0x80000000, 0x3f800000, 0x7f800000, 0xff800000,
                 0x00000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,
                 0x33000000, 0x33800000, 0x387fc000, 0x38800000,
                 0x477fe000, 0x477ff000, 0x3f801000, 0x3f803000,
                 0x3f808000, 0x3f818000, 0x7f7f8000, 0x7f7f7fff]
        words += [rng.randrange(1 << 32) for _ in range(1800)]
        for precision in ('fp32', 'fp16', 'bf16'):
            with self.subTest(precision=precision):
                results = round_words(words, precision)
                for source, result in zip(words, results):
                    number = unword(source)
                    if math.isnan(number):
                        self.assertTrue(math.isnan(unword(result)))
                        self.assertEqual(source >> 31, result >> 31)
                    else:
                        expected = source if precision == 'fp32' else word(stored(number, precision))
                        self.assertEqual(result, expected, (hex(source), precision))

    def test_invalid_dimensions_capacity_budget_and_cli(self):
        invalid = [('--dim', '0'), ('--dim', '129'), ('--dim', '15'), ('--heads', '0'),
                   ('--heads', '3'), ('--kv-heads', '3'), ('--ff-dim', '0'), ('--context', '5'),
                   ('--prefill', '0'), ('--decode', '-1'), ('--seed', '4294967296'),
                   ('--sram-bytes', '1'), ('--precision', 'int4'), ('--dma-bytes-per-cycle', '0'),
                   ('--tensor-macs-per-cycle', 'nan'), ('--vcd', ''), ('--round-f32-bits', '1,'),
                   ('--round-f32-bits', '4294967296'), ('--what', '1'), ('--dim',)]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                result = subprocess.run([str(BINARY), *arguments], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('npu-transformer-demo:', result.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
