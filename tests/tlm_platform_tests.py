#!/usr/bin/env python3
"""Black-box numerical, persistent command, and TLM contract checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

BINARY: Path
# SystemC prints its banner on stderr before sc_main; keep stderr to our own lines.
ENV = {**os.environ, 'SYSTEMC_DISABLE_COPYRIGHT_MESSAGE': '1'}


def invoke(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=20, env=ENV)


def run(*args: str) -> dict:
    completed = invoke(*args)
    if completed.returncode:
        raise AssertionError(f"demo failed: {completed.stdout}\n{completed.stderr}")
    # SystemC may emit an informational stop/VCD message before the JSON object.
    return json.loads(completed.stdout[completed.stdout.index('{'):])


def oracle(seed: int, job: int) -> tuple[list[int], list[int]]:
    """Independent Python dense arithmetic with signed, ties-to-even rounding."""
    a = [(i * 7 + seed + job * 3) % 31 - 15 for i in range(21)]
    w = [(i * 11 + seed * 3 + job * 5) % 23 - 11 for i in range(35)]
    bias = [i * 13 + job * 9 - 20 for i in range(5)]
    acc = [bias[col] + sum((a[row * 7 + k] + 3) * (w[k * 5 + col] - 2)
                           for k in range(7)) for row in range(3) for col in range(5)]
    # division by 16 is exact in binary for these small integer accumulators.
    return acc, [max(-128, min(127, round(value / 16))) for value in acc]


def rejected_accesses(jobs: int) -> int:
    """The host's illegal accesses: eleven in its protocol checks (two MMIO
    address errors, a write to a read-only register, a byte enable, an ignore
    command and a short burst on MMIO; an out-of-range read, an ignore command,
    a byte enable, an oversized burst and a narrow streaming width on DRAM),
    plus one START written while BUSY per job."""
    return 11 + jobs


class TlmPlatformTests(unittest.TestCase):
    def test_two_jobs_full_output_and_traffic(self):
        report = run('--dump-output')
        self.assertEqual(report['model'], 'systemc-tlm2-serial-coprocessor')
        self.assertEqual((report['m'], report['n'], report['k']), (3, 5, 7))
        self.assertEqual(report['clock_period_ns'], 5)
        self.assertEqual(report['jobs_completed'], 2)
        self.assertEqual(report['interrupts_observed'], 2)
        self.assertEqual(report['dma_transactions'], 10)
        self.assertEqual(report['host_memory_transactions'], 12)
        self.assertGreater(report['memory_arbitration_wait_cycles'], 0)
        self.assertEqual(report['dma_read_bytes'], 152)
        self.assertEqual(report['dma_write_bytes'], 150)
        self.assertEqual(report['compute_cycles'], 72)
        self.assertEqual(report['debug_reads'], 2)
        self.assertEqual(report['expected_protocol_errors'], rejected_accesses(2))
        self.assertNotIn('protocol_errors', report)
        self.assertGreater(report['first_start_cycle'], report['reset_release_cycle'])
        self.assertGreater(report['simulation_cycles'], 72)
        self.assertGreater(report['mmio_transactions'], 20)
        for job in report['jobs']:
            acc, outputs = oracle(1, job['id'])
            self.assertEqual(job['accumulators'], acc)
            self.assertEqual(job['outputs'], outputs)
            self.assertGreaterEqual(job['elapsed_cycles'], 63)
        self.assertNotEqual(report['jobs'][0]['outputs'], report['jobs'][1]['outputs'])

    def test_seed_and_persistent_command_loop(self):
        report = run('--jobs', '5', '--seed', '4294967295', '--dump-output')
        self.assertEqual(report['jobs_completed'], 5)
        self.assertEqual(report['interrupts_observed'], 5)
        self.assertEqual(report['dma_transactions'], 25)
        self.assertEqual(report['expected_protocol_errors'], rejected_accesses(5))
        for job in report['jobs']:
            acc, outputs = oracle(4294967295, job['id'])
            self.assertEqual(job['accumulators'], acc)
            self.assertEqual(job['outputs'], outputs)

    def test_start_written_in_reset_release_timestep_is_not_lost(self):
        # Regression: START accepted in the delta after reset release, while the
        # engine thread still waited for its first clock edge, left BUSY set with
        # no job running; the run then hit the watchdog and exited 2.
        report = run('--start-at-reset-release', '--dump-output')
        self.assertEqual(report['first_start_cycle'], report['reset_release_cycle'])
        self.assertEqual(report['jobs_completed'], 2)
        self.assertEqual(report['interrupts_observed'], 2)
        self.assertEqual(report['dma_transactions'], 10)
        self.assertEqual(report['compute_cycles'], 72)
        self.assertEqual(report['expected_protocol_errors'], rejected_accesses(2))
        for job in report['jobs']:
            acc, outputs = oracle(1, job['id'])
            self.assertEqual(job['accumulators'], acc)
            self.assertEqual(job['outputs'], outputs)
            self.assertGreaterEqual(job['elapsed_cycles'], 63)

    def test_vcd_contains_real_clock_reset_irq_and_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / 'tlm'
            report = run('--vcd', str(base))
            self.assertEqual(report['jobs_completed'], 2)
            text = base.with_suffix('.vcd').read_text()
            for signal in ('clock', 'reset_n', 'irq', 'npu_busy', 'npu_done', 'completed_jobs'):
                self.assertIn(signal, text)
            self.assertIn('$enddefinitions', text)
            self.assertGreater(text.count('#'), 100)

    def test_help_anywhere_prints_usage(self):
        for args in (('--help',), ('--jobs', '0', '--help'), ('--wat', '--help')):
            with self.subTest(args=args):
                completed = invoke(*args)
                self.assertEqual(completed.returncode, 0)
                self.assertTrue(completed.stdout.startswith('Usage: npu-tlm-demo'))

    def test_invalid_cli_is_rejected(self):
        for args, named in ((('--jobs', '0'), '--jobs'), (('--jobs', '101'), '--jobs'),
                            (('--seed', '-1'), '--seed'), (('--jobs', 'abc'), '--jobs'),
                            (('--seed', '4294967296'), '--seed'), (('--wat', '1'), '--wat'),
                            (('--vcd', ''), '--vcd'), (('--jobs',), '--jobs')):
            with self.subTest(args=args):
                completed = invoke(*args)
                self.assertEqual(completed.returncode, 2)
                self.assertTrue(completed.stderr.startswith('npu-tlm-demo: '), completed.stderr)
                self.assertIn(named, completed.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
