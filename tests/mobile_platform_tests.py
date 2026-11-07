#!/usr/bin/env python3
"""Independent arithmetic and timed multicore/power contracts for SystemC."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

BINARY: Path
# SystemC prints its banner on stderr before sc_main; keep stderr to our own lines.
ENV = {**os.environ, 'SYSTEMC_DISABLE_COPYRIGHT_MESSAGE': '1'}


def invoke(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(BINARY), *arguments], capture_output=True, text=True, timeout=30, env=ENV)


def run(*arguments: str) -> dict:
    result = invoke(*arguments)
    if result.returncode:
        raise AssertionError(f"mobile simulation failed: {result.stdout}\n{result.stderr}")
    return json.loads(result.stdout)


def oracle(seed: int, job: int, model: int, m: int, n: int, k: int) -> tuple[list[int], list[int]]:
    activations = [(i * 7 + seed + job * 3) % 31 - 15 for i in range(m * k)]
    weights = [(i * 11 + seed * 3 + model * 5) % 23 - 11 for i in range(k * n)]
    bias = [i * 13 + model * 9 - 20 for i in range(n)]
    accumulators = [bias[col] + sum((activations[row * k + index] + 3) *
                                   (weights[index * n + col] - 2) for index in range(k))
                    for row in range(m) for col in range(n)]
    return accumulators, [max(-128, min(127, round(value / 16))) for value in accumulators]


class MobilePlatformTests(unittest.TestCase):
    def test_real_arithmetic_full_outputs_and_repeated_determinism(self):
        flags = ('--cores', '3', '--jobs', '7', '--models', '3', '--m', '3',
                 '--n', '7', '--k', '13', '--seed', '4294967295', '--dump-output')
        report = run(*flags)
        self.assertEqual(report, run(*flags))
        self.assertEqual(report['model'], 'systemc-mobile-multicore')
        self.assertEqual(report['jobs_completed'], 7)
        self.assertTrue(report['numerical_match'])
        self.assertTrue(report['synthetic_architecture'])
        self.assertFalse(report['preemptive'])
        self.assertIsNone(report['energy'])
        for job in report['jobs']:
            expected, quantized = oracle(4294967295, job['id'], job['model'], 3, 7, 13)
            self.assertEqual(job['accumulators'], expected)
            self.assertEqual(job['outputs'], quantized)
            self.assertIsInstance(job['output_hash'], str)
            self.assertTrue(job['numerical_match'])
        tiny = run('--jobs', '1', '--m', '1', '--n', '1', '--k', '1', '--seed', '0', '--dump-output')
        self.assertEqual((tiny['jobs'][0]['accumulators'], tiny['jobs'][0]['outputs']), oracle(0, 0, 0, 1, 1, 1))

    def test_bus_serialization_exact_durations_and_real_byte_counts(self):
        report = run('--cores', '4', '--jobs', '8', '--models', '2', '--bus-bytes-per-ns', '7', '--bus-latency-ns', '13')
        transactions = report['bus_transactions']
        self.assertEqual(len(transactions), 16)
        self.assertGreater(report['bus_wait_ns'], 0)
        previous_end = 0
        for transaction in transactions:
            self.assertGreaterEqual(transaction['begin_ns'], previous_end)
            self.assertGreaterEqual(transaction['begin_ns'], transaction['requested_ns'])
            duration = 13 + math.ceil(transaction['bytes'] / 7)
            self.assertAlmostEqual(transaction['end_ns'] - transaction['begin_ns'], duration, places=6)
            previous_end = transaction['end_ns']
        self.assertEqual(report['bus_read_bytes'], 8 * 64 + report['cache_misses'] * 160)
        self.assertEqual(report['bus_write_bytes'], 8 * 160)
        for job in report['jobs']:
            transfers = [tx for tx in transactions if tx['job'] == job['id']]
            self.assertEqual(len(transfers), 2)
            read, write = transfers
            self.assertFalse(read['write'])
            self.assertTrue(write['write'])
            self.assertEqual(read['core'], job['core'])
            self.assertEqual(write['core'], job['core'])
            self.assertAlmostEqual(read['end_ns'], job['compute_begin_ns'])
            self.assertAlmostEqual(write['requested_ns'], job['compute_end_ns'])
            self.assertAlmostEqual(write['end_ns'], job['finish_ns'])

    def check_round_robin(self, report: dict, cores: int) -> int:
        """Replay the arbiter: a grant at time T goes to the first core at or after
        the round-robin pointer among requests issued strictly before T. With
        none pending the bus was idle and the grant is the request just issued.
        Returns the number of contended grants checked."""
        transactions = sorted(report['bus_transactions'], key=lambda tx: tx['begin_ns'])
        pointer, contended, previous_end = 0, 0, 0.0
        for index, grant in enumerate(transactions):
            self.assertGreaterEqual(grant['begin_ns'], previous_end)
            previous_end = grant['end_ns']
            waiting = {tx['core'] for tx in transactions[index:]
                       if tx['requested_ns'] < grant['begin_ns']}
            if waiting:
                expected = min(waiting, key=lambda core: (core - pointer) % cores)
                self.assertEqual(grant['core'], expected, f'grant {index} at {grant["begin_ns"]} ns')
                contended += len(waiting) > 1
            else:
                self.assertEqual(grant['requested_ns'], grant['begin_ns'])
            pointer = (grant['core'] + 1) % cores
        return contended

    def test_round_robin_bus_arbitration_by_core_id(self):
        # All four cores leave power gating in the same delta and request DRAM
        # at the same timestamp: grants follow core id from the reset pointer.
        same = run('--cores', '4', '--jobs', '4', '--priorities', '0')
        reads = [tx for tx in same['bus_transactions'] if not tx['write']]
        self.assertEqual({tx['requested_ns'] for tx in reads}, {12})
        self.assertEqual([tx['core'] for tx in reads], [0, 1, 2, 3])
        self.check_round_robin(same, 4)
        for flags, cores in ((('--cores', '8', '--jobs', '32', '--models', '4',
                               '--bus-bytes-per-ns', '1', '--bus-latency-ns', '50'), 8),
                             (('--cores', '3', '--jobs', '24', '--models', '5', '--arrival-ns', '200',
                               '--batch-size', '2', '--bus-bytes-per-ns', '2', '--priorities', '0,2,1'), 3)):
            with self.subTest(flags=flags):
                report = run(*flags)
                self.assertEqual(report, run(*flags))
                self.assertGreater(self.check_round_robin(report, cores), 10)

    def test_concurrent_systemc_workers_reduce_compute_bound_makespan(self):
        flags = ('--jobs', '8', '--m', '8', '--n', '8', '--k', '64',
                 '--macs-per-cycle', '1', '--priorities', '0', '--cache-bytes', '0',
                 '--bus-bytes-per-ns', '4096', '--bus-latency-ns', '0')
        serial = run('--cores', '1', *flags)
        parallel = run('--cores', '4', *flags)
        self.assertLess(parallel['simulation_ns'], serial['simulation_ns'] / 3.9)
        self.assertEqual({job['core'] for job in parallel['jobs']}, {0, 1, 2, 3})
        intervals = [(job['compute_begin_ns'], job['compute_end_ns']) for job in parallel['jobs']]
        self.assertTrue(any(max(a, c) < min(b, d) for i, (a, b) in enumerate(intervals)
                            for c, d in intervals[i + 1:]))

    def test_priority_queue_is_nonpreemptive_and_respects_arrivals(self):
        report = run('--cores', '1', '--jobs', '4', '--priorities', '0,3,1,3')
        self.assertEqual(report['dispatch_order'], [1, 3, 2, 0])
        released = run('--cores', '1', '--jobs', '4', '--priorities', '0,3,1,3',
                       '--batch-size', '1', '--arrival-ns', '1000')
        self.assertEqual(released['dispatch_order'], [0, 1, 2, 3])
        for job in released['jobs']:
            self.assertGreaterEqual(job['dispatch_ns'], job['release_ns'])

    def test_finite_lru_cache_changes_traffic_without_changing_results(self):
        common = ('--cores', '1', '--jobs', '4', '--models', '2', '--priorities', '0', '--dump-output')
        small = run(*common, '--cache-bytes', '160')
        large = run(*common, '--cache-bytes', '320')
        self.assertEqual((small['cache_misses'], small['cache_hits']), (4, 0))
        self.assertEqual((large['cache_misses'], large['cache_hits']), (2, 2))
        self.assertEqual(small['bus_read_bytes'] - large['bus_read_bytes'], 320)
        self.assertEqual(small['core_stats'][0]['peak_cache_bytes'], 160)
        self.assertEqual(large['core_stats'][0]['peak_cache_bytes'], 320)
        self.assertEqual([j['outputs'] for j in small['jobs']], [j['outputs'] for j in large['jobs']])
        disabled = run('--cores', '1', '--jobs', '4', '--cache-bytes', '0', '--priorities', '0')
        self.assertEqual((disabled['cache_misses'], disabled['cache_hits']), (4, 0))

    def test_opp_frequency_changes_only_compute_period_and_voltage_energy(self):
        common = ('--cores', '1', '--jobs', '1', '--opps', '100:400,200:800', '--mac-pj', '2')
        low = run(*common, '--opp', '0')
        high = run(*common, '--opp', '1')
        lower, higher = low['jobs'][0], high['jobs'][0]
        self.assertEqual(lower['compute_cycles'], 36)
        self.assertEqual(higher['compute_cycles'], 36)
        self.assertAlmostEqual(lower['compute_end_ns'] - lower['compute_begin_ns'], 360)
        self.assertAlmostEqual(higher['compute_end_ns'] - higher['compute_begin_ns'], 180)
        self.assertAlmostEqual(low['simulation_ns'] - high['simulation_ns'], 180)
        self.assertAlmostEqual(low['energy']['compute_pj'], 256)
        self.assertAlmostEqual(high['energy']['compute_pj'], 1024)
        self.assertEqual(lower['output_hash'], higher['output_hash'])
        fractional = run('--jobs', '1', '--cores', '1', '--opps', '333:700', '--opp', '0')
        job = fractional['jobs'][0]
        self.assertAlmostEqual(job['compute_end_ns'] - job['compute_begin_ns'], 36000 / 333, delta=0.000501)

    def test_deadline_governor_and_truthful_missed_deadlines(self):
        common = ('--cores', '1', '--jobs', '1', '--governor', 'deadline')
        relaxed = run(*common, '--deadline-ns', '1000')
        urgent = run(*common, '--deadline-ns', '100')
        self.assertEqual(relaxed['jobs'][0]['opp'], 0)
        self.assertEqual(urgent['jobs'][0]['opp'], 2)
        self.assertEqual(relaxed['opp_transitions'], 1)
        self.assertEqual(urgent['opp_transitions'], 1)
        self.assertEqual(relaxed['deadline_misses'], 0)
        self.assertEqual(urgent['deadline_misses'], 1)
        self.assertGreater(urgent['jobs'][0]['finish_ns'], 100)

    def test_idle_gating_wakeup_and_energy_units(self):
        report = run('--cores', '1', '--jobs', '3', '--batch-size', '1', '--arrival-ns', '1000',
                     '--mac-pj', '2', '--dram-byte-pj', '3', '--leakage-mw', '5',
                     '--retention-mw', '7', '--gated-mw', '11', '--wakeup-pj', '13', '--switch-pj', '17')
        self.assertEqual(report['wakeup_count'], 3)
        self.assertEqual(report['opp_transitions'], 0)
        stats = report['core_stats'][0]
        self.assertAlmostEqual(stats['retention_ns'], 200)
        self.assertGreater(stats['gated_ns'], 1000)
        self.assertAlmostEqual(stats['active_ns'] + stats['retention_ns'] + stats['gated_ns'], report['simulation_ns'])
        energy = report['energy']
        self.assertFalse(energy['silicon_calibrated'])
        self.assertAlmostEqual(energy['compute_pj'], 3 * 512 * 2)
        self.assertAlmostEqual(energy['dram_pj'], (report['bus_read_bytes'] + report['bus_write_bytes']) * 3)
        self.assertAlmostEqual(energy['active_leakage_pj'], stats['active_ns'] * 5)
        self.assertAlmostEqual(energy['retention_pj'], stats['retention_ns'] * 7)
        self.assertAlmostEqual(energy['gated_pj'], stats['gated_ns'] * 11)
        self.assertAlmostEqual(energy['transition_pj'], 3 * 13)
        self.assertAlmostEqual(energy['total_pj'], sum(energy[key] for key in (
            'compute_pj', 'dram_pj', 'active_leakage_pj', 'retention_pj', 'gated_pj', 'transition_pj')))
        short_idle = run('--cores', '1', '--jobs', '3', '--batch-size', '1', '--arrival-ns', '300')
        self.assertEqual(short_idle['wakeup_count'], 1)
        switched = run('--jobs', '1', '--cores', '1', '--governor', 'deadline',
                       '--deadline-ns', '1000', '--switch-pj', '17')
        self.assertEqual(switched['opp_transitions'], 1)
        self.assertAlmostEqual(switched['energy']['total_pj'], 17)

    def test_unused_core_idle_accounting_covers_common_makespan(self):
        report = run('--jobs', '1', '--cores', '8', '--gated-mw', '2')
        self.assertEqual(sum(core['jobs'] for core in report['core_stats']), 1)
        for core in report['core_stats']:
            self.assertAlmostEqual(core['active_ns'] + core['retention_ns'] + core['gated_ns'],
                                   report['simulation_ns'])
            if core['jobs'] == 0:
                self.assertAlmostEqual(core['gated_ns'], report['simulation_ns'])
        expected_gated = sum(core['gated_ns'] for core in report['core_stats'])
        self.assertAlmostEqual(report['energy']['total_pj'], 2 * expected_gated)

    def test_vcd_contains_kernel_resource_and_dvfs_signals(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / 'mobile'
            report = run('--governor', 'deadline', '--deadline-ns', '1000', '--vcd', str(base))
            self.assertEqual(report['jobs_completed'], 6)
            trace = base.with_suffix('.vcd').read_text()
            self.assertIn('$enddefinitions', trace)
            for name in ('dram_busy', 'dram_owner', 'completed_jobs', 'ready_jobs',
                         'core0_state', 'core0_job', 'core1_frequency_mhz', 'core1_voltage_mv'):
                self.assertIn(name, trace)
            self.assertGreater(trace.count('#'), 25)

    def test_help_anywhere_prints_usage(self):
        for arguments in (('--help',), ('--cores', '2', '--help'), ('--cores', '0', '--help')):
            with self.subTest(arguments=arguments):
                completed = invoke(*arguments)
                self.assertEqual(completed.returncode, 0)
                self.assertTrue(completed.stdout.startswith('Usage: npu-mobile-demo'))

    def test_validation_bounds_and_invalid_inputs(self):
        for arguments in (
            ('--cores', '0'), ('--cores', '9'), ('--jobs', '129'), ('--models', '0'),
            ('--m', '65'), ('--k', '257'), ('--batch-size', '0'), ('--seed', '4294967296'),
            ('--bus-bytes-per-ns', '0'), ('--macs-per-cycle', '0'), ('--cache-bytes', '65537'),
            ('--sram-bytes', '1'), ('--opp', '3'), ('--opps', '200:800,100:700'),
            ('--opps', '100:0'), ('--opps', '0:700'), ('--opps', '100:700,'),
            ('--opps', '100:800,200:700'),
            ('--opps', '100:700:3'), ('--priorities', '256'), ('--priorities', '1,,2'),
            ('--governor', 'magic'), ('--mac-pj', 'nan'), ('--dram-byte-pj', '-1'),
            ('--leakage-mw', 'inf'), ('--mac-pj', 'abc'), ('--mac-pj', '1e999'),
            ('--wakeup-ns', '1000001'), ('--jobs',), ('--cores', 'two'), ('--unknown', '3'),
        ):
            with self.subTest(arguments=arguments):
                completed = invoke(*arguments)
                self.assertEqual(completed.returncode, 2)
                self.assertTrue(completed.stderr.startswith('npu-mobile-demo: '), completed.stderr)
                self.assertIn(arguments[0], completed.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parsed, remaining = parser.parse_known_args()
    BINARY = parsed.binary.resolve()
    unittest.main(argv=[__file__, *remaining])
