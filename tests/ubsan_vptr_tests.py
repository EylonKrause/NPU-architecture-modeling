#!/usr/bin/env python3
"""UBSan's vptr check stays on for project code in sanitizer builds.

Only SystemC's own objects are compiled with -fno-sanitize=vptr (its process dispatch trips the
check). The probe makes a member call on an object of the wrong dynamic type and must be
stopped by UBSan with the vptr diagnostic.
"""

import argparse
import os
import subprocess
import unittest

# UBSan exits with this status instead of aborting, so the outcome is a plain exit code.
UBSAN_EXIT = 3
DIAGNOSTIC = "runtime error: member call on address"
WRONG_TYPE = "which does not point to an object of type 'Square'"

ARGS = argparse.Namespace()


class UbsanVptrTests(unittest.TestCase):
    def test_wrong_dynamic_type_member_call_is_stopped(self):
        env = dict(os.environ)
        env["UBSAN_OPTIONS"] = f"halt_on_error=1:abort_on_error=0:exitcode={UBSAN_EXIT}"
        env["ASAN_OPTIONS"] = "abort_on_error=0:detect_leaks=0"
        result = subprocess.run(
            [ARGS.binary], env=env, capture_output=True, text=True, timeout=60, check=False
        )
        self.assertEqual(result.returncode, UBSAN_EXIT, result.stdout + result.stderr)
        self.assertIn(DIAGNOSTIC, result.stderr)
        self.assertIn(WRONG_TYPE, result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    ARGS, remaining = parser.parse_known_args()
    unittest.main(argv=[__file__, *remaining])
