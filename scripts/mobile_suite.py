#!/usr/bin/env python3
"""Run the weight-compression, transformer and multicore examples and write a summary.

Entry point for npuperf.mobile_suite (also: python -m npuperf.mobile_suite).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npuperf.mobile_suite import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
