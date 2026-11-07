#!/usr/bin/env python3
"""Sweep buffer slots, MAC throughput and DRAM bandwidth over fixed GEMM shapes and report
cycle speedups. Every variant must reproduce the baseline's output hash.

Entry point for npuperf.sweep (also: python -m npuperf.sweep).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npuperf.sweep import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
