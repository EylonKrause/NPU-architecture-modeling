#!/usr/bin/env python3
"""Search tile sizes and buffer slots for one GEMM under an SRAM budget; report the best
mapping and the cycles/SRAM Pareto front.

Entry point for npuperf.autotune (also: python -m npuperf.autotune).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npuperf.autotune import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
