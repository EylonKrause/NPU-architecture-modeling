#!/usr/bin/env python3
"""Estimate MAC, DRAM and static energy for one npu-sim result.

Entry point for npuperf.energy (also: python -m npuperf.energy).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npuperf.energy import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
