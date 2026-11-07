#!/usr/bin/env python3
"""Run a two-layer INT8 MLP through npu-sim with tensors from NPU1 files and check both
layers against hand-computed outputs.

Entry point for npuperf.mlp_demo (also: python -m npuperf.mlp_demo).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npuperf.mlp_demo import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
