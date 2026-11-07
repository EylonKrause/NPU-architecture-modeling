#!/usr/bin/env python3
"""Run an INT8 graph on the NPU model: Dense/MatMul/Conv2D/DepthwiseConv2D on the accelerator
(convolutions via im2col; depthwise as one grouped call when npu-sim has --groups),
Add/ReLU/pooling/Flatten on the host. Every accelerator result is checked against a scalar
Python reference before the next node runs.

Entry point for npuperf.graph (also: python -m npuperf.graph).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npuperf.graph import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
