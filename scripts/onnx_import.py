#!/usr/bin/env python3
"""Convert a signed-INT8 QLinearMatMul/QLinearConv ONNX model to graph-runtime JSON.

Entry point for npuperf.onnx_import (also: python -m npuperf.onnx_import).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npuperf.onnx_import import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
