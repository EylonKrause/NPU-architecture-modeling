#!/usr/bin/env python3
"""Fuse a ReLU into the preceding accelerator op when the intermediate tensor has no other
consumer and is not a graph output. ``--verify`` runs both graphs on npu-sim and compares
every output.

Entry point for npuperf.graph_optimize (also: python -m npuperf.graph_optimize).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npuperf.graph_optimize import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
