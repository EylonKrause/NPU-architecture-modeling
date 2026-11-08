#!/usr/bin/env python3
"""Serve the live dashboard on localhost and drive npu-sim over its --live-stdio protocol.

Entry point for npuperf.live (also: python -m npuperf.live).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npuperf.live import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
