#!/usr/bin/env python3
"""Check that event- and cycle-driven timing give identical results and compare their host
run times.

Entry point for npuperf.compare_modes (also: python -m npuperf.compare_modes).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npuperf.compare_modes import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
