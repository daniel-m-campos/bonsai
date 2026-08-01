"""python -m bonsai.bench dispatches the unified CLI (see bench/cli.py)."""

from __future__ import annotations

import sys

from bonsai.bench.cli import main

sys.exit(main())
