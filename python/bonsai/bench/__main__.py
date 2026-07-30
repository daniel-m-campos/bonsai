"""python -m bonsai.bench dispatches the unified CLI (see bench/cli.py)."""

import sys

from .cli import main

sys.exit(main())
