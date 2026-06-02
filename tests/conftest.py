"""Test bootstrap: put the project package root on ``sys.path``.

Running ``pytest`` from the repo root does not, by default, make the top-level
``mwf`` package importable from inside ``tests/`` (pytest's ``prepend`` import
mode inserts the test directory, not the project root). Prepending the project
root here keeps ``from mwf.X import ...`` working without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
