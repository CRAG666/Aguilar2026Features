"""Console-silencing, deduplicating capture of Python warnings to a log file.

Import this and call :func:`install` *before* importing any noisy dependency
(it touches only the stdlib, so it does not drag in those packages itself) — that
way even import-time warnings, such as pyeer's ``pkg_resources`` deprecation, are
caught. The console then stays clean; each *distinct* warning is appended once to
the file set via :func:`set_logfile` (buffered in memory until a file is chosen).

Worker subprocesses are silenced through ``PYTHONWARNINGS`` so a parallel joblib
pool does not re-emit — and re-print — the same warning from every core.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TextIO

# Distinct-warning keys already written, so a warning is logged at most once.
_seen: set[str] = set()
# Warnings recorded before a log file is chosen (e.g. at import time).
_buffer: list[str] = []
_logfile: Path | None = None


def _record(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """Deduplicating ``warnings.showwarning`` replacement (file, not console)."""
    key = f"{category.__name__}|{filename}|{lineno}|{message}"
    if key in _seen:
        return
    _seen.add(key)
    entry = warnings.formatwarning(message, category, filename, lineno, line)
    if _logfile is None:
        _buffer.append(entry)
        return
    with _logfile.open("a", encoding="utf-8") as fh:
        fh.write(entry)


def install() -> None:
    """Redirect warnings off the console and silence worker subprocesses.

    Call as early as possible, before importing noisy packages. Idempotent.
    """
    warnings.showwarning = _record
    # Fresh loky/joblib interpreters read this at startup; ``setdefault`` keeps a
    # user-supplied filter (e.g. ``PYTHONWARNINGS=error``) intact.
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


def set_logfile(path: Path) -> None:
    """Point capture at ``path`` and flush every warning buffered so far."""
    global _logfile
    path.parent.mkdir(parents=True, exist_ok=True)
    _logfile = path
    if _buffer:
        with path.open("a", encoding="utf-8") as fh:
            fh.writelines(_buffer)
        _buffer.clear()
