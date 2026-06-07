"""Tests for the deduplicating, console-silencing warning capture."""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


@pytest.fixture
def capture():
    """A fresh ``warning_capture`` module with the global state reset."""
    module = importlib.import_module("warning_capture")
    original_showwarning = warnings.showwarning
    module._seen.clear()
    module._buffer.clear()
    module._logfile = None
    yield module
    warnings.showwarning = original_showwarning
    module._seen.clear()
    module._buffer.clear()
    module._logfile = None


def _emit(module, message: str) -> None:
    module._record(UserWarning(message), UserWarning, "file.py", 1)


def test_identical_warnings_logged_once(capture, tmp_path):
    capture.set_logfile(tmp_path / "warnings.log")
    _emit(capture, "duplicate")
    _emit(capture, "duplicate")
    _emit(capture, "distinct")

    lines = (tmp_path / "warnings.log").read_text(encoding="utf-8").splitlines()
    bodies = [line for line in lines if "UserWarning" in line]
    assert sum("duplicate" in b for b in bodies) == 1
    assert sum("distinct" in b for b in bodies) == 1


def test_import_time_warnings_buffered_then_flushed(capture, tmp_path):
    # No log file yet: the warning is buffered, nothing on disk.
    _emit(capture, "before-file")
    assert capture._buffer

    capture.set_logfile(tmp_path / "warnings.log")
    assert not capture._buffer
    assert "before-file" in (tmp_path / "warnings.log").read_text(encoding="utf-8")


def test_install_silences_subprocess_warnings(capture, monkeypatch):
    monkeypatch.delenv("PYTHONWARNINGS", raising=False)
    capture.install()
    assert warnings.showwarning is capture._record
    assert __import__("os").environ["PYTHONWARNINGS"] == "ignore"
