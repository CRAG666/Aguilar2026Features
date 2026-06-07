"""Tests for the tqdm/joblib progress helpers."""

from __future__ import annotations

import joblib
import pytest

from mwf.batch_utils import parallel_map
from mwf.progress import progress_disabled, track, tqdm_joblib


def test_track_yields_every_item():
    assert list(track(range(5), desc="t")) == [0, 1, 2, 3, 4]


def test_progress_disabled_honours_env(monkeypatch):
    monkeypatch.delenv("AGUILAR_FEATURES_NO_PROGRESS", raising=False)
    assert progress_disabled() is False
    monkeypatch.setenv("AGUILAR_FEATURES_NO_PROGRESS", "1")
    assert progress_disabled() is True
    monkeypatch.setenv("AGUILAR_FEATURES_NO_PROGRESS", "0")
    assert progress_disabled() is False


def test_tqdm_joblib_restores_callback():
    original = joblib.parallel.BatchCompletionCallBack
    with tqdm_joblib(3, desc="t"):
        assert joblib.parallel.BatchCompletionCallBack is not original
    assert joblib.parallel.BatchCompletionCallBack is original


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_parallel_map_matches_sequential(n_jobs):
    items = list(range(6))
    assert parallel_map(items, lambda x: x * x, n_jobs=n_jobs, desc="t") == [
        x * x for x in items
    ]
