"""Unit tests for the ``Datasets`` loader → ``BiometricSegments`` adapter.

Uses an in-memory fake loader, so no real data or the optional ``wfdb``
dependency is needed; the concrete loaders are exercised on real data elsewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.dataset import BiometricSegments, segments_from_loader

RNG = np.random.default_rng(seed=20260602)


class _FakeLoader:
    """Minimal stand-in for a ``Datasets`` loader exposing ECG, PPG (+ extras).

    Returns the per-signal arrays it was constructed with; ``get_signal_segments``
    yields only the requested signal plus its labels, exactly like the real base.
    """

    def __init__(self, signals: dict[str, np.ndarray], labels: np.ndarray, rates: dict[str, int]):
        self._signals = signals
        self._labels = labels
        self._rates = rates

    def sampling_rate_of(self, signal: str) -> int:
        return self._rates[signal]

    def get_signal_segments(self, signal: str) -> dict[str, np.ndarray]:
        return {signal: self._signals[signal], "labels": self._labels}


def _loader(n: int = 12, length: int = 64, rate: int = 125, *, ppg_rate: int | None = None):
    labels = np.repeat(np.arange(n // 3, dtype=np.int64), 3)[:n]
    signals = {
        "ECG": RNG.normal(size=(n, length)),
        "PPG": RNG.normal(size=(n, length)),
        "Resp": RNG.normal(size=(n, length)),  # extra signal the adapter must ignore
    }
    rates = {"ECG": rate, "PPG": ppg_rate or rate, "Resp": rate}
    return _FakeLoader(signals, labels, rates)


def test_adapter_returns_aligned_bundle():
    bundle = segments_from_loader(_loader(n=12, length=64))
    assert isinstance(bundle, BiometricSegments)
    assert bundle.ecg.shape == (12, 64) == bundle.ppg.shape
    assert bundle.labels.shape == (12,)
    assert bundle.sampling_rate == 125
    assert bundle.num_subjects == 4


def test_adapter_ignores_extra_signals():
    """A dataset exposing Resp (e.g. BIDMC) yields an ECG/PPG-only bundle.

    The adapter must wire ECG→ecg and PPG→ppg (not pick up the Resp channel),
    so the bundle arrays equal the loader's ECG/PPG and never its Resp.
    """
    loader = _loader()
    bundle = segments_from_loader(loader)
    np.testing.assert_array_equal(bundle.ecg, loader._signals["ECG"])
    np.testing.assert_array_equal(bundle.ppg, loader._signals["PPG"])
    assert not np.array_equal(bundle.ppg, loader._signals["Resp"])


def test_adapter_preserves_label_values():
    loader = _loader(n=9)
    bundle = segments_from_loader(loader)
    np.testing.assert_array_equal(bundle.labels, loader.get_signal_segments("ECG")["labels"])
    assert bundle.labels.dtype == np.int64


def test_adapter_rejects_mismatched_rates():
    with pytest.raises(ValueError, match="share a sampling rate"):
        segments_from_loader(_loader(rate=125, ppg_rate=64))


def test_adapter_rejects_mismatched_labels():
    loader = _loader(n=10, length=32)

    def _bad_segments(signal: str) -> dict[str, np.ndarray]:
        labels = loader._labels if signal == "ECG" else loader._labels[::-1]
        return {signal: loader._signals[signal], "labels": labels}

    loader.get_signal_segments = _bad_segments  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="different per-segment labels"):
        segments_from_loader(loader)


def test_adapted_bundle_validates_shapes():
    """ECG/PPG of different lengths must trip BiometricSegments' own validation."""
    loader = _loader(n=8, length=48)
    loader._signals["PPG"] = RNG.normal(size=(8, 50))  # length mismatch vs ECG
    with pytest.raises(ValueError, match="shape mismatch"):
        segments_from_loader(loader)
