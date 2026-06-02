"""Tests for the computational-cost benchmark helper."""

from __future__ import annotations

import numpy as np

from mwf.timing import TimingResult, benchmark


def test_benchmark_returns_populated_result():
    result = benchmark("noop", lambda: np.zeros(10), repeats=3, inner_loops=2)
    assert isinstance(result, TimingResult)
    assert result.repeats == 3
    assert result.inner_loops == 2
    assert result.mean_s >= 0.0
    assert result.median_s >= 0.0
    assert result.per_call_ms() >= 0.0


def test_benchmark_reports_array_output_bytes():
    result = benchmark("arr", lambda: np.zeros(128, dtype=np.float64), repeats=2)
    assert result.bytes_output == 128 * 8


def test_benchmark_sums_bytes_of_sequence_output():
    result = benchmark(
        "list", lambda: [np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)],
        repeats=2,
    )
    assert result.bytes_output == 2 * 4 * 8


def test_benchmark_single_repeat_has_zero_std():
    result = benchmark("one", lambda: 1, repeats=1)
    assert result.std_s == 0.0
    assert result.bytes_output == 0  # a Python int exposes no nbytes
