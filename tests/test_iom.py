"""Validate Index-of-Max (IoM) hashing for the PPG block."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.iom import (
    derive_iom,
    hash_count,
    iom_dim,
    iom_indices,
    iom_onehot,
)

PPG_DIM = 65  # level-4 PPG block (13 descriptors × 5 subbands)
RNG = np.random.default_rng(seed=20260527)


def _correlated_pair(d: int = PPG_DIM, noise: float = 0.1):
    base = RNG.standard_normal(d)
    return base + noise * RNG.standard_normal(d), base + noise * RNG.standard_normal(d)


def test_hash_count_and_dim():
    assert hash_count(65, 0.25) == 16
    assert hash_count(10, 0.01) == 1  # floored to ≥ 1
    assert iom_dim(16, 16) == 256
    with pytest.raises(ValueError):
        hash_count(0)
    with pytest.raises(ValueError):
        hash_count(10, 0.0)


def test_indices_are_in_window_range():
    x = RNG.standard_normal(PPG_DIM)
    idx = iom_indices(x, "USER_A", n_hashes=16, window=16)
    assert idx.shape == (16,)
    assert idx.dtype == np.int64
    assert idx.min() >= 0 and idx.max() < 16


def test_onehot_has_one_active_per_hash_and_unit_norm():
    x = RNG.standard_normal(PPG_DIM)
    m, q = 16, 16
    code = iom_onehot(x, "USER_A", n_hashes=m, window=q, normalise=False)
    assert code.shape == (m * q,)
    assert code.sum() == m  # exactly one hot entry per hash
    unit = iom_onehot(x, "USER_A", n_hashes=m, window=q, normalise=True)
    np.testing.assert_allclose(np.linalg.norm(unit), 1.0, atol=1e-12)


def test_deterministic_for_same_token():
    x = RNG.standard_normal(PPG_DIM)
    np.testing.assert_array_equal(
        iom_onehot(x, "USER_A", 16, 16), iom_onehot(x, "USER_A", 16, 16)
    )


def test_similarity_preserving_genuine_over_impostor():
    """Genuine (similar) vectors collide more than impostor pairs — the LSH property."""
    gen, imp = [], []
    for t in range(40):
        g1, g2 = _correlated_pair()
        other = RNG.standard_normal(PPG_DIM)
        tok = f"K{t}"
        c1 = iom_onehot(g1, tok, 16, 16)
        c2 = iom_onehot(g2, tok, 16, 16)
        ci = iom_onehot(other, tok, 16, 16)
        gen.append(float(c1 @ c2))   # unit-norm → cosine == collision rate
        imp.append(float(c1 @ ci))
    assert np.mean(gen) > np.mean(imp) + 0.3


def test_revocable_new_token_decorrelates():
    x = RNG.standard_normal(PPG_DIM)
    same = float(iom_onehot(x, "USER_A", 16, 16) @ iom_onehot(x, "USER_A", 16, 16))
    diff = float(iom_onehot(x, "USER_A", 16, 16) @ iom_onehot(x, "USER_B", 16, 16))
    assert same == pytest.approx(1.0, abs=1e-12)
    assert diff < 0.4  # different token ⇒ near-baseline collision rate


def test_collision_rate_is_token_independent_under_shared_seed():
    """Two independent random vectors collide at ~1/q regardless of the token."""
    q = 16
    rates = []
    for t in range(60):
        a, b = RNG.standard_normal(PPG_DIM), RNG.standard_normal(PPG_DIM)
        ca = iom_onehot(a, f"S{t}", 64, q)
        cb = iom_onehot(b, f"S{t}", 64, q)
        rates.append(float(ca @ cb))
    assert abs(np.mean(rates) - 1.0 / q) < 0.03


def test_batch_matches_per_row():
    X = np.vstack([RNG.standard_normal(PPG_DIM) for _ in range(5)])
    batch = iom_onehot(X, "USER_A", 16, 16)
    assert batch.shape == (5, 16 * 16)
    for r in range(5):
        np.testing.assert_array_equal(batch[r], iom_onehot(X[r], "USER_A", 16, 16))


def test_derive_iom_shape_and_validation():
    w = derive_iom("USER_A", in_dim=PPG_DIM, n_hashes=16, window=16)
    assert w.shape == (16, 16, PPG_DIM)
    with pytest.raises(ValueError):
        derive_iom("USER_A", PPG_DIM, n_hashes=0, window=16)
    with pytest.raises(ValueError):
        derive_iom("USER_A", PPG_DIM, n_hashes=16, window=1)
