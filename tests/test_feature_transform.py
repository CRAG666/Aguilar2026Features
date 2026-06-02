"""Validate the BioHashing projection primitives and the multimodal transform."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.feature_transform import (
    FeatureScaler,
    ProjectionKey,
    derive_projection,
    multimodal_dims,
    projection_dim,
    transform_multimodal,
    transform_multimodal_batch,
)

RNG = np.random.default_rng(seed=20260520)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- Projection primitives --------------------------------------------------


def test_projection_key_rows_are_orthonormal():
    """``R Rᵀ = I_m`` — the projection rows form an orthonormal frame."""
    key = derive_projection("USER_A", in_dim=128, out_dim=40)
    assert isinstance(key, ProjectionKey)
    gram = key.matrix @ key.matrix.T
    np.testing.assert_allclose(gram, np.eye(40), atol=1e-10)


def test_projection_keys_are_decorrelated_across_tokens():
    """Two tokens must hash to near-orthogonal projection rows."""
    k0 = derive_projection("USER_A", 256, 64)
    k1 = derive_projection("USER_B", 256, 64)
    cross = k0.matrix @ k1.matrix.T  # (64, 64) cross-correlation
    # Off-diagonal entries scale as 1/√d ≈ 0.0625; 0.30 is a comfortable bound.
    assert float(np.abs(cross).max()) < 0.30


def test_projection_dim_respects_ratio_and_bounds():
    assert projection_dim(100, 0.5) == 50
    assert projection_dim(10, 1.0) == 10
    assert projection_dim(10, 0.01) == 1  # clipped up to ≥ 1
    with pytest.raises(ValueError):
        projection_dim(10, 0.0)
    with pytest.raises(ValueError):
        projection_dim(10, 1.5)


def test_feature_scaler_zscores_and_guards_constant_columns():
    X = RNG.normal(size=(50, 8))
    X[:, 3] = 7.0  # constant column → std 0, must not divide-by-zero
    scaler = FeatureScaler.fit(X)
    out = scaler.apply(X)
    assert np.all(np.isfinite(out))
    # Non-constant columns become unit-variance, zero-mean.
    np.testing.assert_allclose(out[:, 0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(out[:, 0].std(), 1.0, atol=1e-12)
    # Constant column maps to all-zeros (centred, scale guarded to 1).
    np.testing.assert_allclose(out[:, 3], 0.0, atol=1e-12)


# --- Multimodal hybrid transform (ECG BioHashing ‖ PPG IoM) -----------------

MULTI_DIM = 130  # level-4 multimodal vector (ECG 65 ‖ PPG 65)


def _multimodal_pair(d: int = MULTI_DIM, noise: float = 0.05):
    base = RNG.normal(size=d)
    return base + noise * RNG.normal(size=d), base + noise * RNG.normal(size=d)


def test_multimodal_output_dim_matches_helper():
    x = RNG.normal(size=MULTI_DIM)
    ecg_dim, ppg_dim = multimodal_dims(MULTI_DIM)
    out = transform_multimodal(x, "USER_A")
    assert out.shape == (ecg_dim + ppg_dim,)


def test_multimodal_is_deterministic():
    x = RNG.normal(size=MULTI_DIM)
    np.testing.assert_array_equal(
        transform_multimodal(x, "USER_A"), transform_multimodal(x, "USER_A")
    )


def test_multimodal_rejects_odd_length():
    with pytest.raises(ValueError):
        transform_multimodal(RNG.normal(size=129), "USER_A")


def test_multimodal_batch_matches_single_without_standardisation():
    X = np.vstack([RNG.normal(size=MULTI_DIM) for _ in range(4)])
    tokens = ["USER_A", "USER_A", "USER_B", "USER_C"]
    batch = transform_multimodal_batch(X, tokens, standardize=False)
    for row, tok in enumerate(tokens):
        np.testing.assert_allclose(
            batch[row], transform_multimodal(X[row], tok), atol=1e-12
        )


def test_multimodal_preserves_genuine_similarity_under_shared_key():
    """Same key, similar inputs ⇒ high cosine; impostor ⇒ low (key-independent LSH)."""
    gen, imp = [], []
    for t in range(20):
        g1, g2 = _multimodal_pair()
        other = RNG.normal(size=MULTI_DIM)
        tok = f"SHARED_{t % 3}"  # shared (stolen-token style) keys
        gen.append(_cos(transform_multimodal(g1, tok), transform_multimodal(g2, tok)))
        imp.append(_cos(transform_multimodal(g1, tok), transform_multimodal(other, tok)))
    assert np.mean(gen) > np.mean(imp) + 0.4


def test_multimodal_is_revocable():
    x = RNG.normal(size=MULTI_DIM)
    assert _cos(transform_multimodal(x, "USER_A"), transform_multimodal(x, "USER_B")) < 0.4
