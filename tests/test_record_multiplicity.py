"""Validate the record-multiplicity (ARM) analysis of the BioHashing projection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.record_multiplicity import (
    SHARED_SUBSPACE,
    RecordMultiplicityReport,
    record_multiplicity_leakage,
    revoked_projections,
)

BLOCK_DIM = 91  # e.g. the ECG half of multimodal bior3.3 level 6 (182 / 2)
RNG = np.random.default_rng(seed=20260527)


def test_returns_one_report_per_revocation_count():
    x = RNG.normal(size=BLOCK_DIM)
    reports = record_multiplicity_leakage(x, "user", n_templates=4)
    assert len(reports) == 4
    assert [r.n_templates for r in reports] == [1, 2, 3, 4]
    assert all(isinstance(r, RecordMultiplicityReport) for r in reports)


def test_quantities_in_legal_range():
    x = RNG.normal(size=BLOCK_DIM)
    for r in record_multiplicity_leakage(x, "user", n_templates=3):
        assert -1.0 <= r.correlation <= 1.0
        assert 0.0 <= r.feature_recovery_energy_ratio <= 1.0 + 1e-9
        assert 0 <= r.stacked_rank <= r.in_dim


def test_revocations_are_independent_projections():
    """Distinct revocation tags must seed independent projections.

    Each ``(out_dim=60, d=91)`` projection has rank 60; were the two revocations
    the same matrix, stacking them would stay rank 60. Reaching full rank 91
    proves they span independent subspaces — the root of the ARM leak.
    """
    mats = revoked_projections("user", BLOCK_DIM, out_dim=60, n_templates=2)
    stacked = np.vstack(mats)
    assert np.linalg.matrix_rank(stacked) == BLOCK_DIM


def test_recovery_climbs_to_exact_with_enough_revocations():
    """At ratio 0.5 a second revocation makes the stack full-rank → exact recovery.

    This is the vulnerability the test exists to catch: independent-token
    revocation leaks the whole block once the adversary collects 1/ratio of them.
    """
    x = RNG.normal(size=BLOCK_DIM)
    reports = record_multiplicity_leakage(x, "user", n_templates=2, projection_ratio=0.5)
    first, second = reports[0], reports[1]
    assert first.correlation < 0.95            # one template: only the row space leaks
    assert second.stacked_rank == BLOCK_DIM    # two templates: full rank
    assert second.correlation > 0.999          # → exact recovery
    assert second.feature_recovery_energy_ratio == pytest.approx(1.0, abs=1e-6)


def test_lower_ratio_needs_more_revocations_for_full_rank():
    x = RNG.normal(size=BLOCK_DIM)
    # ratio 0.25 → each template leaks ~d/4 measurements; one is far from full rank.
    reports = record_multiplicity_leakage(x, "user", n_templates=1, projection_ratio=0.25)
    assert reports[0].stacked_rank < BLOCK_DIM
    assert reports[0].correlation < 0.95


def test_shared_subspace_revocation_caps_recovery():
    """The hardened policy reuses one fixed row space → ARM cannot improve recovery.

    Stacking any number of rotated re-issues never exceeds rank ``m``, so the
    correlation stays pinned at the single-template leak instead of reaching 1.
    """
    x = RNG.normal(size=BLOCK_DIM)
    reports = record_multiplicity_leakage(
        x, "user", n_templates=4, projection_ratio=0.5, revocation=SHARED_SUBSPACE,
    )
    m = reports[0].out_dim_per_template
    assert all(r.revocation == SHARED_SUBSPACE for r in reports)
    assert all(r.stacked_rank == m for r in reports)        # rank never grows past m
    assert all(r.correlation < 0.95 for r in reports)       # no exact recovery, ever
    # The leak is flat: the 4th re-issue is no more revealing than the 1st.
    assert reports[-1].correlation == pytest.approx(reports[0].correlation, abs=1e-6)


def test_shared_subspace_reissues_are_decorrelated():
    """Hardened re-issues must still be diverse (renewable/unlinkable)."""
    mats = revoked_projections(
        "user", BLOCK_DIM, out_dim=46, n_templates=2, revocation=SHARED_SUBSPACE,
    )
    x = RNG.normal(size=BLOCK_DIM)
    t1, t2 = mats[0] @ x, mats[1] @ x
    assert abs(np.corrcoef(t1, t2)[0, 1]) < 0.5             # templates decorrelated
    # ...yet they live in the same row space: stacking does not add rank.
    assert np.linalg.matrix_rank(np.vstack(mats)) == 46


def test_unknown_revocation_policy_rejected():
    with pytest.raises(ValueError, match="revocation"):
        revoked_projections("user", BLOCK_DIM, out_dim=46, n_templates=2, revocation="nope")


def test_rejects_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        record_multiplicity_leakage(RNG.normal(size=(2, BLOCK_DIM)), "user", n_templates=2)


def test_rejects_non_positive_template_count():
    with pytest.raises(ValueError, match="n_templates"):
        record_multiplicity_leakage(RNG.normal(size=BLOCK_DIM), "user", n_templates=0)
