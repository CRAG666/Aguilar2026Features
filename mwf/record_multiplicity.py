"""Record-multiplicity (ARM) analysis for the BioHashing projection.

A cancelable template is *revoked* by handing the user a fresh token, which
re-seeds an independent orthonormal projection ``R``. This module asks the
attack-resistance question that single-template inversion (:mod:`mwf.inversion`)
cannot: **what does an adversary learn from several revoked templates of the
same biometric?** — the Attack via Record Multiplicity (ARM).

The ECG block is a real-valued BioHashing projection ``y = R x`` with
``R ∈ ℝ^{m×d}``, ``R Rᵀ = I_m`` and ``m = ratio·d`` (``ratio`` defaults to 0.5).
A single template leaks only ``x``'s component in the ``m``-dimensional row space
of ``R``. But each revocation re-issues an *independent* ``R_j``, so an attacker
who has collected the templates of ``k`` successive revocations holds ``k``
independent linear views of the **same** ``x``:

    y_1 = R_1 x,  …,  y_k = R_k x        ⇒        [R_1; …; R_k] x = [y_1; …; y_k].

Stacking them gives ``k·m`` equations in ``d`` unknowns; once the stack reaches
full column rank (``k·m ≥ d``, i.e. ``k ≥ 1/ratio`` — only **two** revocations at
``ratio = 0.5``) the least-squares pre-image recovers ``x`` *exactly*. This is the
record-multiplicity vulnerability of independent-token revocation; the
permutation/shared-subspace revocation of Kim & Chun (IEEE Access 2019) is the
standard fix, deferred to the mitigation experiment.

As in :mod:`mwf.inversion`, the analysis reconstructs from the **real-valued**
projection: sign-binarisation discards magnitudes and is strictly harder to
invert, so this is a conservative, adversary-favouring bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from typing import Final, Literal

from .feature_transform import (
    ECG_SALT,
    derive_projection,
    derive_revoked_projection,
    projection_dim,
)
from .inversion import _safe_corr
from .constants import DEFAULT_PROJECTION_RATIO

DEFAULT_RATIO = DEFAULT_PROJECTION_RATIO

Revocation = Literal["independent", "shared_subspace"]
INDEPENDENT: Final[str] = "independent"
SHARED_SUBSPACE: Final[str] = "shared_subspace"


@dataclass(frozen=True, slots=True)
class RecordMultiplicityReport:
    """ARM recovery quality after stacking ``n_templates`` revoked templates.

    Attributes:
        revocation: Revocation policy in force (``"independent"`` or
            ``"shared_subspace"``).
        n_templates: Number of revoked templates the adversary has stacked
            (``k``); the templates all protect the same feature vector.
        in_dim: Original block dimension ``d`` (the recovery target's length).
        out_dim_per_template: Projection rows per template ``m = ratio·d`` — the
            number of new linear measurements each revocation leaks.
        stacked_rank: Rank of the stacked projection ``[R_1; …; R_k]`` (``≤ d``);
            recovery is exact once this reaches ``d``.
        correlation: Pearson ``r`` between the least-squares pre-image and the
            original block — the leakage. Approaches ``1`` once the stack is
            full-rank.
        feature_recovery_energy_ratio: ``‖x̂‖² / ‖x‖²``, the fraction of the
            block energy captured by the stacked row space.
    """

    revocation: str
    n_templates: int
    in_dim: int
    out_dim_per_template: int
    stacked_rank: int
    correlation: float
    feature_recovery_energy_ratio: float


def revoked_projections(
    token: str,
    in_dim: int,
    out_dim: int,
    n_templates: int,
    revocation: Revocation = INDEPENDENT,
) -> list[NDArray[np.float64]]:
    """Return the projection of each successive revocation under a revocation policy.

    Args:
        token: Base user token.
        in_dim: Block dimension ``d``.
        out_dim: Projection rows per template ``m``.
        n_templates: Number of revocations to materialise.
        revocation: ``"independent"`` (the operational policy: re-issue tag
            ``::REV{j}`` seeds a wholly new, independent orthonormal ``R_j``) or
            ``"shared_subspace"`` (the hardened policy: a fixed base ``R_base``
            anchored on ``token`` rotated by a per-revocation ``Q_j`` — see
            :func:`mwf.feature_transform.derive_revoked_projection`).

    Returns:
        List of ``n_templates`` projection matrices, each ``(m, d)``.

    Raises:
        ValueError: If ``revocation`` is not a recognised policy.
    """
    if revocation == INDEPENDENT:
        return [
            derive_projection(f"{token}::REV{j}{ECG_SALT}", in_dim, out_dim).matrix
            for j in range(n_templates)
        ]
    if revocation == SHARED_SUBSPACE:
        base_token = f"{token}{ECG_SALT}"
        return [
            derive_revoked_projection(
                base_token, f"{token}::REV{j}{ECG_SALT}", in_dim, out_dim
            ).matrix
            for j in range(n_templates)
        ]
    raise ValueError(
        f"revocation must be {INDEPENDENT!r} or {SHARED_SUBSPACE!r}; got {revocation!r}."
    )


def record_multiplicity_leakage(
    block_features: NDArray[np.float64],
    token: str,
    n_templates: int,
    projection_ratio: float = DEFAULT_RATIO,
    revocation: Revocation = INDEPENDENT,
) -> list[RecordMultiplicityReport]:
    """Cumulative ARM recovery of one feature block from up to ``n_templates`` revocations.

    For each ``k = 1 … n_templates`` the adversary stacks the first ``k``
    projections of the same block and solves the least-squares pre-image. Under
    ``"independent"`` revocation the recovery correlation climbs to ``1`` as the
    stacked row space fills ``ℝ^d`` (at ``k ≈ 1/ratio``); under
    ``"shared_subspace"`` every revocation reuses one fixed row space, so the
    correlation stays pinned at the single-template leak.

    Args:
        block_features: 1-D feature block to attack (e.g. the ECG half).
        token: Base user token; revocations are derived from it.
        n_templates: Maximum number of revoked templates to stack (``≥ 1``).
        projection_ratio: BioHashing ``m/d`` per template.
        revocation: Revocation policy (see :func:`revoked_projections`).

    Returns:
        One :class:`RecordMultiplicityReport` per ``k`` in ``1 … n_templates``.

    Raises:
        ValueError: If ``block_features`` is not 1-D or ``n_templates < 1``.
    """
    if block_features.ndim != 1:
        raise ValueError("block_features must be a 1-D array.")
    if n_templates < 1:
        raise ValueError(f"n_templates must be ≥ 1; got {n_templates}.")

    x = block_features.astype(np.float64, copy=False)
    in_dim = int(x.shape[0])
    out_dim = projection_dim(in_dim, projection_ratio)
    matrices = revoked_projections(token, in_dim, out_dim, n_templates, revocation)

    x_energy = float(np.dot(x, x))
    reports: list[RecordMultiplicityReport] = []
    for k in range(1, n_templates + 1):
        stacked = np.vstack(matrices[:k])              # (k·m, d)
        y = stacked @ x                                # the k collected templates
        x_hat, *_ = np.linalg.lstsq(stacked, y, rcond=None)
        rec_energy = float(np.dot(x_hat, x_hat))
        reports.append(
            RecordMultiplicityReport(
                revocation=revocation,
                n_templates=k,
                in_dim=in_dim,
                out_dim_per_template=out_dim,
                stacked_rank=int(np.linalg.matrix_rank(stacked)),
                correlation=_safe_corr(x_hat, x),
                feature_recovery_energy_ratio=(
                    rec_energy / x_energy if x_energy > 0 else float("nan")
                ),
            )
        )
    return reports


__all__ = [
    "INDEPENDENT",
    "SHARED_SUBSPACE",
    "RecordMultiplicityReport",
    "Revocation",
    "record_multiplicity_leakage",
    "revoked_projections",
]
