"""DET / CMC / operating-point curves for verification and identification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pyeer.eer_stats import get_fmr_op, get_fnmr_op
from sklearn.metrics import (
    average_precision_score,
    det_curve,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from statsmodels.stats.proportion import proportion_confint

DEFAULT_FMR_TARGETS: Final[tuple[float, ...]] = (1e-2, 1e-3, 1e-4)
DEFAULT_FNMR_TARGETS: Final[tuple[float, ...]] = (1e-2, 1e-3, 1e-4)
DEFAULT_RANKS: Final[tuple[int, ...]] = (1, 5, 10, 20)
DEFAULT_CI_ALPHA: Final[float] = 0.05  # 95 % Wilson CI


# ---------------------------------------------------------------------------
# Verification curves
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DetCurve:
    """DET curve in raw FMR/FNMR coordinates.

    Attributes:
        fmr: False-match rates.
        fnmr: False-non-match rates.
        thresholds: Decision thresholds aligned with ``fmr``/``fnmr``.
    """

    fmr: NDArray[np.float64]
    fnmr: NDArray[np.float64]
    thresholds: NDArray[np.float64]


def det_curve_from_scores(
    genuine: NDArray[np.float64], impostor: NDArray[np.float64]
) -> DetCurve:
    """Build a DET curve from genuine and impostor score pools.

    Args:
        genuine: 1-D array of genuine scores.
        impostor: 1-D array of impostor scores.

    Returns:
        A :class:`DetCurve`.
    """
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_score = np.concatenate([genuine, impostor])
    fmr, fnmr, thr = det_curve(y_true, y_score)
    return DetCurve(fmr=fmr, fnmr=fnmr, thresholds=thr)


def eer_from_scores(
    genuine: NDArray[np.float64], impostor: NDArray[np.float64]
) -> float:
    """Equal-error rate at the DET operating point where FMR ≈ FNMR.

    Args:
        genuine: 1-D array of genuine scores.
        impostor: 1-D array of impostor scores.

    Returns:
        ``0.5·(FMR + FNMR)`` at the crossover, or NaN if either pool has < 2.
    """
    if genuine.size < 2 or impostor.size < 2:
        return float("nan")
    det = det_curve_from_scores(genuine, impostor)
    idx = int(np.argmin(np.abs(det.fmr - det.fnmr)))
    return float(0.5 * (det.fmr[idx] + det.fnmr[idx]))


def bootstrap_eer_ci(
    genuine: NDArray[np.float64],
    impostor: NDArray[np.float64],
    *,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the EER of pooled genuine/impostor scores.

    Genuine and impostor pools are resampled with replacement independently, the
    EER is recomputed on each resample, and the central ``level`` percentile band
    is returned — a distribution-free interval for the pooled-score EER that the
    single point estimate lacks.

    Args:
        genuine: 1-D genuine score pool.
        impostor: 1-D impostor score pool.
        n_resamples: Bootstrap resamples.
        level: Confidence level in ``(0, 1)``.
        seed: RNG seed for the resampling.

    Returns:
        ``(low, high)`` EER bounds; ``(NaN, NaN)`` when either pool has < 2.
    """
    if genuine.size < 2 or impostor.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    eers = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        g = rng.choice(genuine, size=genuine.size, replace=True)
        i = rng.choice(impostor, size=impostor.size, replace=True)
        eers[b] = eer_from_scores(g, i)
    eers = eers[np.isfinite(eers)]
    if eers.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return (
        float(np.quantile(eers, tail)),
        float(np.quantile(eers, 1.0 - tail)),
    )


@dataclass(frozen=True, slots=True)
class RocCurve:
    """ROC curve in (FPR, TPR) coordinates with its summary AUC.

    Attributes:
        fpr: False-positive (false-match) rates.
        tpr: True-positive (genuine-accept) rates.
        thresholds: Decision thresholds aligned with ``fpr``/``tpr``.
        auc: Area under the ROC curve.
    """

    fpr: NDArray[np.float64]
    tpr: NDArray[np.float64]
    thresholds: NDArray[np.float64]
    auc: float


def roc_curve_from_scores(
    genuine: NDArray[np.float64], impostor: NDArray[np.float64]
) -> RocCurve:
    """Build a ROC curve from genuine and impostor score pools.

    Treats genuine comparisons as the positive class and impostor
    comparisons as the negative class, mirroring the convention used by
    :func:`det_curve_from_scores` so the ROC, DET and reported EER all
    derive from the same score pools.

    Args:
        genuine: 1-D array of genuine scores.
        impostor: 1-D array of impostor scores.

    Returns:
        A :class:`RocCurve`.
    """
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_score = np.concatenate([genuine, impostor])
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return RocCurve(fpr=fpr, tpr=tpr, thresholds=thr, auc=float(roc_auc_score(y_true, y_score)))


@dataclass(frozen=True, slots=True)
class PrCurve:
    """Precision-recall curve with its average-precision summary.

    Attributes:
        precision: Precision values.
        recall: Recall values aligned with ``precision``.
        thresholds: Decision thresholds (length ``len(precision) - 1``).
        average_precision: Average precision (AP) summary score.
    """

    precision: NDArray[np.float64]
    recall: NDArray[np.float64]
    thresholds: NDArray[np.float64]
    average_precision: float


def pr_curve_from_scores(
    genuine: NDArray[np.float64], impostor: NDArray[np.float64]
) -> PrCurve:
    """Build a precision-recall curve from genuine and impostor score pools.

    Uses the same positive/negative convention as
    :func:`roc_curve_from_scores`, so the AP summary is comparable with the
    DET/ROC curves built from the same pools.

    Args:
        genuine: 1-D array of genuine scores.
        impostor: 1-D array of impostor scores.

    Returns:
        A :class:`PrCurve`.
    """
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_score = np.concatenate([genuine, impostor])
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    return PrCurve(
        precision=precision,
        recall=recall,
        thresholds=thr,
        average_precision=float(average_precision_score(y_true, y_score)),
    )


def _wilson_ci(p_hat: float, n: int, alpha: float = DEFAULT_CI_ALPHA) -> tuple[float, float]:
    """Wilson score CI for a Bernoulli proportion.

    Args:
        p_hat: Observed proportion.
        n: Number of independent trials.
        alpha: Significance level.

    Returns:
        ``(low, high)`` bounds; ``(NaN, NaN)`` when ``n <= 0``.
    """
    if n <= 0:
        return float("nan"), float("nan")
    lo, hi = proportion_confint(
        count=int(round(p_hat * n)), nobs=n,
        alpha=alpha, method="wilson",
    )
    return float(lo), float(hi)


def _add_operating_point(
    out: dict[str, float], key: str, value: float, n: int, include_ci: bool
) -> None:
    """Insert one operating point and, when requested, its Wilson CI bounds.

    Args:
        out: Destination dict, mutated in place.
        key: Operating-point column name.
        value: Operating-point value.
        n: Trial count backing the Wilson CI.
        include_ci: Whether to add ``"<key>_ci_lo"`` / ``"<key>_ci_hi"``.
    """
    out[key] = value
    if include_ci:
        lo, hi = _wilson_ci(value, n)
        out[f"{key}_ci_lo"] = lo
        out[f"{key}_ci_hi"] = hi


def operating_points(
    genuine: NDArray[np.float64],
    impostor: NDArray[np.float64],
    fmr_targets: tuple[float, ...] = DEFAULT_FMR_TARGETS,
    fnmr_targets: tuple[float, ...] = DEFAULT_FNMR_TARGETS,
    *,
    include_ci: bool = True,
) -> dict[str, float]:
    """Compute FNMR@FMR and FMR@FNMR operating points.

    Args:
        genuine: 1-D array of genuine scores.
        impostor: 1-D array of impostor scores.
        fmr_targets: Target FMR values.
        fnmr_targets: Target FNMR values.
        include_ci: If ``True``, add Wilson 95 % CI bounds per operating point.

    Returns:
        Dict keyed by ``"fnmr_at_fmr_<v>"``/``"fmr_at_fnmr_<v>"`` (plus
        ``"_ci_lo"`` / ``"_ci_hi"`` when CIs are included).
    """
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_score = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fmr, fnmr = fpr[::-1], (1.0 - tpr)[::-1]
    n_impostor, n_genuine = impostor.size, genuine.size

    out: dict[str, float] = {}
    for op in fmr_targets:
        _add_operating_point(
            out, f"fnmr_at_fmr_{op:g}", float(get_fnmr_op(fmr, fnmr, op)[1]),
            n_genuine, include_ci,
        )
    for op in fnmr_targets:
        _add_operating_point(
            out, f"fmr_at_fnmr_{op:g}", float(get_fmr_op(fmr, fnmr, op)[1]),
            n_impostor, include_ci,
        )
    return out


# ---------------------------------------------------------------------------
# Identification curves (CMC)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CmcCurve:
    """Cumulative Match Characteristic curve.

    Attributes:
        ranks: 1-D array of ranks ``[1, max_rank]``.
        accuracies: Rank-``k`` accuracies aligned with ``ranks``.
    """

    ranks: NDArray[np.int64]
    accuracies: NDArray[np.float64]

    def rank_k(self, k: int) -> float:
        """Return the rank-``k`` accuracy.

        Args:
            k: Desired rank.

        Returns:
            Accuracy at rank ``k``.

        Raises:
            KeyError: If ``k`` is not present in the curve.
        """
        idx = int(np.searchsorted(self.ranks, k))
        if idx >= self.ranks.size or self.ranks[idx] != k:
            raise KeyError(f"Rank {k} not present in this CMC curve.")
        return float(self.accuracies[idx])


def cmc_curve(
    y_true: NDArray[np.int64],
    y_score: NDArray[np.float64],
    classes: NDArray[np.int64],
    max_rank: int | None = None,
) -> CmcCurve:
    """Build a CMC curve via :func:`top_k_accuracy_score`.

    Args:
        y_true: ``(n,)`` ground-truth labels.
        y_score: ``(n, n_classes)`` score matrix aligned with ``classes``.
        classes: Ordered class labels.
        max_rank: Maximum rank; defaults to ``n_classes``.

    Returns:
        A :class:`CmcCurve` covering ``ranks = 1 … max_rank``.
    """
    n_classes = classes.shape[0]
    upper = n_classes if max_rank is None else min(int(max_rank), n_classes)
    ranks = np.arange(1, upper + 1, dtype=np.int64)
    acc = np.fromiter(
        (
            top_k_accuracy_score(y_true, y_score, k=int(k), labels=classes)
            for k in ranks
        ),
        dtype=np.float64,
        count=ranks.size,
    )
    return CmcCurve(ranks=ranks, accuracies=acc)


def rank_k_accuracies(
    y_true: NDArray[np.int64],
    y_score: NDArray[np.float64],
    classes: NDArray[np.int64],
    ranks: tuple[int, ...] = DEFAULT_RANKS,
) -> dict[str, float]:
    """Return top-``k`` accuracy for each requested rank as a flat dict.

    Args:
        y_true: ``(n,)`` ground-truth labels.
        y_score: ``(n, n_classes)`` score matrix.
        classes: Ordered class labels.
        ranks: Ranks to evaluate.

    Returns:
        Dict keyed by ``"rank_<k>_accuracy"``.
    """
    n_classes = classes.shape[0]
    return {
        f"rank_{k}_accuracy": float(
            top_k_accuracy_score(
                y_true, y_score, k=int(min(k, n_classes)), labels=classes
            )
        )
        for k in ranks
    }


__all__ = [
    "CmcCurve",
    "DetCurve",
    "PrCurve",
    "RocCurve",
    "bootstrap_eer_ci",
    "cmc_curve",
    "det_curve_from_scores",
    "eer_from_scores",
    "operating_points",
    "pr_curve_from_scores",
    "rank_k_accuracies",
    "roc_curve_from_scores",
]
