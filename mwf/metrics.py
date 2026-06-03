"""Macro one-vs-rest classification metrics (AUC, EER, P, R, F1, AP)."""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pyeer.eer_stats import get_eer_values
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

_MAX_SKIPPED_CLASSES_RATIO: Final[float] = 0.05


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Macro one-vs-rest classification metrics.

    Attributes:
        accuracy: Top-1 accuracy.
        balanced_accuracy: Top-1 accuracy averaged per class (recall macro-mean).
            Robust to unequal segments-per-subject, which plain accuracy is not;
            report it alongside accuracy for the closed-set identification task.
        auc: Macro one-vs-rest ROC AUC.
        eer: Macro equal-error rate.
        precision: Macro precision (``zero_division=0``).
        recall: Macro recall (``zero_division=0``).
        f1: Macro F1 (``zero_division=0``).
        ap: Macro average precision.
        n_classes_total: One-vs-rest classes considered.
        n_classes_evaluated_eer: Subset of classes that contributed to
            the macro EER (both positives and negatives present).
    """

    accuracy: float
    balanced_accuracy: float
    auc: float
    eer: float
    precision: float
    recall: float
    f1: float
    ap: float
    n_classes_total: int = 0
    n_classes_evaluated_eer: int = 0

    def as_dict(self) -> dict[str, float]:
        """Return a shallow ``dict`` of the metric fields."""
        return asdict(self)


def _binary_eer(
    y_true: NDArray[np.int64], y_score: NDArray[np.float64]
) -> float:
    """Compute EER for a single binary problem.

    Args:
        y_true: Binary ground truth (0/1).
        y_score: Continuous scores aligned with ``y_true``.

    Returns:
        Equal-error rate in ``[0, 1]``.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fmr, fnmr = fpr[::-1], (1.0 - tpr)[::-1]
    _, _, _, eer = get_eer_values(fmr, fnmr)
    return float(eer)


def _macro_eer(
    y_true_bin: NDArray[np.int64], y_score: NDArray[np.float64]
) -> tuple[float, int, int]:
    """Macro-averaged EER over non-degenerate classes.

    Args:
        y_true_bin: ``(n_samples, n_classes)`` one-hot truth matrix.
        y_score: ``(n_samples, n_classes)`` score matrix.

    Returns:
        Tuple ``(macro_eer, n_classes_total, n_classes_evaluated)``;
        ``macro_eer`` is NaN if every class was degenerate.
    """
    per_class: list[float] = []
    skipped = 0
    n_total = int(y_true_bin.shape[1])
    for col in range(n_total):
        pos = y_true_bin[:, col]
        n_pos = int(pos.sum())
        if n_pos in (0, pos.size):
            skipped += 1
            continue
        per_class.append(_binary_eer(pos, y_score[:, col]))

    if skipped:
        ratio = skipped / max(1, n_total)
        logger.info(
            "EER macro: skipped %d / %d classes (ratio %.3f).",
            skipped, n_total, ratio,
        )
        if ratio > _MAX_SKIPPED_CLASSES_RATIO:
            warnings.warn(
                f"EER macro skipped {ratio:.0%} of classes — split may be invalid.",
                RuntimeWarning, stacklevel=2,
            )
    eer = float(np.mean(per_class)) if per_class else float("nan")
    return eer, n_total, n_total - skipped


def _macro_auc(
    y_true_bin: NDArray[np.int64], y_score: NDArray[np.float64]
) -> float:
    """Macro one-vs-rest ROC AUC over non-degenerate classes.

    Mirrors :func:`_macro_eer`: one-vs-rest columns whose positives are
    absent (``n_pos == 0``) or fill the whole column (``n_pos == n``) carry
    no AUC information and would make :func:`roc_auc_score` raise. They are
    skipped, so a degenerate CV split yields a macro AUC over the evaluable
    classes instead of crashing.

    Args:
        y_true_bin: ``(n_samples, n_classes)`` one-hot truth matrix.
        y_score: ``(n_samples, n_classes)`` score matrix.

    Returns:
        Macro one-vs-rest AUC, or NaN if every class is degenerate.
    """
    per_class: list[float] = []
    for col in range(y_true_bin.shape[1]):
        pos = y_true_bin[:, col]
        n_pos = int(pos.sum())
        if n_pos in (0, pos.size):
            continue
        per_class.append(float(roc_auc_score(pos, y_score[:, col])))
    return float(np.mean(per_class)) if per_class else float("nan")


def _binarise(
    y_true: NDArray[np.int64], classes: NDArray[np.int64]
) -> NDArray[np.int64]:
    """One-vs-rest binarisation, expanding the binary case to two columns.

    Args:
        y_true: 1-D array of integer class labels.
        classes: 1-D ordered array of class labels.

    Returns:
        ``(n_samples, n_classes)`` one-hot matrix.
    """
    out = np.asarray(label_binarize(y_true, classes=classes), dtype=np.int64)
    return np.hstack([1 - out, out]) if out.shape[1] == 1 else out


def evaluate(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    y_score: NDArray[np.float64],
    classes: NDArray[np.int64],
) -> ClassificationMetrics:
    """Compute the macro classification metrics for one prediction.

    Args:
        y_true: ``(n,)`` ground-truth labels.
        y_pred: ``(n,)`` hard predictions.
        y_score: ``(n, n_classes)`` per-class scores aligned with ``classes``.
        classes: Ordered class labels matching the columns of ``y_score``.

    Returns:
        A :class:`ClassificationMetrics` with all six metrics plus EER counts.

    Raises:
        ValueError: If ``y_score`` is not 2-D or its column count differs
            from the number of classes.
    """
    if y_score.ndim != 2:
        raise ValueError("y_score must be 2-D (n_samples, n_classes).")
    if y_score.shape[1] != classes.shape[0]:
        raise ValueError(
            "y_score columns must equal the number of classes "
            f"({y_score.shape[1]} vs {classes.shape[0]})."
        )

    y_true_bin = _binarise(y_true, classes)
    eer, n_total, n_evaluated = _macro_eer(y_true_bin, y_score)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        auc=_macro_auc(y_true_bin, y_score),
        eer=eer,
        precision=float(precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        recall=float(recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        f1=float(f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        ap=float(average_precision_score(
            y_true_bin, y_score, average="macro"
        )),
        n_classes_total=n_total,
        n_classes_evaluated_eer=n_evaluated,
    )


__all__ = ["ClassificationMetrics", "evaluate"]
