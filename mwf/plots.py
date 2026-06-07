"""Publication figures for the BioHashing/IoM cancelable experiments.

Mirrors the figure suite of the signal-domain sibling (``Aguilar2026Wavelet``),
adapted to this project's clean, single-scheme protocol (no chaotic-map / noise
sweep):

* per-regime **DET**, **ROC** and **PR** curves from the closed-set score pools;
* per-regime genuine/impostor **score KDEs**;
* a **regime summary** (EER & AUC vs protected-template size) and a
  **classifier comparison** (AP & EER vs template size) read from the metrics
  frame.

Plotting is decoupled from the pipeline: the curve/KDE figures take precomputed
genuine/impostor score pools (a ``{regime_label: (genuine, impostor)}`` map),
and the summary figures take the identification-metrics :class:`DataFrame`. The
caller (``scripts/run_experiment.py``) owns template building and scoring.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless figure export

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from .operating_curves import (  # noqa: E402
    det_curve_from_scores,
    pr_curve_from_scores,
    roc_curve_from_scores,
)

# {regime label -> (genuine scores, impostor scores)}.
ScorePools = Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]]

_DPI = 150


def _save(fig: plt.Figure, out_path: Path) -> None:
    """Write ``fig`` to ``out_path`` (creating parents) and close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)


def _annotate(ax: plt.Axes, lines: list[str]) -> None:
    """Place quantitative annotations in a boxed corner of the axes (no title)."""
    if not lines:
        return
    ax.text(
        0.97, 0.97, "\n".join(lines), transform=ax.transAxes,
        ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )


def plot_det_curves(pools: ScorePools, out_path: Path) -> None:
    """Overlay per-regime DET curves (log-log FMR vs FNMR)."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for label, (genuine, impostor) in pools.items():
        det = det_curve_from_scores(genuine, impostor)
        ax.loglog(np.maximum(det.fmr, 1e-6), np.maximum(det.fnmr, 1e-6), label=label)
    ax.set_xlabel("False Match Rate")
    ax.set_ylabel("False Non-Match Rate")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_roc_curves(pools: ScorePools, out_path: Path) -> None:
    """Overlay per-regime ROC curves with their verification AUC in the legend."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for label, (genuine, impostor) in pools.items():
        roc = roc_curve_from_scores(genuine, impostor)
        ax.plot(roc.fpr, roc.tpr, label=f"{label} (AUC = {roc.auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out_path)


def plot_pr_curves(pools: ScorePools, out_path: Path) -> None:
    """Overlay per-regime precision-recall curves with their AP in the legend."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for label, (genuine, impostor) in pools.items():
        pr = pr_curve_from_scores(genuine, impostor)
        ax.plot(pr.recall, pr.precision, label=f"{label} (AP = {pr.average_precision:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, out_path)


def plot_score_distribution(
    genuine: NDArray[np.float64],
    impostor: NDArray[np.float64],
    out_path: Path,
) -> None:
    """Plot the genuine/impostor cosine-score KDEs for one regime."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(genuine, ax=ax, label="Genuine", fill=True, alpha=0.4)
    sns.kdeplot(impostor, ax=ax, label="Impostor", fill=True, alpha=0.4)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.legend()
    _save(fig, out_path)


def plot_regime_summary(
    metrics_df: pd.DataFrame,
    out_path: Path,
    size_col: str = "n_template_dims",
) -> None:
    """Plot EER and AUC versus template size for the best classifier per regime.

    Reports the single best-performing classifier per regime (the one with the
    lowest mean EER across template sizes) rather than averaging over all
    classifiers: a mean over classifiers is dominated by the weak ones (e.g. the
    decision tree) and is not a meaningful operating point.

    Args:
        metrics_df: Identification-metrics frame (``metrics.csv``); needs the
            ``regime``, ``classifier``, ``size_col``, ``eer_mean`` and
            ``auc_mean`` columns.
        out_path: PNG output path.
        size_col: Column used for the x-axis (protected-template size).
    """
    if metrics_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for regime, group in metrics_df.groupby("regime"):
        best_clf = group.groupby("classifier")["eer_mean"].mean().idxmin()
        best = group[group["classifier"] == best_clf].sort_values(size_col)
        label = f"{regime} ({best_clf})"
        axes[0].plot(best[size_col], best["eer_mean"], marker="o", label=label)
        axes[1].plot(best[size_col], best["auc_mean"], marker="o", label=label)
    axes[0].set(xlabel="Template size", ylabel="EER")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Template size", ylabel="AUC")
    axes[1].legend(fontsize=8)
    _save(fig, out_path)


def plot_classifier_comparison(
    metrics_df: pd.DataFrame,
    out_path: Path,
    regime: str,
    size_col: str = "n_template_dims",
) -> None:
    """Plot AP and EER versus template size, one line per classifier, for a regime.

    Args:
        metrics_df: Identification-metrics frame (``metrics.csv``); needs the
            ``regime``, ``classifier``, ``size_col``, ``ap_mean`` and
            ``eer_mean`` columns.
        out_path: PNG output path.
        regime: Regime to filter on (e.g. ``"per_subject"``).
        size_col: Column used for the x-axis (protected-template size).
    """
    sub = metrics_df[metrics_df["regime"] == regime]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for classifier, group in sub.groupby("classifier"):
        agg = group.groupby(size_col)[["ap_mean", "eer_mean"]].mean().reset_index()
        axes[0].plot(agg[size_col], agg["ap_mean"], marker="o", label=classifier)
        axes[1].plot(agg[size_col], agg["eer_mean"], marker="o", label=classifier)
    axes[0].set(xlabel="Template size", ylabel="AP")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Template size", ylabel="EER")
    axes[1].legend(fontsize=8)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Cancelability story (IoM-specific; not in the signal-domain sibling)
# ---------------------------------------------------------------------------


def plot_stolen_token_scores(
    genuine: NDArray[np.float64],
    impostor: NDArray[np.float64],
    out_path: Path,
    eer: float | None = None,
    decidability: float | None = None,
) -> None:
    """Genuine/impostor KDEs under a *shared* (stolen) token.

    With the key neutralised, the separation here is carried by the biometric
    alone (the honest cancelable-security figure). EER and decidability are
    annotated in-axes when provided.

    Args:
        genuine: Genuine comparison scores under the stolen token.
        impostor: Impostor comparison scores under the stolen token.
        out_path: PNG output path.
        eer: Equal-error rate to annotate (optional).
        decidability: Daugman's ``d'`` to annotate (optional).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(genuine, ax=ax, label="Genuine", fill=True, alpha=0.4)
    sns.kdeplot(impostor, ax=ax, label="Impostor", fill=True, alpha=0.4)
    ax.set_xlabel("Similarity score")
    ax.set_ylabel("Density")
    annotations = []
    if eer is not None:
        annotations.append(f"EER = {eer:.3f}")
    if decidability is not None:
        annotations.append(f"d' = {decidability:.2f}")
    _annotate(ax, annotations)
    ax.legend()
    _save(fig, out_path)


def plot_non_invertibility(
    pools: Mapping[str, NDArray[np.float64]],
    out_path: Path,
    sar_type1: float | None = None,
    sar_type2: float | None = None,
) -> None:
    """KDEs of the three Wu-style correlation pools, with SAR annotations.

    Mated should sit between the non-mated baseline (chance) and the genuine
    reference (intra-subject ceiling). A non-invertible transform pushes mated
    onto non-mated; an invertible one pushes mated onto the reference.

    Args:
        pools: ``{"mated", "non_mated", "genuine_ref"}`` to absolute correlation
            samples.
        out_path: PNG output path.
        sar_type1: Optional protected-system SAR for the in-axes annotation.
        sar_type2: Optional raw-feature SAR for the in-axes annotation.
    """
    if not pools:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    styles = {
        "non_mated": ("#9e9e9e", "Non-mated (chance baseline)"),
        "mated": ("#d62728", "Mated reconstruction"),
        "genuine_ref": ("#2ca02c", "Genuine reference (intra-subject)"),
    }
    for key, (colour, label) in styles.items():
        arr = pools.get(key)
        if arr is None or arr.size < 2:
            continue
        sns.kdeplot(arr, ax=ax, fill=True, alpha=0.35, color=colour, label=label)
    ax.set_xlabel("Absolute reconstruction-reference correlation")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    annotations = []
    if sar_type1 is not None and np.isfinite(sar_type1):
        annotations.append(f"SAR-I = {sar_type1:.3f}")
    if sar_type2 is not None and np.isfinite(sar_type2):
        annotations.append(f"SAR-II = {sar_type2:.3f}")
    _annotate(ax, annotations)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle=":")
    _save(fig, out_path)


__all__ = [
    "ScorePools",
    "plot_classifier_comparison",
    "plot_det_curves",
    "plot_non_invertibility",
    "plot_pr_curves",
    "plot_regime_summary",
    "plot_roc_curves",
    "plot_score_distribution",
    "plot_stolen_token_scores",
]
