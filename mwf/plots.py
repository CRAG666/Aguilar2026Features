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

# {regime label → (genuine scores, impostor scores)}.
ScorePools = Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]]

_DPI = 150


def _save(fig: plt.Figure, out_path: Path) -> None:
    """Write ``fig`` to ``out_path`` (creating parents) and close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)


def plot_det_curves(pools: ScorePools, out_path: Path, title: str = "DET") -> None:
    """Overlay per-regime DET curves (log-log FMR vs FNMR)."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for label, (genuine, impostor) in pools.items():
        det = det_curve_from_scores(genuine, impostor)
        ax.loglog(np.maximum(det.fmr, 1e-6), np.maximum(det.fnmr, 1e-6), label=label)
    ax.set_xlabel("FMR")
    ax.set_ylabel("FNMR")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":")
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_roc_curves(pools: ScorePools, out_path: Path, title: str = "ROC") -> None:
    """Overlay per-regime ROC curves with their verification AUC in the legend."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for label, (genuine, impostor) in pools.items():
        roc = roc_curve_from_scores(genuine, impostor)
        ax.plot(roc.fpr, roc.tpr, label=f"{label} (AUC = {roc.auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate (FMR)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out_path)


def plot_pr_curves(pools: ScorePools, out_path: Path, title: str = "Precision–Recall") -> None:
    """Overlay per-regime precision-recall curves with their AP in the legend."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for label, (genuine, impostor) in pools.items():
        pr = pr_curve_from_scores(genuine, impostor)
        ax.plot(pr.recall, pr.precision, label=f"{label} (AP = {pr.average_precision:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, out_path)


def plot_score_distribution(
    genuine: NDArray[np.float64],
    impostor: NDArray[np.float64],
    out_path: Path,
    title: str = "Score distribution",
) -> None:
    """Plot the genuine/impostor cosine-score KDEs for one regime."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(genuine, ax=ax, label="genuine", fill=True, alpha=0.4)
    sns.kdeplot(impostor, ax=ax, label="impostor", fill=True, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
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
    axes[0].set(title="EER vs template size (best classifier)",
                xlabel="Template size", ylabel="EER")
    axes[0].legend(fontsize=8)
    axes[1].set(title="AUC vs template size (best classifier)",
                xlabel="Template size", ylabel="AUC")
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
    axes[0].set(title=f"AP vs template size — {regime}", xlabel="Template size", ylabel="AP")
    axes[0].legend(fontsize=8)
    axes[1].set(title=f"EER vs template size — {regime}", xlabel="Template size", ylabel="EER")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Classifier comparison — {regime}")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Cancelability story (IoM-specific; not in the signal-domain sibling)
# ---------------------------------------------------------------------------


def plot_inversion_leakage(
    inversion_df: pd.DataFrame,
    out_path: Path,
    title: str = "Inversion leakage — IoM PPG vs linear baseline",
) -> None:
    """Compare recovered-feature correlation: IoM PPG vs the linear baseline.

    The headline non-invertibility figure. For each probed segment the bars show
    the absolute correlation between an adversary's best-effort reconstruction
    and the original descriptors, for: the PPG block under plain BioHashing
    (``ppg_linear_correlation``), the PPG block under IoM (``ppg_correlation`` —
    ours), and the ECG block under BioHashing (``ecg_correlation``, for context).
    Lower IoM than linear ⇒ the IoM transform leaks less.

    Args:
        inversion_df: Per-segment frame from the inversion analysis
            (``inversion.csv``).
        out_path: PNG output path.
        title: Figure title.
    """
    if inversion_df is None or inversion_df.empty:
        return
    columns = [
        ("PPG · linear\n(BioHashing)", "ppg_linear_correlation", "#bdbdbd"),
        ("PPG · IoM\n(ours)", "ppg_correlation", "#1f77b4"),
        ("ECG · linear\n(BioHashing)", "ecg_correlation", "#bdbdbd"),
    ]
    data, labels, colours = [], [], []
    for label, col, colour in columns:
        if col in inversion_df.columns:
            vals = inversion_df[col].abs().dropna().to_numpy()
            if vals.size:
                data.append(vals)
                labels.append(label)
                colours.append(colour)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    ax.set_ylabel("|recovered ↔ original| correlation")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":")
    _save(fig, out_path)


def plot_record_multiplicity(
    arm_df: pd.DataFrame,
    out_path: Path,
    title: str = "Record-multiplicity attack on revoked templates",
) -> None:
    """Plot ECG-block recovery correlation vs the number of revoked templates.

    One curve (mean ± min–max band over probed segments) per revocation policy.
    ``independent``-token revocation leaks fresh measurements each re-issue, so the
    correlation climbs to ~1 once the stack is full-rank; ``shared_subspace``
    rotation reuses one fixed row space, so it stays pinned at the one-template
    leak — the record-multiplicity hole and its fix, side by side.

    Args:
        arm_df: Per-(segment, revocation, n_templates) frame (``arm.csv``) with
            the ``n_templates``, ``correlation`` and (optional) ``revocation``
            columns.
        out_path: PNG output path.
        title: Figure title.
    """
    if arm_df is None or arm_df.empty:
        return
    styles = {
        "independent": ("#d62728", "independent token (current)"),
        "shared_subspace": ("#2ca02c", "shared subspace Q·R (hardened)"),
    }
    policies = (
        [p for p in styles if p in set(arm_df["revocation"])]
        if "revocation" in arm_df.columns
        else ["independent"]
    )
    ks = np.asarray(sorted(arm_df["n_templates"].unique()), dtype=int)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for policy in policies:
        sub = arm_df[arm_df["revocation"] == policy] if "revocation" in arm_df.columns else arm_df
        grouped = sub.groupby("n_templates")["correlation"]
        mean = grouped.mean().reindex(ks).to_numpy()
        lo = grouped.min().reindex(ks).to_numpy()
        hi = grouped.max().reindex(ks).to_numpy()
        colour, label = styles.get(policy, ("#1f77b4", policy))
        ax.fill_between(ks, lo, hi, alpha=0.18, color=colour)
        ax.plot(ks, mean, marker="o", color=colour, label=label)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="exact recovery")
    ax.set_xlabel("number of revoked templates collected")
    ax.set_ylabel("|recovered ↔ original| correlation")
    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, linestyle=":")
    ax.legend(fontsize=8, loc="center right")
    _save(fig, out_path)


def plot_stolen_token_scores(
    genuine: NDArray[np.float64],
    impostor: NDArray[np.float64],
    out_path: Path,
    eer: float | None = None,
    decidability: float | None = None,
    title: str = "Stolen-token verification (worst case)",
) -> None:
    """Genuine/impostor KDEs under a *shared* (stolen) token.

    With the key neutralised, the separation here is carried by the biometric
    alone — the honest cancelable-security figure. EER and decidability are
    annotated when provided.

    Args:
        genuine: Genuine comparison scores under the stolen token.
        impostor: Impostor comparison scores under the stolen token.
        out_path: PNG output path.
        eer: Equal-error rate to annotate (optional).
        decidability: Daugman's ``d'`` to annotate (optional).
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(genuine, ax=ax, label="genuine", fill=True, alpha=0.4)
    sns.kdeplot(impostor, ax=ax, label="impostor", fill=True, alpha=0.4)
    ax.set_xlabel("similarity score")
    ax.set_ylabel("density")
    annotations = []
    if eer is not None:
        annotations.append(f"EER = {eer:.3f}")
    if decidability is not None:
        annotations.append(f"d' = {decidability:.2f}")
    ax.set_title(title + (f"  ({', '.join(annotations)})" if annotations else ""))
    ax.legend()
    _save(fig, out_path)


def plot_non_invertibility(
    pools: Mapping[str, NDArray[np.float64]],
    out_path: Path,
    sar_type1: float | None = None,
    sar_type2: float | None = None,
    title: str = "Non-invertibility — Wu-style 3-distribution report",
) -> None:
    """KDEs of the three Wu-style correlation pools, with SAR annotations.

    Mated should sit between the non-mated baseline (chance) and the genuine
    reference (intra-subject ceiling). A non-invertible transform pushes mated
    onto non-mated; an invertible one pushes mated onto the reference.

    Args:
        pools: ``{"mated", "non_mated", "genuine_ref"}`` → absolute correlation
            samples.
        out_path: PNG output path.
        sar_type1: Optional protected-system SAR for the title annotation.
        sar_type2: Optional raw-feature SAR for the title annotation.
        title: Figure title (annotations appended).
    """
    if not pools:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    styles = {
        "non_mated": ("#9e9e9e", "non-mated (chance baseline)"),
        "mated": ("#d62728", "mated reconstruction"),
        "genuine_ref": ("#2ca02c", "genuine reference (intra-subject)"),
    }
    for key, (colour, label) in styles.items():
        arr = pools.get(key)
        if arr is None or arr.size < 2:
            continue
        sns.kdeplot(arr, ax=ax, fill=True, alpha=0.35, color=colour, label=label)
    ax.set_xlabel("|reconstruction ↔ reference| correlation")
    ax.set_ylabel("density")
    ax.set_xlim(0.0, 1.0)
    annotations = []
    if sar_type1 is not None and np.isfinite(sar_type1):
        annotations.append(f"SAR-I = {sar_type1:.3f}")
    if sar_type2 is not None and np.isfinite(sar_type2):
        annotations.append(f"SAR-II = {sar_type2:.3f}")
    ax.set_title(title + (f"  ({', '.join(annotations)})" if annotations else ""))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle=":")
    _save(fig, out_path)


def plot_per_subject_ablation(
    ablation_df: pd.DataFrame,
    out_path: Path,
    title: str = "Per-subject ablation — key vs biometric contribution",
) -> None:
    """EER and within/across-group impostor means as a function of group size.

    Two panels: (i) EER vs group size (log-x), going from per_subject at
    ``group_size = 1`` to single_key at the cohort cap; (ii) within-group vs
    across-group impostor cosine means, which separate the biometric-only
    floor from the token-induced separation.

    Args:
        ablation_df: Per-group-size frame (``per_subject_ablation.csv``) with
            ``group_size``, ``eer``, ``within_group_impostor_mean``,
            ``across_group_impostor_mean`` and ``genuine_mean`` columns.
        out_path: PNG output path.
        title: Figure title.
    """
    if ablation_df is None or ablation_df.empty:
        return
    sub = ablation_df.sort_values("group_size")
    sizes = sub["group_size"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(sizes, sub["eer"].to_numpy(), marker="o", color="#1f77b4")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("subjects per shared token (1 = per_subject)")
    axes[0].set_ylabel("verification EER")
    axes[0].set_title("EER vs token-sharing group size")
    axes[0].grid(True, which="both", linestyle=":")

    axes[1].plot(sizes, sub["genuine_mean"].to_numpy(),
                 marker="o", color="#2ca02c", label="genuine")
    if "within_group_impostor_mean" in sub.columns:
        within = sub["within_group_impostor_mean"].to_numpy()
        axes[1].plot(sizes, within, marker="s", color="#d62728",
                     label="impostor (same token)")
    if "across_group_impostor_mean" in sub.columns:
        across = sub["across_group_impostor_mean"].to_numpy()
        axes[1].plot(sizes, across, marker="^", color="#9e9e9e",
                     label="impostor (different token)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("subjects per shared token (1 = per_subject)")
    axes[1].set_ylabel("mean cosine score")
    axes[1].set_title("Score means vs token-sharing group size")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", linestyle=":")
    fig.suptitle(title)
    _save(fig, out_path)


def plot_ratio_sweep(
    sweep_df: pd.DataFrame,
    out_path: Path,
    title: str = "Recognition–leakage trade-off vs BioHashing ratio",
) -> None:
    """EER and inversion correlation vs the BioHashing ratio ``m/d``.

    Two panels: (i) single-key and stolen-token EER vs ratio (the recognition
    axis); (ii) mean inversion correlation vs ratio (the leakage axis). The
    operating-point sweet spot is where both curves stay low simultaneously.

    Args:
        sweep_df: Per-ratio frame (``ratio_sweep.csv``) with
            ``ratio``, ``single_key_eer``, ``stolen_token_eer``,
            ``inversion_mean_correlation`` and ``inversion_std_correlation``.
        out_path: PNG output path.
        title: Figure title.
    """
    if sweep_df is None or sweep_df.empty:
        return
    sub = sweep_df.sort_values("ratio")
    r = sub["ratio"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(r, sub["single_key_eer"].to_numpy(),
                 marker="o", color="#1f77b4", label="single-key EER")
    axes[0].plot(r, sub["stolen_token_eer"].to_numpy(),
                 marker="s", color="#d62728", label="stolen-token EER")
    axes[0].set_xlabel("BioHashing ratio m/d")
    axes[0].set_ylabel("EER")
    axes[0].set_title("Recognition vs ratio")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle=":")

    inv = sub["inversion_mean_correlation"].to_numpy()
    err = sub["inversion_std_correlation"].to_numpy() if "inversion_std_correlation" in sub.columns else None
    if err is not None:
        axes[1].errorbar(r, inv, yerr=err, marker="o", color="#2ca02c",
                         label="|recovered ↔ original| corr")
    else:
        axes[1].plot(r, inv, marker="o", color="#2ca02c",
                     label="|recovered ↔ original| corr")
    axes[1].set_xlabel("BioHashing ratio m/d")
    axes[1].set_ylabel("inversion correlation")
    axes[1].set_title("Leakage vs ratio")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle=":")
    fig.suptitle(title)
    _save(fig, out_path)


def plot_key_sensitivity(
    key_sensitivity_df: pd.DataFrame,
    out_path: Path,
    title: str = "Template key sensitivity (renewability)",
) -> None:
    """Bit-error-rate and template correlation under one-bit token edits.

    A renewable template should change like a fresh random one when the token is
    perturbed: BER concentrated at the ideal 0.5 (avalanche) and near-zero
    correlation between templates from edited tokens.

    Args:
        key_sensitivity_df: Per-segment frame (``key_sensitivity.csv``) with
            ``bit_error_rate_mean`` and ``correlation_mean`` columns.
        out_path: PNG output path.
        title: Figure title.
    """
    if key_sensitivity_df is None or key_sensitivity_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(key_sensitivity_df["bit_error_rate_mean"], bins=15,
                 color="#1f77b4", alpha=0.7, edgecolor="white")
    axes[0].axvline(0.5, color="crimson", linestyle="--", label="ideal (0.5)")
    axes[0].set(title="Bit-error rate", xlabel="BER under 1-bit token edit", ylabel="count")
    axes[0].legend(fontsize=8)
    axes[1].hist(key_sensitivity_df["correlation_mean"], bins=15,
                 color="#2ca02c", alpha=0.7, edgecolor="white")
    axes[1].axvline(0.0, color="crimson", linestyle="--", label="ideal (0)")
    axes[1].set(title="Template correlation", xlabel="corr under 1-bit token edit", ylabel="count")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    _save(fig, out_path)


__all__ = [
    "ScorePools",
    "plot_classifier_comparison",
    "plot_det_curves",
    "plot_inversion_leakage",
    "plot_key_sensitivity",
    "plot_non_invertibility",
    "plot_per_subject_ablation",
    "plot_pr_curves",
    "plot_ratio_sweep",
    "plot_record_multiplicity",
    "plot_regime_summary",
    "plot_roc_curves",
    "plot_score_distribution",
    "plot_stolen_token_scores",
]
