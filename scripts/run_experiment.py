"""Focused end-to-end experiment driver for the BioHashing cancelable system.

Runs the core protocol of the signal-domain sibling project, adapted to the
feature-domain BioHashing transform:

    identification CV · verification CV · cancelability · key sensitivity ·
    inversion / leakage · DET curves.

Examples
--------
  # Identification + verification over the default regimes, clean signals:
  python scripts/run_experiment.py -v

  # Quick smoke test (8 subjects, 3 folds, one feature level):
  python scripts/run_experiment.py --max-subjects 8 --cv-folds 3 \
                                   --feature-levels 4 -v

  # Everything (cancelability, key sensitivity, inversion, DET plots):
  python scripts/run_experiment.py --all -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from mwf import (  # noqa: E402
    DEFAULT_RATIOS,
    BiometricSegments,
    DEFAULT_PROJECTION_RATIO,
    DEFAULT_SPLIT_SEEDS,
    FEATURE_WAVELET,
    KeyMode,
    METRIC_NAMES,
    benchmark,
    build_templates,
    closed_set_score_pools,
    cross_validate_classifier_multiseed,
    decidability,
    det_curve_from_scores,
    evaluate,
    evaluate_cancelability,
    extract_features,
    extract_features_batch,
    feature_dimension,
    key_sensitivity,
    load_mimic100,
    make_pipeline,
    max_feature_level,
    multimodal_leakage_metrics,
    non_invertibility_analysis,
    per_subject_ablation,
    rank_k_accuracies,
    ratio_sweep,
    record_multiplicity_leakage,
    INDEPENDENT,
    SHARED_SUBSPACE,
    run_verification_cv,
    set_global_seeds,
    stolen_token_score_pools,
    stolen_token_verification,
    summarise_run,
    temporal_holdout_per_subject,
)
from mwf.classifiers import CLASSIFIER_NAMES, build_classifier  # noqa: E402
from mwf.feature_transform import (  # noqa: E402
    transform_multimodal,
    transform_multimodal_batch,
)
from mwf.plots import (  # noqa: E402
    plot_classifier_comparison,
    plot_det_curves,
    plot_inversion_leakage,
    plot_key_sensitivity,
    plot_non_invertibility,
    plot_per_subject_ablation,
    plot_pr_curves,
    plot_ratio_sweep,
    plot_record_multiplicity,
    plot_regime_summary,
    plot_roc_curves,
    plot_score_distribution,
    plot_stolen_token_scores,
)

logger = logging.getLogger("run_experiment")

DEFAULT_REGIMES: tuple[KeyMode, ...] = (
    KeyMode.IDENTITY,
    KeyMode.SINGLE_KEY,
    KeyMode.PER_SUBJECT,
)
RANK_TARGETS: tuple[int, ...] = (1, 5, 10, 20)


def _configure_logging(verbose: bool) -> None:
    """Initialise the root logger.

    Args:
        verbose: If ``True`` log at ``INFO``; otherwise at ``WARNING``.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _subset_segments(segments: BiometricSegments, max_subjects: int | None) -> BiometricSegments:
    """Restrict ``segments`` to the first ``max_subjects`` subjects.

    Args:
        segments: Source cohort.
        max_subjects: Subject cap; ``None`` returns ``segments`` unchanged.

    Returns:
        A :class:`BiometricSegments` covering at most ``max_subjects`` subjects.
    """
    if max_subjects is None:
        return segments
    keep = np.unique(segments.labels)[:max_subjects]
    mask = np.isin(segments.labels, keep)
    return BiometricSegments(
        ecg=segments.ecg[mask],
        ppg=segments.ppg[mask],
        labels=segments.labels[mask],
        sampling_rate=segments.sampling_rate,
    )


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` as CSV, skipping empty dataframes.

    Args:
        df: Dataframe to persist.
        path: Destination path; parent directories are created if needed.
    """
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows).", path, len(df))


def _identification_and_verification(
    segments: BiometricSegments,
    *,
    regimes: tuple[KeyMode, ...],
    feature_levels: tuple[int, ...],
    projection_ratio: float,
    binarise: bool,
    n_folds: int,
    split_seeds: tuple[int, ...],
    run_identification: bool,
    run_verification: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep regimes × feature levels for identification and verification CV.

    Returns:
        Tuple ``(identification_df, verification_df)``.
    """
    id_rows: list[dict] = []
    ver_rows: list[dict] = []
    for regime in regimes:
        for level in feature_levels:
            bundle = build_templates(
                segments,
                feature_level=level,
                projection_ratio=projection_ratio,
                binarise=binarise,
                key_mode=regime,
            )
            if run_identification:
                for name in CLASSIFIER_NAMES:
                    result = cross_validate_classifier_multiseed(
                        bundle, name, build_classifier(name),
                        n_folds=n_folds, split_seeds=split_seeds, ranks=RANK_TARGETS,
                    )
                    summaries = summarise_run(result)
                    row = {
                        "regime": regime.value,
                        "feature_level": level,
                        "n_template_dims": result.n_features,
                        "classifier": name,
                        "n_folds": result.n_folds,
                    }
                    for m in METRIC_NAMES:
                        row[f"{m}_mean"] = summaries[m].mean
                        row[f"{m}_std"] = summaries[m].std
                        row[f"{m}_ci_lo"] = summaries[m].ci_low
                        row[f"{m}_ci_hi"] = summaries[m].ci_high
                    for k in RANK_TARGETS:
                        row[f"rank_{k}_accuracy_mean"] = float(
                            np.nanmean(result.per_extra_values(f"rank_{k}_accuracy"))
                        )
                    id_rows.append(row)
            if run_verification:
                for mode in ("closed_set", "open_set"):
                    ver = run_verification_cv(bundle, mode=mode, n_folds=n_folds)
                    eer = ver.eer_values()
                    deci = ver.decidability_values()
                    ver_rows.append({
                        "regime": regime.value,
                        "feature_level": level,
                        "n_template_dims": ver.n_features,
                        "verification_mode": mode,
                        "n_folds": ver.n_folds,
                        "eer_mean": float(np.nanmean(eer)),
                        "eer_std": float(np.nanstd(eer, ddof=1)) if eer.size > 1 else 0.0,
                        "decidability_mean": float(np.nanmean(deci)),
                    })
    return pd.DataFrame(id_rows), pd.DataFrame(ver_rows)


def _cancelability_df(
    segments: BiometricSegments,
    *,
    feature_levels: tuple[int, ...],
    projection_ratio: float,
    binarise: bool,
    n_keys: int,
    seed: int,
) -> pd.DataFrame:
    """Run the ISO/IEC 30136 cancelability protocol per feature level."""
    rows = []
    for level in feature_levels:
        report = evaluate_cancelability(
            segments, feature_level=level, projection_ratio=projection_ratio,
            binarise=binarise, n_keys=n_keys, seed=seed,
        )
        rows.append({"feature_level": level, **asdict(report)})
    return pd.DataFrame(rows)


def _key_sensitivity_df(
    segments: BiometricSegments,
    *,
    feature_level: int,
    projection_ratio: float,
    binarise: bool,
    n_segments: int,
    n_trials: int,
    seed: int,
) -> pd.DataFrame:
    """Bit-flip key sensitivity of the template on a random segment sample."""
    rng = np.random.default_rng(seed + 7)
    idx = rng.choice(
        segments.num_segments, size=min(n_segments, segments.num_segments), replace=False,
    )
    rows = []
    for i in idx:
        x = extract_features(segments.ecg[i], segments.ppg[i], level=feature_level)
        report = key_sensitivity(
            transform_fn=lambda tok: transform_multimodal(
                x, tok, projection_ratio=projection_ratio, binarise=binarise,
            ),
            base_password=f"KEY_SENS_{i}",
            n_trials=n_trials,
        )
        rows.append({"segment": int(i), **asdict(report)})
    return pd.DataFrame(rows)


def _inversion_df(
    segments: BiometricSegments,
    *,
    feature_level: int,
    projection_ratio: float,
    n_segments: int,
    seed: int,
) -> pd.DataFrame:
    """Leakage of the hybrid transform: linear ECG inverse vs IoM PPG best-effort."""
    rng = np.random.default_rng(seed + 11)
    idx = rng.choice(
        segments.num_segments, size=min(n_segments, segments.num_segments), replace=False,
    )
    half = feature_dimension(feature_level) // 2
    rows = []
    for i in idx:
        x = extract_features(segments.ecg[i], segments.ppg[i], level=feature_level)
        token = f"USER_INV_{i}"
        report = multimodal_leakage_metrics(
            x[:half], x[half:], token, projection_ratio=projection_ratio,
        )
        rows.append({"segment": int(i), "feature_level": feature_level, **asdict(report)})
    return pd.DataFrame(rows)


def _record_multiplicity_df(
    segments: BiometricSegments,
    *,
    feature_level: int,
    projection_ratio: float,
    n_segments: int,
    n_templates: int,
    seed: int,
) -> pd.DataFrame:
    """ARM recovery of the ECG block per probed segment, for both revocation policies.

    Contrasts the operational ``independent``-token revocation (vulnerable) against
    the hardened ``shared_subspace`` rotation, so the figure shows the leakage
    climbing to 1 for the former and staying flat for the latter.
    """
    rng = np.random.default_rng(seed + 13)
    idx = rng.choice(
        segments.num_segments, size=min(n_segments, segments.num_segments), replace=False,
    )
    half = feature_dimension(feature_level) // 2
    rows = []
    for i in idx:
        x = extract_features(segments.ecg[i], segments.ppg[i], level=feature_level)
        token = f"USER_ARM_{i}"
        for revocation in (INDEPENDENT, SHARED_SUBSPACE):
            for report in record_multiplicity_leakage(
                x[:half], token, n_templates,
                projection_ratio=projection_ratio, revocation=revocation,
            ):
                rows.append({"segment": int(i), "feature_level": feature_level, **asdict(report)})
    return pd.DataFrame(rows)


def _non_invertibility_outputs(
    segments: BiometricSegments,
    *,
    feature_level: int,
    projection_ratio: float,
    binarise: bool,
    max_victims: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Wu-style non-invertibility: aggregate report + per-pool sample frame.

    Returns:
        Tuple ``(report_df, pools_df, pools)``. ``report_df`` is a single-row
        summary with the SAR / mean-correlation numbers; ``pools_df`` stacks the
        per-sample correlations of the three populations (long format) for
        downstream analysis; ``pools`` is the in-memory dict consumed by the
        plotting helper.
    """
    report, pools = non_invertibility_analysis(
        segments,
        feature_level=feature_level,
        projection_ratio=projection_ratio,
        binarise=binarise,
        max_victims=max_victims,
        seed=seed,
    )
    report_df = pd.DataFrame([{
        "feature_level": feature_level,
        "projection_ratio": projection_ratio,
        **asdict(report),
    }])
    pool_rows: list[dict] = []
    for population, samples in pools.items():
        for value in samples:
            pool_rows.append({
                "feature_level": feature_level,
                "population": population,
                "abs_correlation": float(value),
            })
    pools_df = pd.DataFrame(pool_rows)
    return report_df, pools_df, pools


def _per_subject_ablation_df(
    segments: BiometricSegments,
    *,
    feature_level: int,
    projection_ratio: float,
    binarise: bool,
    seed: int,
    group_sizes: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Key-vs-biometric ablation: EER as group size (subjects per token) grows."""
    points = per_subject_ablation(
        segments,
        feature_level=feature_level,
        projection_ratio=projection_ratio,
        binarise=binarise,
        group_sizes=group_sizes,
        seed=seed,
    )
    return pd.DataFrame([
        {"feature_level": feature_level, **asdict(p)} for p in points
    ])


def _ratio_sweep_df(
    segments: BiometricSegments,
    *,
    feature_level: int,
    binarise: bool,
    ratios: tuple[float, ...],
    seed: int,
    max_victims: int | None = 40,
    n_inversion_segments: int = 16,
) -> pd.DataFrame:
    """Recognition-leakage trade-off across ``ratios`` at one feature level."""
    points = ratio_sweep(
        segments,
        feature_level=feature_level,
        ratios=ratios,
        binarise=binarise,
        max_victims=max_victims,
        n_inversion_segments=n_inversion_segments,
        seed=seed,
    )
    return pd.DataFrame([
        {"feature_level": feature_level, **asdict(p)} for p in points
    ])


def _stolen_token_df(
    segments: BiometricSegments,
    *,
    feature_levels: tuple[int, ...],
    projection_ratio: float,
    binarise: bool,
    seed: int,
) -> pd.DataFrame:
    """Worst-case (stolen-key) EER per feature level and score-normalisation."""
    rows = []
    for level in feature_levels:
        for score_norm in (None, "znorm"):
            r = stolen_token_verification(
                segments, feature_level=level, projection_ratio=projection_ratio,
                binarise=binarise, seed=seed, score_norm=score_norm,
            )
            rows.append({
                "feature_level": level,
                "score_norm": score_norm or "raw",
                "n_victims": r.n_victims,
                "n_genuine": r.n_genuine,
                "n_impostor": r.n_impostor,
                "eer": r.eer,
                "decidability": r.decidability,
                "genuine_mean": r.genuine_mean,
                "impostor_mean": r.impostor_mean,
            })
    return pd.DataFrame(rows)


def _timing_df(
    segments: BiometricSegments,
    *,
    feature_level: int,
    projection_ratio: float,
    binarise: bool,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Benchmark the extraction and BioHashing-projection stages."""
    sample_n = min(8, segments.num_segments)
    ecg, ppg = segments.ecg[:sample_n], segments.ppg[:sample_n]
    extract_bench = benchmark(
        "extract_features_batch",
        lambda: extract_features_batch(ecg, ppg, level=feature_level),
        repeats=n_repeats, inner_loops=sample_n,
    )
    feats = extract_features_batch(ecg, ppg, level=feature_level)
    tokens = [f"BENCH_{k}" for k in range(sample_n)]
    project_bench = benchmark(
        "transform_multimodal_batch",
        lambda: transform_multimodal_batch(
            feats, tokens, projection_ratio=projection_ratio, binarise=binarise,
        ),
        repeats=n_repeats, inner_loops=sample_n,
    )
    return pd.DataFrame([
        {
            "stage": r.label,
            "per_call_ms": r.per_call_ms(),
            "mean_s": r.mean_s,
            "std_s": r.std_s,
            "ci_low_s": r.ci_low,
            "ci_high_s": r.ci_high,
            "bytes_output": r.bytes_output,
        }
        for r in (extract_bench, project_bench)
    ])


def _holdout_df(
    segments: BiometricSegments,
    *,
    regimes: tuple[KeyMode, ...],
    feature_levels: tuple[int, ...],
    projection_ratio: float,
    binarise: bool,
    test_fraction: float,
) -> pd.DataFrame:
    """Sealed temporal-holdout evaluation: fit on train, score the held-out tail."""
    split = temporal_holdout_per_subject(segments, test_fraction=test_fraction)
    rows = []
    for regime in regimes:
        for level in feature_levels:
            train_b = build_templates(
                split.train, feature_level=level, projection_ratio=projection_ratio,
                binarise=binarise, key_mode=regime,
            )
            test_b = build_templates(
                split.test, feature_level=level, projection_ratio=projection_ratio,
                binarise=binarise, key_mode=regime,
            )
            for name in CLASSIFIER_NAMES:
                pipe = make_pipeline(clone(build_classifier(name)))
                pipe.fit(train_b.features, train_b.labels)
                y_score = np.asarray(pipe.predict_proba(test_b.features), dtype=np.float64)
                classes = np.asarray(pipe.named_steps["clf"].classes_, dtype=np.int64)
                y_pred = pipe.predict(test_b.features)
                metrics = evaluate(test_b.labels, y_pred, y_score, classes)
                ranks = rank_k_accuracies(test_b.labels, y_score, classes)
                rows.append({
                    "regime": regime.value,
                    "feature_level": level,
                    "classifier": name,
                    "n_train": int(train_b.features.shape[0]),
                    "n_test": int(test_b.features.shape[0]),
                    **metrics.as_dict(),
                    **ranks,
                })
    return pd.DataFrame(rows)


def _plot_figures(
    segments: BiometricSegments,
    id_df: pd.DataFrame,
    *,
    regimes: tuple[KeyMode, ...],
    feature_level: int,
    projection_ratio: float,
    binarise: bool,
    n_folds: int,
    fig_dir: Path,
    inversion_df: pd.DataFrame | None = None,
    key_sensitivity_df: pd.DataFrame | None = None,
    record_multiplicity_df: pd.DataFrame | None = None,
    non_invertibility_pools: dict[str, np.ndarray] | None = None,
    non_invertibility_report: dict | None = None,
    per_subject_ablation_df: pd.DataFrame | None = None,
    ratio_sweep_df: pd.DataFrame | None = None,
    plot_stolen_token: bool = False,
    seed: int = 42,
) -> None:
    """Render the full figure suite (curves, score KDEs, metric + cancelability).

    The closed-set genuine/impostor pools are computed once per regime (at
    ``feature_level``) and reused across the DET/ROC/PR overlays and the
    per-regime score KDEs. The EER/AUC and classifier-comparison summaries are
    read from the identification-metrics frame ``id_df``. The cancelability
    figures (IoM inversion leakage, key sensitivity, stolen-token scores) are
    rendered when their source data is available.

    Args:
        segments: Source cohort for the verification score pools.
        id_df: Identification-metrics frame (``metrics.csv``); may be empty.
        regimes: Regimes to overlay / summarise.
        feature_level: Level used for the curve/KDE figures.
        projection_ratio: ECG BioHashing ratio.
        binarise: Whether to sign-binarise the ECG block.
        n_folds: CV folds for the closed-set score pools.
        fig_dir: Output directory for the PNGs.
        inversion_df: Per-segment inversion frame for the leakage figure.
        key_sensitivity_df: Per-segment key-sensitivity frame for that figure.
        record_multiplicity_df: Per-(segment, n_templates) ARM frame for the
            record-multiplicity figure.
        plot_stolen_token: If ``True``, compute the stolen-token pools and plot
            their genuine/impostor distributions (the worst-case figure).
        seed: Seed forwarded to the stolen-token pooling.
    """
    pools: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for regime in regimes:
        bundle = build_templates(
            segments, feature_level=feature_level,
            projection_ratio=projection_ratio, binarise=binarise, key_mode=regime,
        )
        pools[regime.value] = closed_set_score_pools(bundle, n_folds=n_folds)

    suffix = f"feat_lvl={feature_level}"
    plot_det_curves(pools, fig_dir / "det.png", title=f"DET — {suffix}")
    plot_roc_curves(pools, fig_dir / "roc.png", title=f"ROC — {suffix}")
    plot_pr_curves(pools, fig_dir / "pr.png", title=f"Precision–Recall — {suffix}")
    for regime_label, (genuine, impostor) in pools.items():
        plot_score_distribution(
            genuine, impostor, fig_dir / f"scores_{regime_label}.png",
            title=f"Score distribution — {regime_label} | {suffix}",
        )

    if not id_df.empty:
        plot_regime_summary(id_df, fig_dir / "regime_summary.png")
        for regime in regimes:
            plot_classifier_comparison(
                id_df, fig_dir / f"clf_vs_features_{regime.value}.png", regime.value,
            )

    # --- cancelability story (IoM-specific) ---
    if inversion_df is not None and not inversion_df.empty:
        plot_inversion_leakage(inversion_df, fig_dir / "inversion_leakage.png")
    if key_sensitivity_df is not None and not key_sensitivity_df.empty:
        plot_key_sensitivity(key_sensitivity_df, fig_dir / "key_sensitivity.png")
    if record_multiplicity_df is not None and not record_multiplicity_df.empty:
        plot_record_multiplicity(record_multiplicity_df, fig_dir / "record_multiplicity.png")
    if non_invertibility_pools:
        sar_i = (
            float(non_invertibility_report["sar_type1"])
            if non_invertibility_report is not None else None
        )
        sar_ii = (
            float(non_invertibility_report["sar_type2"])
            if non_invertibility_report is not None else None
        )
        plot_non_invertibility(
            non_invertibility_pools, fig_dir / "non_invertibility.png",
            sar_type1=sar_i, sar_type2=sar_ii,
        )
    if per_subject_ablation_df is not None and not per_subject_ablation_df.empty:
        plot_per_subject_ablation(
            per_subject_ablation_df, fig_dir / "per_subject_ablation.png",
        )
    if ratio_sweep_df is not None and not ratio_sweep_df.empty:
        plot_ratio_sweep(ratio_sweep_df, fig_dir / "ratio_sweep.png")
    if plot_stolen_token:
        # Cap victims so the illustrative figure stays fast on large cohorts.
        genuine, impostor = stolen_token_score_pools(
            segments, feature_level=feature_level, projection_ratio=projection_ratio,
            binarise=binarise, max_victims=25, seed=seed,
        )
        det = det_curve_from_scores(genuine, impostor)
        eer = float(det.fnmr[int(np.argmin(np.abs(det.fmr - det.fnmr)))])
        plot_stolen_token_scores(
            genuine, impostor, fig_dir / "stolen_token_scores.png",
            eer=eer, decidability=decidability(genuine, impostor),
        )
    logger.info("Wrote figure suite to %s.", fig_dir)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments for the experiment driver."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=_PKG_ROOT / "results")
    parser.add_argument(
        "--feature-levels", type=int, nargs="+", default=None,
        help="DWT depths to sweep. Default: 1 .. the deepest level usable for "
             f"the feature wavelet ({FEATURE_WAVELET!r}) and the segment length.",
    )
    parser.add_argument("--projection-ratio", type=float, default=DEFAULT_PROJECTION_RATIO)
    parser.add_argument("--binarise", action="store_true", help="Use ±1 BioCodes.")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(DEFAULT_SPLIT_SEEDS))
    parser.add_argument(
        "--regimes", type=str, nargs="+",
        default=[r.value for r in DEFAULT_REGIMES], choices=[r.value for r in KeyMode],
    )
    parser.add_argument(
        "--protocol", type=str, default="both",
        choices=("identification", "verification", "both"),
    )
    parser.add_argument("--cancelability-keys", type=int, default=0,
                        help="K random keys for the Gomez-Barrero protocol (≥ 2 to enable).")
    parser.add_argument("--key-sensitivity", action="store_true")
    parser.add_argument("--inversion", action="store_true")
    parser.add_argument("--record-multiplicity", action="store_true",
                        help="ARM: recover the ECG block from N revoked templates.")
    parser.add_argument("--arm-templates", type=int, default=4,
                        help="Max revoked templates the ARM adversary stacks.")
    parser.add_argument("--non-invertibility", action="store_true",
                        help="Wu-style 3-distribution + SAR non-invertibility report.")
    parser.add_argument("--non-invertibility-victims", type=int, default=50,
                        help="Max subjects exercised as Wu-style reconstruction targets.")
    parser.add_argument("--per-subject-ablation", action="store_true",
                        help="Sweep how many subjects share a token (key vs biometric).")
    parser.add_argument("--ratio-sweep", action="store_true",
                        help="Sweep BioHashing ratio m/d (recognition vs leakage).")
    parser.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS),
                        help="Ratios m/d to sweep when --ratio-sweep is enabled.")
    parser.add_argument("--stolen-token", action="store_true",
                        help="Worst-case (stolen-key) EER — the honest biometric figure of merit.")
    parser.add_argument("--timing", action="store_true",
                        help="Per-stage computational-cost benchmark with bootstrap CI.")
    parser.add_argument("--holdout", action="store_true",
                        help="Sealed temporal-holdout evaluation (fit on train, score the tail).")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--det-plots", action="store_true",
                        help="Render the figure suite: DET/ROC/PR curves, per-regime "
                             "score KDEs, regime summary and classifier comparison.")
    parser.add_argument("--all", action="store_true", help="Enable every optional analysis.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    if args.all:
        args.key_sensitivity = True
        args.inversion = True
        args.record_multiplicity = True
        args.non_invertibility = True
        args.per_subject_ablation = True
        args.ratio_sweep = True
        args.stolen_token = True
        args.timing = True
        args.holdout = True
        args.det_plots = True
        if args.cancelability_keys < 2:
            args.cancelability_keys = 16
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the configured experiment blocks and persist their outputs."""
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    set_global_seeds(args.seed)

    segments = _subset_segments(load_mimic100(), args.max_subjects)
    regimes = tuple(KeyMode(r) for r in args.regimes)
    if args.feature_levels:
        feature_levels = tuple(args.feature_levels)
    else:
        # Sweep every usable DWT depth: 1 .. max for the feature wavelet at this
        # segment length (e.g. 1..6 for bior3.3 on MIMIC-100's 750-sample segments).
        max_level = max_feature_level(segments.segment_length, FEATURE_WAVELET)
        feature_levels = tuple(range(1, max_level + 1))
    # Single-level analyses (key sensitivity, inversion, timing, holdout, DET)
    # report on the deepest level — the richest template and the most demanding
    # case for the security/cost figures.
    representative_level = max(feature_levels)
    logger.info(
        "Feature levels: %s (representative single-level=%d).",
        feature_levels, representative_level,
    )
    out = args.output_dir

    id_df, ver_df = _identification_and_verification(
        segments,
        regimes=regimes,
        feature_levels=feature_levels,
        projection_ratio=args.projection_ratio,
        binarise=args.binarise,
        n_folds=args.cv_folds,
        split_seeds=tuple(args.split_seeds),
        run_identification=args.protocol in ("identification", "both"),
        run_verification=args.protocol in ("verification", "both"),
    )
    _save_csv(id_df, out / "metrics.csv")
    _save_csv(ver_df, out / "verification.csv")

    if args.cancelability_keys >= 2:
        _save_csv(
            _cancelability_df(
                segments, feature_levels=feature_levels,
                projection_ratio=args.projection_ratio, binarise=args.binarise,
                n_keys=args.cancelability_keys, seed=args.seed,
            ),
            out / "cancelability.csv",
        )
    ks_df: pd.DataFrame | None = None
    if args.key_sensitivity:
        ks_df = _key_sensitivity_df(
            segments, feature_level=representative_level,
            projection_ratio=args.projection_ratio, binarise=args.binarise,
            n_segments=16, n_trials=24, seed=args.seed,
        )
        _save_csv(ks_df, out / "key_sensitivity.csv")
    inv_df: pd.DataFrame | None = None
    if args.inversion:
        inv_df = _inversion_df(
            segments, feature_level=representative_level,
            projection_ratio=args.projection_ratio, n_segments=16, seed=args.seed,
        )
        _save_csv(inv_df, out / "inversion.csv")
    arm_df: pd.DataFrame | None = None
    if args.record_multiplicity:
        arm_df = _record_multiplicity_df(
            segments, feature_level=representative_level,
            projection_ratio=args.projection_ratio, n_segments=16,
            n_templates=args.arm_templates, seed=args.seed,
        )
        _save_csv(arm_df, out / "arm.csv")
    ni_pools: dict[str, np.ndarray] | None = None
    ni_report: dict | None = None
    if args.non_invertibility:
        ni_report_df, ni_pools_df, ni_pools = _non_invertibility_outputs(
            segments, feature_level=representative_level,
            projection_ratio=args.projection_ratio, binarise=args.binarise,
            max_victims=args.non_invertibility_victims, seed=args.seed,
        )
        _save_csv(ni_report_df, out / "non_invertibility.csv")
        _save_csv(ni_pools_df, out / "non_invertibility_pools.csv")
        if not ni_report_df.empty:
            ni_report = ni_report_df.iloc[0].to_dict()
    psa_df: pd.DataFrame | None = None
    if args.per_subject_ablation:
        psa_df = _per_subject_ablation_df(
            segments, feature_level=representative_level,
            projection_ratio=args.projection_ratio, binarise=args.binarise,
            seed=args.seed,
        )
        _save_csv(psa_df, out / "per_subject_ablation.csv")
    rs_df: pd.DataFrame | None = None
    if args.ratio_sweep:
        rs_df = _ratio_sweep_df(
            segments, feature_level=representative_level,
            binarise=args.binarise, ratios=tuple(args.ratios), seed=args.seed,
        )
        _save_csv(rs_df, out / "ratio_sweep.csv")
    if args.stolen_token:
        _save_csv(
            _stolen_token_df(
                segments, feature_levels=feature_levels,
                projection_ratio=args.projection_ratio, binarise=args.binarise,
                seed=args.seed,
            ),
            out / "stolen_token.csv",
        )
    if args.timing:
        _save_csv(
            _timing_df(
                segments, feature_level=representative_level,
                projection_ratio=args.projection_ratio, binarise=args.binarise,
            ),
            out / "timing.csv",
        )
    if args.holdout:
        _save_csv(
            _holdout_df(
                segments, regimes=regimes, feature_levels=(representative_level,),
                projection_ratio=args.projection_ratio, binarise=args.binarise,
                test_fraction=args.holdout_fraction,
            ),
            out / "holdout.csv",
        )
    if args.det_plots:
        _plot_figures(
            segments, id_df, regimes=regimes, feature_level=representative_level,
            projection_ratio=args.projection_ratio, binarise=args.binarise,
            n_folds=args.cv_folds, fig_dir=out / "figures",
            inversion_df=inv_df, key_sensitivity_df=ks_df,
            record_multiplicity_df=arm_df,
            non_invertibility_pools=ni_pools,
            non_invertibility_report=ni_report,
            per_subject_ablation_df=psa_df,
            ratio_sweep_df=rs_df,
            plot_stolen_token=args.stolen_token, seed=args.seed,
        )
    logger.info("Done. Outputs under %s", out)


if __name__ == "__main__":
    main()
