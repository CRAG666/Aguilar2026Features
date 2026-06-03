"""Focused end-to-end experiment driver for the BioHashing cancelable system.

Runs the core protocol of the signal-domain sibling project, adapted to the
feature-domain BioHashing transform:

    identification CV · verification CV · cancelability · non-invertibility ·
    stolen-token worst case · cross-activity · DET curves.

Runs over one or more cohorts (MIMIC-100, BIDMC, PTT-PPG). Each dataset's
outputs land in `results/<Name>_<date>_<hour>/`; a cross-dataset headline
comparison lands in `results/shared/`.

Examples
--------
  # Identification + verification over every dataset, clean signals:
  python scripts/run_experiment.py -v

  # Everything, every dataset (the full Q1 battery + cross-activity + figures):
  python scripts/run_experiment.py --all -v

  # Restrict to one dataset:
  python scripts/run_experiment.py --datasets bidmc --all -v

  # Quick smoke test (8 subjects, 3 folds, one feature level, MIMIC only):
  python scripts/run_experiment.py --datasets mimic --max-subjects 8 \
                                   --cv-folds 3 --feature-levels 4 -v
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.base import clone

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from mwf import (  # noqa: E402
    BiometricSegments,
    DEFAULT_PROJECTION_RATIO,
    DEFAULT_SPLIT_SEEDS,
    FEATURE_WAVELET,
    KeyMode,
    METRIC_NAMES,
    build_templates,
    class_score_matrix,
    closed_set_score_pools,
    compare_classifiers,
    cross_session_verification,
    cross_validate_tasks,
    CrossValidationResult,
    CVTask,
    decidability,
    det_curve_from_scores,
    evaluate,
    evaluate_cancelability,
    load_bidmc,
    load_mimic100,
    load_ptt_ppg,
    make_pipeline,
    max_feature_level,
    nadeau_bengio_ci_mean,
    non_invertibility_analysis,
    rank_k_accuracies,
    run_verification_cv,
    set_global_seeds,
    stolen_token_score_pools,
    stolen_token_verification,
    subject_holdout,
    summarise_run,
    temporal_holdout_per_subject,
    VerificationMode,
)
from mwf.batch_utils import parallel_map  # noqa: E402
from mwf.classifiers import (  # noqa: E402
    CLASSIFIER_NAMES,
    build_classifier,
    build_param_grid,
)
from mwf.plots import (  # noqa: E402
    plot_classifier_comparison,
    plot_det_curves,
    plot_non_invertibility,
    plot_pr_curves,
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
VERIFICATION_MODES: tuple[VerificationMode, ...] = ("closed_set", "open_set")


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One evaluable cohort: how to load it and its cross-activity structure.

    Attributes:
        key: CLI selector for ``--datasets``.
        name: Output-folder / report display name.
        load: Zero-arg loader returning the cohort's ``BiometricSegments`` (the
            enrolment condition for multi-activity datasets).
        enrol_activity: Enrolment condition name for multi-activity datasets;
            ``None`` for single-recording cohorts (no cross-activity protocol).
        probe_activities: Probe conditions verified against ``enrol_activity``.
        load_activity: Loader for one named activity (cross-activity only).
    """

    key: str
    name: str
    load: Callable[[], BiometricSegments]
    enrol_activity: str | None = None
    probe_activities: tuple[str, ...] = ()
    load_activity: Callable[[str], BiometricSegments] | None = None


# MIMIC-100 and BIDMC are single-recording cohorts at 125 Hz; PTT-PPG records
# sit/walk/run separately at 500 Hz, enabling the cross-activity protocol.
DATASET_SPECS: dict[str, DatasetSpec] = {
    "mimic": DatasetSpec("mimic", "MIMIC-100", load_mimic100),
    "bidmc": DatasetSpec("bidmc", "BIDMC", load_bidmc),
    "ptt": DatasetSpec(
        "ptt", "PTT-PPG", load=lambda: load_ptt_ppg("sit"),
        enrol_activity="sit", probe_activities=("walk", "run"),
        load_activity=load_ptt_ppg,
    ),
}


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


def _git_sha() -> str | None:
    """Return the current commit SHA, or ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PKG_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _file_sha256(path: Path) -> str | None:
    """Return the SHA-256 of ``path``, or ``None`` when it does not exist."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(out: Path, args: argparse.Namespace, *, n_segments: int, n_subjects: int) -> None:
    """Persist a provenance manifest so every results run is reproducible.

    Records the git commit, full CLI args, dataset shape, key dependency versions
    and the lockfile hash next to the CSVs — without this a ``metrics.csv`` cannot
    be tied back to the exact code, environment and configuration that produced it.
    """
    deps: dict[str, str | None] = {}
    for pkg in ("numpy", "scipy", "scikit-learn", "statsmodels", "pyeer", "neurokit2"):
        try:
            deps[pkg] = version(pkg)
        except PackageNotFoundError:
            deps[pkg] = None
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python": sys.version,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "dataset": {"n_segments": n_segments, "n_subjects": n_subjects},
        "dependencies": deps,
        "uv_lock_sha256": _file_sha256(_PKG_ROOT / "uv.lock"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote provenance manifest to %s.", out / "run_manifest.json")


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


def _identification_metric_row(
    regime_value: str,
    feature_level: int,
    classifier_name: str,
    result,
    *,
    tuned: bool,
) -> dict:
    """Flatten one classifier's CV result into a ``metrics.csv`` row.

    Keeps the (wide) identification schema in one place, separate from the
    orchestration that produces the results.
    """
    summaries = summarise_run(result)
    row: dict = {
        "regime": regime_value,
        "feature_level": feature_level,
        "n_template_dims": result.n_features,
        "classifier": classifier_name,
        "n_folds": result.n_folds,
        "tuned": tuned,
    }
    for m in METRIC_NAMES:
        s = summaries[m]
        row[f"{m}_mean"], row[f"{m}_std"] = s.mean, s.std
        row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = s.ci_low, s.ci_high
    for k in RANK_TARGETS:
        row[f"rank_{k}_accuracy_mean"] = float(
            np.nanmean(result.per_extra_values(f"rank_{k}_accuracy"))
        )
    return row


def _verification_metric_row(
    regime_value: str,
    feature_level: int,
    mode: str,
    ver,
    *,
    n_folds: int,
) -> dict:
    """Flatten one verification CV result into a ``verification.csv`` row."""
    eer = ver.eer_values()
    deci = ver.decidability_values()
    finite_eer = eer[np.isfinite(eer)]
    # Verification EER gets the same Nadeau-Bengio corrected CI as identification,
    # so the two protocols are reported at equal rigour.
    eer_ci_lo, eer_ci_hi = (
        nadeau_bengio_ci_mean(finite_eer, n_folds=n_folds)
        if finite_eer.size > 1 else (float("nan"), float("nan"))
    )
    return {
        "regime": regime_value,
        "feature_level": feature_level,
        "n_template_dims": ver.n_features,
        "verification_mode": mode,
        "n_folds": ver.n_folds,
        "eer_mean": float(np.nanmean(eer)),
        "eer_std": float(np.nanstd(eer, ddof=1)) if eer.size > 1 else 0.0,
        "eer_ci_lo": eer_ci_lo,
        "eer_ci_hi": eer_ci_hi,
        "decidability_mean": float(np.nanmean(deci)),
        "decidability_std": float(np.nanstd(deci, ddof=1)) if deci.size > 1 else 0.0,
    }


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
    tune: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, CrossValidationResult]]]:
    """Sweep regimes × feature levels for identification and verification CV.

    Args:
        tune: When ``True``, each classifier is tuned with a group-aware inner CV
            on every outer fold's train slice (nested CV) — removing the
            selection-on-the-evaluation-data bias of fixed hyperparameters.

    Returns:
        Tuple ``(identification_df, verification_df, id_results)``. ``id_results``
        maps ``"{regime}@L{level}"`` → ``{classifier: CrossValidationResult}`` so
        the paired significance tests can compare classifiers on identical folds.

    Parallelism: the templates for every ``(regime, level)`` are built once and
    reused. The whole identification grid (regime × level × classifier) is then
    cross-validated through a single flat worker pool (:func:`cross_validate_tasks`)
    so cores stay busy until the last fold, and the independent verification CVs
    fan out across cores too. Row order, fold splits and aggregates are identical
    to the previous sequential nested loops.
    """
    id_rows: list[dict] = []
    ver_rows: list[dict] = []
    id_results: dict[str, dict[str, CrossValidationResult]] = {}

    # Build the templates for each (regime, level) once; reused by both protocols
    # (and shared read-only across the worker pool via joblib's memmap).
    bundles = {
        (regime, level): build_templates(
            segments, feature_level=level, projection_ratio=projection_ratio,
            binarise=binarise, key_mode=regime,
        )
        for regime in regimes
        for level in feature_levels
    }

    if run_identification:
        # One task per (regime, level, classifier); a single pool covers them all.
        tasks = [
            CVTask(
                key=(regime, level, name),
                bundle=bundles[(regime, level)],
                classifier_name=name,
                classifier=build_classifier(name),
                param_grid=build_param_grid(name) if tune else None,
            )
            for regime in regimes
            for level in feature_levels
            for name in CLASSIFIER_NAMES
        ]
        results = cross_validate_tasks(
            tasks, n_folds=n_folds, split_seeds=split_seeds, ranks=RANK_TARGETS,
        )
        for task, result in zip(tasks, results):
            regime, level, name = cast(tuple[KeyMode, int, str], task.key)
            id_results.setdefault(f"{regime.value}@L{level}", {})[name] = result
            id_rows.append(
                _identification_metric_row(regime.value, level, name, result, tuned=tune)
            )

    if run_verification:
        # Each (regime, level, mode) verification CV is independent → fan out.
        # The bundle travels inside the work item (not a closure) so joblib only
        # ships each template matrix to the worker that needs it.
        ver_keys = [
            (regime, level, mode)
            for regime in regimes
            for level in feature_levels
            for mode in VERIFICATION_MODES
        ]
        ver_items = [(bundles[(regime, level)], mode) for regime, level, mode in ver_keys]
        vers = parallel_map(
            ver_items,
            lambda item: run_verification_cv(item[0], mode=item[1], n_folds=n_folds),
        )
        ver_rows = [
            _verification_metric_row(regime.value, level, mode, ver, n_folds=n_folds)
            for (regime, level, mode), ver in zip(ver_keys, vers)
        ]

    return pd.DataFrame(id_rows), pd.DataFrame(ver_rows), id_results


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
                "eer_ci_lo": r.eer_ci_low,
                "eer_ci_hi": r.eer_ci_high,
                "decidability": r.decidability,
                "genuine_mean": r.genuine_mean,
                "impostor_mean": r.impostor_mean,
            })
    return pd.DataFrame(rows)


def _holdout_df(
    segments: BiometricSegments,
    *,
    regimes: tuple[KeyMode, ...],
    feature_levels: tuple[int, ...],
    projection_ratio: float,
    binarise: bool,
    test_fraction: float,
) -> pd.DataFrame:
    """Temporal-holdout evaluation: fit on train, score the held-out tail.

    Every subject appears in both halves (this is a within-subject temporal
    split, not an unseen-subject test — for that see :func:`_subject_holdout_df`).
    The pre-projection standardiser is fitted on the **train** split only and
    reused on the test split, so the held-out templates never see their own
    statistics (leakage-free).
    """
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
                binarise=binarise, key_mode=regime, scaler=train_b.scaler,
            )
            for name in CLASSIFIER_NAMES:
                pipe = make_pipeline(clone(build_classifier(name)))
                pipe.fit(train_b.features, train_b.labels)
                y_score = class_score_matrix(pipe, test_b.features)
                classes = np.asarray(pipe.named_steps["clf"].classes_, dtype=np.int64)
                y_pred = np.asarray(pipe.predict(test_b.features), dtype=np.int64)
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


def _subject_holdout_df(
    segments: BiometricSegments,
    *,
    regimes: tuple[KeyMode, ...],
    feature_levels: tuple[int, ...],
    projection_ratio: float,
    binarise: bool,
    test_fraction: float,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Quasi-external (subject-disjoint) verification on unseen subjects.

    The test cohort shares **no subject** with the train cohort, so this is the
    non-random, unseen-subject generalisation evidence a Q1 reviewer expects on
    top of internal CV. Only the 1:1 verification protocol is reported: closed-set
    identification on unseen subjects is undefined (their classes never appear in
    any training set), so reporting it would be meaningless.
    """
    split = subject_holdout(segments, test_fraction=test_fraction, seed=seed)
    n_test_subjects = int(np.unique(split.test.labels).size)
    # Verification CV partitions by group, so it needs at least as many test
    # subjects as folds. Shrink the fold count to fit a small held-out cohort and
    # skip entirely if even two subjects are unavailable (no impostor pairs).
    eff_folds = min(n_folds, n_test_subjects)
    if n_test_subjects < 2 or eff_folds < 2:
        logger.warning(
            "Subject-holdout skipped: only %d held-out subject(s) — need ≥ 2 for "
            "verification (raise --holdout-fraction or use more subjects).",
            n_test_subjects,
        )
        return pd.DataFrame()
    rows = []
    for regime in regimes:
        for level in feature_levels:
            test_b = build_templates(
                split.test, feature_level=level, projection_ratio=projection_ratio,
                binarise=binarise, key_mode=regime,
            )
            for mode in ("closed_set", "open_set"):
                try:
                    ver = run_verification_cv(test_b, mode=mode, n_folds=eff_folds)
                except ValueError as exc:  # degenerate split (too few groups/blocks)
                    logger.warning(
                        "Subject-holdout %s verification skipped (%s).", mode, exc,
                    )
                    continue
                eer = ver.eer_values()
                deci = ver.decidability_values()
                finite_eer = eer[np.isfinite(eer)]
                eer_ci_lo, eer_ci_hi = nadeau_bengio_ci_mean(
                    finite_eer, n_folds=eff_folds,
                ) if finite_eer.size > 1 else (float("nan"), float("nan"))
                rows.append({
                    "regime": regime.value,
                    "feature_level": level,
                    "verification_mode": mode,
                    "n_test_subjects": n_test_subjects,
                    "n_folds": ver.n_folds,
                    "eer_mean": float(np.nanmean(eer)),
                    "eer_std": float(np.nanstd(eer, ddof=1)) if eer.size > 1 else 0.0,
                    "eer_ci_lo": eer_ci_lo,
                    "eer_ci_hi": eer_ci_hi,
                    "decidability_mean": float(np.nanmean(deci)),
                })
    return pd.DataFrame(rows)


def _cross_activity_df(
    spec: DatasetSpec,
    enrol_segments: BiometricSegments,
    *,
    max_subjects: int | None,
    projection_ratio: float,
    binarise: bool,
    seed: int,
) -> pd.DataFrame:
    """Verify each activity's probes against the enrol-activity centroid.

    One row per probe activity plus a within-condition reference; empty for
    single-recording cohorts. Sibling activities are loaded on demand.
    """
    if spec.enrol_activity is None or spec.load_activity is None:
        return pd.DataFrame()
    rows = []
    for activity in (spec.enrol_activity, *spec.probe_activities):
        within = activity == spec.enrol_activity
        probe_segments = (
            enrol_segments if within
            else _subset_segments(spec.load_activity(activity), max_subjects)
        )
        r = cross_session_verification(
            enrol_segments, probe_segments,
            enrol_label=spec.enrol_activity, probe_label=activity,
            projection_ratio=projection_ratio, binarise=binarise,
            score_norm="znorm", seed=seed,
        )
        rows.append({
            "enrol": r.enrol_label, "probe": r.probe_label, "within_condition": within,
            "n_subjects": r.n_subjects, "n_genuine": r.n_genuine, "n_impostor": r.n_impostor,
            "eer": r.eer, "eer_ci_lo": r.eer_ci_low, "eer_ci_hi": r.eer_ci_high,
            "decidability": r.decidability,
            "genuine_mean": r.genuine_mean, "impostor_mean": r.impostor_mean,
        })
    return pd.DataFrame(rows)


def _significance_df(
    id_results: dict[str, dict[str, CrossValidationResult]], *, n_folds: int,
) -> pd.DataFrame:
    """Pairwise classifier significance with Benjamini-Hochberg FDR correction.

    Compares classifiers on identical folds with the Nadeau-Bengio corrected
    paired t-test, then controls the false-discovery rate over the whole
    (regime × level × metric × pair) family.
    """
    comparisons = compare_classifiers(id_results, METRIC_NAMES, n_folds=n_folds)
    return pd.DataFrame([asdict(c) for c in comparisons])


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
    non_invertibility_pools: dict[str, np.ndarray] | None = None,
    non_invertibility_report: dict | None = None,
    plot_stolen_token: bool = False,
    seed: int = 42,
) -> None:
    """Render the full figure suite (curves, score KDEs, metric + cancelability).

    The closed-set genuine/impostor pools are computed once per regime (at
    ``feature_level``) and reused across the DET/ROC/PR overlays and the
    per-regime score KDEs. The EER/AUC and classifier-comparison summaries are
    read from the identification-metrics frame ``id_df``. The cancelability
    figures (non-invertibility, stolen-token scores) are rendered when their
    source data is available.

    Args:
        segments: Source cohort for the verification score pools.
        id_df: Identification-metrics frame (``metrics.csv``); may be empty.
        regimes: Regimes to overlay / summarise.
        feature_level: Level used for the curve/KDE figures.
        projection_ratio: ECG BioHashing ratio.
        binarise: Whether to sign-binarise the ECG block.
        n_folds: CV folds for the closed-set score pools.
        fig_dir: Output directory for the PNGs.
        non_invertibility_pools: Wu-style correlation pools for that figure.
        non_invertibility_report: Single-row non-invertibility report (for SAR
            annotations on the figure).
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
        "--datasets", type=str, nargs="+", default=["all"],
        choices=[*DATASET_SPECS, "all"],
        help="Datasets to evaluate (default: all). E.g. `--datasets bidmc ptt`. "
             "Each writes to `<output-dir>/<Name>_<date>_<hour>/`.",
    )
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
        "--tune", action="store_true",
        help="Tune classifier hyperparameters with a group-aware inner CV on each "
             "outer fold (nested CV). Without it the fixed reference configs are used.",
    )
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
    parser.add_argument("--non-invertibility", action="store_true",
                        help="Wu-style 3-distribution + SAR non-invertibility report.")
    parser.add_argument("--non-invertibility-victims", type=int, default=50,
                        help="Max subjects exercised as Wu-style reconstruction targets.")
    parser.add_argument("--stolen-token", action="store_true",
                        help="Worst-case (stolen-key) EER — the honest biometric figure of merit.")
    parser.add_argument("--cross-activity", action="store_true",
                        help="Cross-activity verification for multi-activity datasets "
                             "(PTT-PPG: enrol sit, probe walk/run) — cross-condition robustness.")
    parser.add_argument("--holdout", action="store_true",
                        help="Within-subject temporal-holdout (fit on train, score the tail; "
                             "scaler fitted on train only).")
    parser.add_argument("--subject-holdout", action="store_true",
                        help="Subject-disjoint (unseen-subject) verification holdout — the "
                             "non-random external-generalisation evidence.")
    parser.add_argument("--significance", action="store_true",
                        help="Pairwise classifier significance tests (Nadeau-Bengio paired t "
                             "+ Benjamini-Hochberg FDR correction) over the identification grid.")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--det-plots", action="store_true",
                        help="Render the figure suite: DET/ROC/PR curves, per-regime "
                             "score KDEs, regime summary and classifier comparison.")
    parser.add_argument("--all", action="store_true", help="Enable every optional analysis.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    if args.all:
        args.non_invertibility = True
        args.stolen_token = True
        args.cross_activity = True
        args.holdout = True
        args.subject_holdout = True
        args.significance = True
        args.det_plots = True
        if args.cancelability_keys < 2:
            args.cancelability_keys = 16
    return args


def run_dataset(
    spec: DatasetSpec,
    segments: BiometricSegments,
    *,
    args: argparse.Namespace,
    regimes: tuple[KeyMode, ...],
    out_dir: Path,
) -> dict[str, object]:
    """Run the configured analysis blocks for one cohort, writing to ``out_dir``.

    Feature levels are derived from this cohort's own segment length, so no config
    bleeds across datasets.

    Args:
        spec: The cohort being evaluated.
        segments: Its loaded ECG/PPG segments.
        args: Parsed CLI options.
        regimes: Key regimes to sweep.
        out_dir: Per-dataset output directory.

    Returns:
        A one-row headline summary for the cross-dataset comparison table.
    """
    if args.feature_levels:
        feature_levels = tuple(args.feature_levels)
    else:
        max_level = max_feature_level(segments.segment_length, FEATURE_WAVELET)
        feature_levels = tuple(range(1, max_level + 1))
    representative_level = max(feature_levels)
    logger.info(
        "[%s] %d subjects · %d segments · %d-sample @ %d Hz · feature levels %s.",
        spec.name, segments.num_subjects, segments.num_segments,
        segments.segment_length, segments.sampling_rate, feature_levels,
    )
    _write_manifest(
        out_dir, args, n_segments=segments.num_segments, n_subjects=segments.num_subjects,
    )

    summary: dict[str, object] = {
        "dataset": spec.name,
        "n_subjects": segments.num_subjects,
        "n_segments": segments.num_segments,
        "segment_length": segments.segment_length,
        "sampling_rate": segments.sampling_rate,
    }

    id_df, ver_df, id_results = _identification_and_verification(
        segments,
        regimes=regimes,
        feature_levels=feature_levels,
        projection_ratio=args.projection_ratio,
        binarise=args.binarise,
        n_folds=args.cv_folds,
        split_seeds=tuple(args.split_seeds),
        run_identification=args.protocol in ("identification", "both"),
        run_verification=args.protocol in ("verification", "both"),
        tune=args.tune,
    )
    _save_csv(id_df, out_dir / "metrics.csv")
    _save_csv(ver_df, out_dir / "verification.csv")
    if not id_df.empty:
        best = id_df.loc[id_df["eer_mean"].idxmin()]
        summary["identification_eer"] = float(id_df["eer_mean"].min())
        summary["identification_auc"] = float(id_df["auc_mean"].max())
        summary["best_regime"] = str(best["regime"])
        summary["best_classifier"] = str(best["classifier"])
    if not ver_df.empty:
        closed = ver_df[ver_df["verification_mode"] == "closed_set"]
        if not closed.empty:
            summary["verification_eer"] = float(closed["eer_mean"].min())

    if args.significance and id_results:
        _save_csv(
            _significance_df(id_results, n_folds=args.cv_folds),
            out_dir / "significance.csv",
        )

    if args.cancelability_keys >= 2:
        _save_csv(
            _cancelability_df(
                segments, feature_levels=feature_levels,
                projection_ratio=args.projection_ratio, binarise=args.binarise,
                n_keys=args.cancelability_keys, seed=args.seed,
            ),
            out_dir / "cancelability.csv",
        )
    ni_pools: dict[str, np.ndarray] | None = None
    ni_report: dict | None = None
    if args.non_invertibility:
        ni_report_df, ni_pools_df, ni_pools = _non_invertibility_outputs(
            segments, feature_level=representative_level,
            projection_ratio=args.projection_ratio, binarise=args.binarise,
            max_victims=args.non_invertibility_victims, seed=args.seed,
        )
        _save_csv(ni_report_df, out_dir / "non_invertibility.csv")
        _save_csv(ni_pools_df, out_dir / "non_invertibility_pools.csv")
        if not ni_report_df.empty:
            ni_report = ni_report_df.iloc[0].to_dict()
            summary["non_invertibility_leakage_gap"] = float(ni_report["leakage_gap"])
            summary["non_invertibility_sar1"] = float(ni_report["sar_type1"])
    if args.stolen_token:
        st_df = _stolen_token_df(
            segments, feature_levels=feature_levels,
            projection_ratio=args.projection_ratio, binarise=args.binarise, seed=args.seed,
        )
        _save_csv(st_df, out_dir / "stolen_token.csv")
        raw = st_df[st_df["score_norm"] == "raw"]
        if not raw.empty:
            summary["stolen_token_eer"] = float(raw["eer"].min())
    if spec.enrol_activity is not None and args.cross_activity:
        ca_df = _cross_activity_df(
            spec, segments, max_subjects=args.max_subjects,
            projection_ratio=args.projection_ratio, binarise=args.binarise, seed=args.seed,
        )
        _save_csv(ca_df, out_dir / "cross_activity.csv")
        cross = ca_df[~ca_df["within_condition"]] if not ca_df.empty else ca_df
        if not cross.empty:
            summary["cross_activity_mean_eer"] = float(cross["eer"].mean())
    if args.holdout:
        _save_csv(
            _holdout_df(
                segments, regimes=regimes, feature_levels=(representative_level,),
                projection_ratio=args.projection_ratio, binarise=args.binarise,
                test_fraction=args.holdout_fraction,
            ),
            out_dir / "holdout.csv",
        )
    if args.subject_holdout:
        _save_csv(
            _subject_holdout_df(
                segments, regimes=regimes, feature_levels=(representative_level,),
                projection_ratio=args.projection_ratio, binarise=args.binarise,
                test_fraction=args.holdout_fraction, n_folds=args.cv_folds,
                seed=args.seed,
            ),
            out_dir / "subject_holdout.csv",
        )
    if args.det_plots:
        _plot_figures(
            segments, id_df, regimes=regimes, feature_level=representative_level,
            projection_ratio=args.projection_ratio, binarise=args.binarise,
            n_folds=args.cv_folds, fig_dir=out_dir / "figures",
            non_invertibility_pools=ni_pools,
            non_invertibility_report=ni_report,
            plot_stolen_token=args.stolen_token, seed=args.seed,
        )
    logger.info("[%s] Done. Outputs under %s", spec.name, out_dir)
    return summary


def _resolve_datasets(selected: list[str]) -> list[DatasetSpec]:
    """Map the ``--datasets`` selection (``all`` or explicit keys) to ordered specs."""
    keys = list(DATASET_SPECS) if "all" in selected else list(dict.fromkeys(selected))
    return [DATASET_SPECS[k] for k in keys]


def main(argv: list[str] | None = None) -> None:
    """Run the configured analyses across every selected dataset.

    Each dataset writes to ``<output-dir>/<Name>_<date>_<hour>/`` (all sharing one
    run tag); the cross-dataset comparison goes to ``<output-dir>/shared/``.
    """
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    set_global_seeds(args.seed)

    regimes = tuple(KeyMode(r) for r in args.regimes)
    specs = _resolve_datasets(args.datasets)
    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M")
    logger.info(
        "Evaluating %d dataset(s): %s (run tag %s).",
        len(specs), ", ".join(s.name for s in specs), run_tag,
    )

    summaries: list[dict[str, object]] = []
    for spec in specs:
        segments = _subset_segments(spec.load(), args.max_subjects)
        out_dir = args.output_dir / f"{spec.name}_{run_tag}"
        summaries.append(
            run_dataset(spec, segments, args=args, regimes=regimes, out_dir=out_dir)
        )

    # Shared cross-dataset headline comparison (one row per dataset).
    _save_csv(
        pd.DataFrame(summaries),
        args.output_dir / "shared" / f"dataset_comparison_{run_tag}.csv",
    )
    logger.info("Done. Per-dataset folders + shared/ under %s", args.output_dir)


if __name__ == "__main__":
    main()
