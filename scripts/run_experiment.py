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
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# Silence warnings on the console and route unique ones to a file. Installed
# before importing mwf (→ pyeer) so the import-time deprecation warnings are
# caught too; the log file is chosen later, once the run tag is known.
import warning_capture  # noqa: E402

warning_capture.install()

from mwf import (  # noqa: E402
    BiometricSegments,
    DEFAULT_PROJECTION_RATIO,
    DEFAULT_SPLIT_SEEDS,
    FEATURE_WAVELET,
    KeyMode,
    METRIC_NAMES,
    build_templates_from_features,
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
    extract_features_batch,
    load_bidmc,
    load_mimic100,
    load_ptt_ppg,
    make_pipeline,
    max_feature_level,
    preprocess_signals,
    nadeau_bengio_ci_mean,
    non_invertibility_analysis,
    rank_k_accuracies,
    run_verification_cv,
    set_global_seeds,
    stolen_token_score_pools,
    stolen_token_verification,
    subject_holdout,
    summarise_run,
    TemplateBundle,
    temporal_holdout_per_subject,
    VerificationMode,
)
from mwf.batch_utils import parallel_map  # noqa: E402
from mwf.cancelability import (  # noqa: E402
    _assemble_cancelability_report,
    _per_class_abs_corr,
    _random_tokens,
    _same_key_genuine_mean,
    _standardize_columns,
    _templates_for_token,
)
from mwf.pipeline import _single_threaded  # noqa: E402
from mwf.scoring import genuine_impostor_scores  # noqa: E402
from mwf.progress import track  # noqa: E402
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
    """Initialise logging so the run always narrates its current step.

    The root logger follows ``verbose`` (``INFO`` vs ``WARNING``) to gate library
    chatter, but this driver's own logger is pinned to ``INFO`` regardless so the
    "Step: …" banners that tell you what stage the run is on always appear — even
    without ``--verbose`` — alongside the progress bars.

    Args:
        verbose: If ``True`` also surface library ``INFO`` logs; otherwise only
            this driver's step banners and warnings.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(logging.INFO)


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


def _build_bundles(
    segments: BiometricSegments,
    *,
    regimes: tuple[KeyMode, ...],
    feature_levels: tuple[int, ...],
    projection_ratio: float,
    binarise: bool,
) -> tuple[dict[tuple[KeyMode, int], TemplateBundle], dict[int, np.ndarray]]:
    """Build the BioHashing templates for every ``(regime, level)`` once.

    Returns ``(bundles, features_by_level)`` so the caller can cache the
    already-extracted features and pass them to downstream analyses
    (cancelability, stolen-token, non-invertibility, figures) instead of
    re-running NeuroKit cleaning + DWT for each one.
    """
    features_by_level = _features_per_level(segments, feature_levels)
    bundles = {
        (regime, level): build_templates_from_features(
            features_by_level[level], segments.labels,
            key_mode=regime, projection_ratio=projection_ratio, binarise=binarise,
        )
        for regime in regimes
        for level in feature_levels
    }
    return bundles, features_by_level


def _features_per_level(
    segments: BiometricSegments, feature_levels: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Clean the cohort once and extract features once per DWT level.

    Cleaning (NeuroKit) and DWT extraction are the dominant per-cohort costs and
    depend only on the signals / level — never on the key regime — so this caches
    them for callers that sweep regimes (or analyses) over one cohort. Pair with
    :func:`build_templates_from_features` to apply the cheap cancelable transform
    per regime without re-cleaning. Results are byte-identical to per-bundle
    :func:`build_templates`.
    """
    ecg, ppg = preprocess_signals(
        segments.ecg, segments.ppg, sampling_rate=segments.sampling_rate,
    )
    return {
        level: extract_features_batch(ecg, ppg, wavelet=FEATURE_WAVELET, level=level)
        for level in feature_levels
    }


def _identification_tasks(
    bundles: dict[tuple[KeyMode, int], TemplateBundle],
    *,
    regimes: tuple[KeyMode, ...],
    feature_levels: tuple[int, ...],
    tune: bool,
) -> list[CVTask]:
    """Build one :class:`CVTask` per ``(regime, level, classifier)``.

    The grid is returned flat (keyed by ``(regime, level, name)``) so several
    cohorts' grids can be merged into a single worker pool — one scheduling tail
    for the whole battery instead of one per dataset.

    Args:
        tune: When ``True`` each task carries the classifier's hyperparameter grid
            so every outer fold is tuned with a group-aware inner CV (nested CV),
            removing the selection-on-the-evaluation-data bias of fixed configs.
    """
    return [
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


def _assemble_identification(
    tasks: Sequence[CVTask],
    results: Sequence[CrossValidationResult],
    *,
    tuned: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, CrossValidationResult]]]:
    """Flatten this cohort's pooled CV results into its metrics frame + map.

    Args:
        tasks: The cohort's identification tasks, in submission order.
        results: The :class:`CrossValidationResult` per task, aligned with ``tasks``.
        tuned: Whether the classifiers were tuned (recorded on each row).

    Returns:
        Tuple ``(identification_df, id_results)``; ``id_results`` maps
        ``"{regime}@L{level}"`` → ``{classifier: result}`` so the paired
        significance tests compare classifiers on identical folds.
    """
    id_rows: list[dict] = []
    id_results: dict[str, dict[str, CrossValidationResult]] = {}
    for task, result in zip(tasks, results):
        regime, level, name = cast(tuple[KeyMode, int, str], task.key)
        id_results.setdefault(f"{regime.value}@L{level}", {})[name] = result
        id_rows.append(
            _identification_metric_row(regime.value, level, name, result, tuned=tuned)
        )
    return pd.DataFrame(id_rows), id_results


def _verification_df(
    bundles: dict[tuple[KeyMode, int], TemplateBundle],
    *,
    regimes: tuple[KeyMode, ...],
    feature_levels: tuple[int, ...],
    n_folds: int,
) -> pd.DataFrame:
    """Fan the independent ``(regime, level, mode)`` verification CVs over cores.

    The bundle travels inside the work item (not a closure) so joblib only ships
    each template matrix to the worker that needs it. Row order, fold splits and
    aggregates are identical to the previous sequential nested loops.
    """
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
        desc="Verification CV",
    )
    return pd.DataFrame([
        _verification_metric_row(regime.value, level, mode, ver, n_folds=n_folds)
        for (regime, level, mode), ver in zip(ver_keys, vers)
    ])


def _cancelability_df(
    segments: BiometricSegments,
    *,
    feature_levels: tuple[int, ...],
    projection_ratio: float,
    binarise: bool,
    n_keys: int,
    seed: int,
    features_by_level: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Run the ISO/IEC 30136 cancelability protocol per feature level.

    Flattens the (level × token) space into one joblib pool: features are
    extracted once per level, base templates are built once per level
    (sequential, cheap), then all n_levels × (n_keys − 1) token jobs run
    concurrently.  Results are assembled per level after the pool drains;
    the bootstrap CIs run sequentially on the collected arrays.  CSVs are
    byte-identical to the old sequential implementation under OMP=1.
    """
    from collections import defaultdict

    from joblib import Parallel, delayed, parallel_config
    from mwf.progress import tqdm_joblib

    if features_by_level is None:
        features_by_level = _features_per_level(segments, feature_levels)
    tokens = _random_tokens(n_keys, seed=seed)
    labels = segments.labels
    n_subjects = max(1, int(np.unique(labels).size))

    # Build base templates per level (sequential; one BioHashing call per level).
    bases_by_level: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}
    for level in feature_levels:
        base = _templates_for_token(
            features_by_level[level], tokens[0], projection_ratio, binarise,
        )
        bases_by_level[level] = (
            base,
            _same_key_genuine_mean(base, labels),
            _standardize_columns(base),
        )

    # One job per (level, token) pair — n_levels × (n_keys − 1) total.
    # Arrays are top-level positional args so joblib memory-maps them read-only.
    def _one_token_job(
        level: int,
        token: str,
        feats: np.ndarray,
        base: np.ndarray,
        baseline_mean: float,
        base_z: np.ndarray,
    ) -> tuple:
        reissued = _templates_for_token(feats, token, projection_ratio, binarise)
        mated, non_mated = genuine_impostor_scores(base, reissued, labels)
        reissued_z = _standardize_columns(reissued)
        diversity = _per_class_abs_corr(base_z, reissued_z, labels)
        return level, baseline_mean, mated, non_mated, diversity

    jobs = [
        (level, tok, features_by_level[level], *bases_by_level[level])
        for level in feature_levels
        for tok in tokens[1:]
    ]
    n_jobs = len(jobs)
    with parallel_config(backend="loky", inner_max_num_threads=1), \
            tqdm_joblib(n_jobs, desc="Cancelability"):
        raw = Parallel(n_jobs=-1)(
            delayed(_one_token_job)(level, tok, feats, base, bm, bz)
            for level, tok, feats, base, bm, bz in jobs
        )

    # Collect per-level arrays in tokens[1:] order (preserved by joblib).
    groups: dict[int, dict] = {
        lv: {"renew": [], "mated": [], "non_mated": [], "div": [], "bm": None}
        for lv in feature_levels
    }
    for level, baseline_mean, mated, non_mated, diversity in raw:
        g = groups[level]
        g["renew"].append(mated)
        g["mated"].append(mated)
        g["non_mated"].append(non_mated)
        g["div"].append(diversity)
        g["bm"] = baseline_mean

    rows = []
    for level in feature_levels:
        g = groups[level]
        base_shape = bases_by_level[level][0].shape
        report = _assemble_cancelability_report(
            n_keys=n_keys,
            renew_per_key=g["renew"],
            mated_pool=g["mated"],
            non_mated_pool=g["non_mated"],
            diversity_per_key=g["div"],
            baseline_mean=g["bm"],
            template_dim=int(base_shape[1]),
            segs_per_subject=base_shape[0] / n_subjects,
            seed=seed,
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
    features: np.ndarray | None = None,
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
        features=features,
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
    features_by_level: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Worst-case (stolen-key) EER per feature level and score-normalisation.

    Features are cleaned+extracted once per level and reused for both
    score-normalisations (the front-end is token- and norm-independent), so the
    two ``score_norm`` runs of a level no longer re-clean the cohort. The
    ``(level, score_norm)`` evaluations are independent, so they fan out across
    cores; the work list keeps the original ``level`` → ``score_norm`` nesting
    order, so the CSV is byte-identical to the sequential loops. The per-level
    feature matrix rides inside the work item so joblib memory-maps it once and
    shares it read-only across the two norm jobs.
    """
    if features_by_level is None:
        features_by_level = _features_per_level(segments, feature_levels)

    # Iterate (level, norm) sequentially so the per-victim Parallel pool inside
    # stolen_token_score_pools fires at the top level and fills all 64 cores.
    # A parallel_map outer loop suppresses that inner pool via joblib nesting and
    # leaves most cores idle (14 outer jobs × 1 core each vs 64-core victim pool).
    rows: list[dict] = []
    for level in track(feature_levels, desc="Stolen-token levels"):
        feats = features_by_level[level]
        for score_norm in (None, "znorm"):
            r = stolen_token_verification(
                segments, feature_level=level, projection_ratio=projection_ratio,
                binarise=binarise, seed=seed, score_norm=score_norm, features=feats,
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

    Parallelism: the train/test templates are built once per ``(regime, level)``
    (each build self-parallelises its feature extraction), then the independent
    ``(regime, level, classifier)`` fit/score jobs fan out across cores. Row order
    matches the sequential nested loops, so ``holdout.csv`` is byte-identical;
    each classifier is forced single-threaded so the fan-out owns the cores.
    """
    split = temporal_holdout_per_subject(segments, test_fraction=test_fraction)
    # Clean + extract each split once per level (regime-independent), then build
    # the train/test templates per (regime, level) from the cached features; the
    # scaler is fitted on train and reused on test (leakage-free). Reused by every
    # classifier. Byte-identical to the old per-bundle build.
    train_feats = _features_per_level(split.train, feature_levels)
    test_feats = _features_per_level(split.test, feature_levels)
    bundles: dict[tuple[KeyMode, int], tuple[TemplateBundle, TemplateBundle]] = {
        (regime, level): (
            train_b := build_templates_from_features(
                train_feats[level], split.train.labels,
                key_mode=regime, projection_ratio=projection_ratio, binarise=binarise,
            ),
            build_templates_from_features(
                test_feats[level], split.test.labels,
                key_mode=regime, projection_ratio=projection_ratio, binarise=binarise,
                scaler=train_b.scaler,
            ),
        )
        for regime in regimes
        for level in feature_levels
    }
    jobs = [
        (regime, level, name, *bundles[(regime, level)])
        for regime in regimes
        for level in feature_levels
        for name in CLASSIFIER_NAMES
    ]

    def _fit_eval(
        job: tuple[KeyMode, int, str, TemplateBundle, TemplateBundle],
    ) -> dict:
        regime, level, name, train_b, test_b = job
        pipe = make_pipeline(_single_threaded(build_classifier(name)))
        pipe.fit(train_b.features, train_b.labels)
        y_score = class_score_matrix(pipe, test_b.features)
        classes = np.asarray(pipe.named_steps["clf"].classes_, dtype=np.int64)
        y_pred = np.asarray(pipe.predict(test_b.features), dtype=np.int64)
        metrics = evaluate(test_b.labels, y_pred, y_score, classes)
        ranks = rank_k_accuracies(test_b.labels, y_score, classes)
        return {
            "regime": regime.value,
            "feature_level": level,
            "classifier": name,
            "n_train": int(train_b.features.shape[0]),
            "n_test": int(test_b.features.shape[0]),
            **metrics.as_dict(),
            **ranks,
        }

    return pd.DataFrame(parallel_map(jobs, _fit_eval, desc="Temporal holdout"))


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

    Parallelism: each ``(regime, level)`` test bundle is built once, then the
    independent ``(regime, level, mode)`` verification CVs fan out across cores
    (the same pattern as the main verification sweep). Row order, splits and
    aggregates match the sequential loops, so ``subject_holdout.csv`` is identical.
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
    # Clean + extract the held-out cohort once per level, then build each
    # (regime, level) test bundle from the cached features; reused across both
    # modes. Byte-identical to the old per-bundle build.
    test_feats = _features_per_level(split.test, feature_levels)
    test_bundles = {
        (regime, level): build_templates_from_features(
            test_feats[level], split.test.labels,
            key_mode=regime, projection_ratio=projection_ratio, binarise=binarise,
        )
        for regime in regimes
        for level in feature_levels
    }
    jobs = [
        (regime, level, mode, test_bundles[(regime, level)])
        for regime in regimes
        for level in feature_levels
        for mode in ("closed_set", "open_set")
    ]

    def _verify(
        job: tuple[KeyMode, int, str, TemplateBundle],
    ) -> tuple[str, ...] | tuple[str, dict]:
        regime, level, mode, test_b = job
        try:
            ver = run_verification_cv(test_b, mode=mode, n_folds=eff_folds)
        except ValueError as exc:  # degenerate split (too few groups/blocks)
            return ("skipped", mode, str(exc))
        eer = ver.eer_values()
        deci = ver.decidability_values()
        finite_eer = eer[np.isfinite(eer)]
        eer_ci_lo, eer_ci_hi = nadeau_bengio_ci_mean(
            finite_eer, n_folds=eff_folds,
        ) if finite_eer.size > 1 else (float("nan"), float("nan"))
        return ("ok", {
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

    rows: list[dict] = []
    for status, *payload in parallel_map(jobs, _verify, desc="Subject holdout"):
        if status == "skipped":
            mode, exc = payload
            logger.warning("Subject-holdout %s verification skipped (%s).", mode, exc)
        else:
            rows.append(payload[0])
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
    single-recording cohorts. Activities are iterated sequentially so the
    per-victim Parallel pool inside cross_session_score_pools fires at the
    top level and fills all 64 cores (outer activity parallelism would suppress
    the inner pool via joblib nesting, leaving most cores idle).
    """
    if spec.enrol_activity is None or spec.load_activity is None:
        return pd.DataFrame()
    activities = (spec.enrol_activity, *spec.probe_activities)
    rows: list[dict] = []
    for activity in track(list(activities), desc="Cross-activity"):
        within = activity == spec.enrol_activity
        probe_segments = (
            enrol_segments if within
            else _subset_segments(spec.load_activity(activity), max_subjects)  # type: ignore[misc]
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
    cached_features: np.ndarray | None = None,
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
    from joblib import Parallel, delayed, parallel_config

    features = (
        cached_features
        if cached_features is not None
        else _features_per_level(segments, (feature_level,))[feature_level]
    )

    def _regime_pools(regime: KeyMode) -> tuple[str, tuple[np.ndarray, np.ndarray]]:
        bundle = build_templates_from_features(
            features, segments.labels,
            key_mode=regime, projection_ratio=projection_ratio, binarise=binarise,
        )
        return regime.value, closed_set_score_pools(bundle, n_folds=n_folds)

    with parallel_config(backend="loky", inner_max_num_threads=1):
        regime_results = Parallel(n_jobs=-1)(
            delayed(_regime_pools)(regime) for regime in regimes
        )
    pools: dict[str, tuple[np.ndarray, np.ndarray]] = dict(regime_results)

    plot_det_curves(pools, fig_dir / "det.png")
    plot_roc_curves(pools, fig_dir / "roc.png")
    plot_pr_curves(pools, fig_dir / "pr.png")
    for regime_label, (genuine, impostor) in pools.items():
        plot_score_distribution(
            genuine, impostor, fig_dir / f"scores_{regime_label}.png",
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


@dataclass(frozen=True, slots=True)
class DatasetWork:
    """A prepared cohort awaiting (and then carrying) its pooled CV results.

    Produced by :func:`_prepare_dataset` (load → templates → identification task
    list). The identification grids of *all* cohorts are then cross-validated
    through one flat pool, and :func:`_finish_dataset` turns each cohort's slice
    of those results into its CSVs, figures and headline summary.

    Attributes:
        spec: The cohort being evaluated.
        segments: Its loaded ECG/PPG segments.
        regimes: Key regimes swept for this cohort.
        out_dir: Per-dataset output directory.
        feature_levels: DWT depths swept (derived from this cohort's segments).
        representative_level: Deepest level, used by the single-level analyses.
        bundles: Templates per ``(regime, level)``, shared by both protocols.
        id_tasks: The flat identification grid (empty when identification is off).
    """

    spec: DatasetSpec
    segments: BiometricSegments
    regimes: tuple[KeyMode, ...]
    out_dir: Path
    feature_levels: tuple[int, ...]
    representative_level: int
    bundles: dict[tuple[KeyMode, int], TemplateBundle]
    features_by_level: dict[int, np.ndarray]
    id_tasks: tuple[CVTask, ...]


def _prepare_dataset(
    spec: DatasetSpec,
    segments: BiometricSegments,
    *,
    args: argparse.Namespace,
    regimes: tuple[KeyMode, ...],
    out_dir: Path,
) -> DatasetWork:
    """Set up one cohort: feature levels, templates and the identification grid.

    Feature levels are derived from this cohort's own segment length, so no config
    bleeds across datasets. The templates are built once (reused by both protocols)
    and the flat identification task list is assembled, but no CV is run — that is
    pooled across every cohort by :func:`main`, so cores stay full across the
    unequal cohorts instead of draining on each dataset's tail.

    Args:
        spec: The cohort being evaluated.
        segments: Its loaded ECG/PPG segments.
        args: Parsed CLI options.
        regimes: Key regimes to sweep.
        out_dir: Per-dataset output directory.

    Returns:
        A :class:`DatasetWork` ready for the pooled CV and :func:`_finish_dataset`.
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
    bundles, features_by_level = _build_bundles(
        segments, regimes=regimes, feature_levels=feature_levels,
        projection_ratio=args.projection_ratio, binarise=args.binarise,
    )
    id_tasks = (
        tuple(_identification_tasks(
            bundles, regimes=regimes, feature_levels=feature_levels, tune=args.tune,
        ))
        if args.protocol in ("identification", "both") else ()
    )
    return DatasetWork(
        spec=spec, segments=segments, regimes=regimes, out_dir=out_dir,
        feature_levels=feature_levels, representative_level=representative_level,
        bundles=bundles, features_by_level=features_by_level, id_tasks=id_tasks,
    )


def _finish_dataset(
    work: DatasetWork,
    id_results_raw: Sequence[CrossValidationResult],
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Turn one cohort's pooled CV results into its CSVs, figures and summary.

    Args:
        work: The prepared cohort (templates, identification tasks, output dir).
        id_results_raw: This cohort's slice of the pooled identification results,
            aligned with ``work.id_tasks`` (empty when identification is off).
        args: Parsed CLI options.

    Returns:
        A one-row headline summary for the cross-dataset comparison table.
    """
    spec = work.spec
    segments = work.segments
    regimes = work.regimes
    out_dir = work.out_dir
    feature_levels = work.feature_levels
    representative_level = work.representative_level

    summary: dict[str, object] = {
        "dataset": spec.name,
        "n_subjects": segments.num_subjects,
        "n_segments": segments.num_segments,
        "segment_length": segments.segment_length,
        "sampling_rate": segments.sampling_rate,
    }

    if work.id_tasks:
        id_df, id_results = _assemble_identification(
            work.id_tasks, id_results_raw, tuned=args.tune,
        )
    else:
        id_df, id_results = pd.DataFrame(), {}

    if args.protocol in ("verification", "both"):
        logger.info("[%s] Step: verification CV.", spec.name)
        ver_df = _verification_df(
            work.bundles, regimes=regimes, feature_levels=feature_levels,
            n_folds=args.cv_folds,
        )
    else:
        ver_df = pd.DataFrame()
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
        logger.info("[%s] Step: classifier significance tests.", spec.name)
        _save_csv(
            _significance_df(id_results, n_folds=args.cv_folds),
            out_dir / "significance.csv",
        )

    if args.cancelability_keys >= 2:
        logger.info("[%s] Step: cancelability protocol (%d keys).",
                    spec.name, args.cancelability_keys)
        _save_csv(
            _cancelability_df(
                segments, feature_levels=feature_levels,
                projection_ratio=args.projection_ratio, binarise=args.binarise,
                n_keys=args.cancelability_keys, seed=args.seed,
                features_by_level=work.features_by_level,
            ),
            out_dir / "cancelability.csv",
        )
    ni_pools: dict[str, np.ndarray] | None = None
    ni_report: dict | None = None
    if args.non_invertibility:
        logger.info("[%s] Step: non-invertibility analysis.", spec.name)
        ni_report_df, ni_pools_df, ni_pools = _non_invertibility_outputs(
            segments, feature_level=representative_level,
            projection_ratio=args.projection_ratio, binarise=args.binarise,
            max_victims=args.non_invertibility_victims, seed=args.seed,
            features=work.features_by_level.get(representative_level),
        )
        _save_csv(ni_report_df, out_dir / "non_invertibility.csv")
        _save_csv(ni_pools_df, out_dir / "non_invertibility_pools.csv")
        if not ni_report_df.empty:
            ni_report = ni_report_df.iloc[0].to_dict()
            summary["non_invertibility_leakage_gap"] = float(ni_report["leakage_gap"])
            summary["non_invertibility_sar1"] = float(ni_report["sar_type1"])
    if args.stolen_token:
        logger.info("[%s] Step: stolen-token worst-case EER.", spec.name)
        st_df = _stolen_token_df(
            segments, feature_levels=feature_levels,
            projection_ratio=args.projection_ratio, binarise=args.binarise, seed=args.seed,
            features_by_level=work.features_by_level,
        )
        _save_csv(st_df, out_dir / "stolen_token.csv")
        raw = st_df[st_df["score_norm"] == "raw"]
        if not raw.empty:
            summary["stolen_token_eer"] = float(raw["eer"].min())
    if spec.enrol_activity is not None and args.cross_activity:
        logger.info("[%s] Step: cross-activity verification.", spec.name)
        ca_df = _cross_activity_df(
            spec, segments, max_subjects=args.max_subjects,
            projection_ratio=args.projection_ratio, binarise=args.binarise, seed=args.seed,
        )
        _save_csv(ca_df, out_dir / "cross_activity.csv")
        cross = ca_df[~ca_df["within_condition"]] if not ca_df.empty else ca_df
        if not cross.empty:
            summary["cross_activity_mean_eer"] = float(cross["eer"].mean())
    if args.holdout:
        logger.info("[%s] Step: temporal holdout.", spec.name)
        _save_csv(
            _holdout_df(
                segments, regimes=regimes, feature_levels=(representative_level,),
                projection_ratio=args.projection_ratio, binarise=args.binarise,
                test_fraction=args.holdout_fraction,
            ),
            out_dir / "holdout.csv",
        )
    if args.subject_holdout:
        logger.info("[%s] Step: subject-disjoint holdout.", spec.name)
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
        logger.info("[%s] Step: rendering figure suite.", spec.name)
        _plot_figures(
            segments, id_df, regimes=regimes, feature_level=representative_level,
            projection_ratio=args.projection_ratio, binarise=args.binarise,
            n_folds=args.cv_folds, fig_dir=out_dir / "figures",
            non_invertibility_pools=ni_pools,
            non_invertibility_report=ni_report,
            plot_stolen_token=args.stolen_token, seed=args.seed,
            cached_features=work.features_by_level.get(representative_level),
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

    The identification grids of *every* cohort are merged into one flat worker
    pool (a single scheduling tail for the whole battery) instead of one pool per
    dataset, so cores stay full across the unequal cohorts rather than draining on
    each dataset's tail. Fold partitions are deterministic per seed and each
    task's result depends only on its own bundle, so pooling changes wall-clock
    only — every CSV is byte-identical to evaluating each dataset on its own.
    """
    args = _parse_args(argv)
    _configure_logging(args.verbose)
    set_global_seeds(args.seed)

    regimes = tuple(KeyMode(r) for r in args.regimes)
    specs = _resolve_datasets(args.datasets)
    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M")
    warnings_log = args.output_dir / f"warnings_{run_tag}.log"
    warning_capture.set_logfile(warnings_log)
    logger.info(
        "Evaluating %d dataset(s): %s (run tag %s).",
        len(specs), ", ".join(s.name for s in specs), run_tag,
    )
    logger.info("Console warnings silenced; unique warnings → %s", warnings_log)

    # Phase 1: load every cohort and build its templates + identification grid.
    # Loading (disk I/O + WFDB parsing) is parallelised across datasets with
    # threads; template building (NeuroKit + DWT, already row-parallel) runs
    # sequentially so each dataset's internal pool keeps all cores.
    logger.info("Phase 1/3: loading cohorts and building BioHashing templates.")

    def _load_one(spec: DatasetSpec) -> BiometricSegments:
        logger.info("Loading %s …", spec.name)
        return _subset_segments(spec.load(), args.max_subjects)

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        loaded_segments = list(pool.map(_load_one, specs))

    works = []
    for spec, segments in track(zip(specs, loaded_segments), desc="Building templates", total=len(specs)):
        works.append(
            _prepare_dataset(
                spec, segments,
                args=args, regimes=regimes,
                out_dir=args.output_dir / f"{spec.name}_{run_tag}",
            )
        )

    # Phase 2: cross-validate every cohort's identification grid in ONE pool, so
    # the small cohorts' cores backfill the big cohort's tail instead of idling.
    all_tasks = [task for work in works for task in work.id_tasks]
    if all_tasks:
        logger.info(
            "Phase 2/3: identification CV over %d task(s), %d fold(s) × %d seed(s).",
            len(all_tasks), args.cv_folds, len(args.split_seeds),
        )
    pooled = (
        cross_validate_tasks(
            all_tasks, n_folds=args.cv_folds,
            split_seeds=tuple(args.split_seeds), ranks=RANK_TARGETS,
        )
        if all_tasks else []
    )

    # Phase 3: hand each cohort its slice of the pooled results and finish it.
    logger.info("Phase 3/3: per-cohort analyses, CSVs and figures.")
    summaries: list[dict[str, object]] = []
    offset = 0
    for work in works:
        n_tasks = len(work.id_tasks)
        summaries.append(
            _finish_dataset(work, pooled[offset:offset + n_tasks], args=args)
        )
        offset += n_tasks

    # Shared cross-dataset headline comparison (one row per dataset).
    _save_csv(
        pd.DataFrame(summaries),
        args.output_dir / "shared" / f"dataset_comparison_{run_tag}.csv",
    )
    logger.info("Done. Per-dataset folders + shared/ under %s", args.output_dir)


if __name__ == "__main__":
    main()
