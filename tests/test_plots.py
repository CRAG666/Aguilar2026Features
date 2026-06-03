"""Smoke tests for the figure suite (mwf.plots) — files render and are non-empty."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.plots import (
    plot_classifier_comparison,
    plot_det_curves,
    plot_pr_curves,
    plot_regime_summary,
    plot_roc_curves,
    plot_score_distribution,
    plot_stolen_token_scores,
)

RNG = np.random.default_rng(seed=20260527)


def _pools():
    """Two regimes with separable genuine (high) and impostor (low) cosine pools."""
    return {
        "identity": (RNG.normal(0.8, 0.1, 400), RNG.normal(0.1, 0.1, 400)),
        "per_subject": (RNG.normal(0.7, 0.1, 400), RNG.normal(0.0, 0.1, 400)),
    }


def _metrics_df():
    rows = []
    for regime in ("identity", "single_key", "per_subject"):
        for clf in ("LR", "SVM", "RF"):
            for dim in (52, 104, 156):
                rows.append({
                    "regime": regime, "classifier": clf, "n_template_dims": dim,
                    "eer_mean": RNG.uniform(0.01, 0.1),
                    "auc_mean": RNG.uniform(0.9, 0.999),
                    "ap_mean": RNG.uniform(0.9, 0.999),
                })
    return pd.DataFrame(rows)


def _nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def test_curve_figures_render(tmp_path):
    pools = _pools()
    for fn in (plot_det_curves, plot_roc_curves, plot_pr_curves):
        out = tmp_path / f"{fn.__name__}.png"
        fn(pools, out)
        assert _nonempty(out)


def test_score_distribution_renders(tmp_path):
    g, i = _pools()["identity"]
    out = tmp_path / "scores.png"
    plot_score_distribution(g, i, out)
    assert _nonempty(out)


def test_summary_figures_render(tmp_path):
    df = _metrics_df()
    summary = tmp_path / "regime_summary.png"
    plot_regime_summary(df, summary)
    assert _nonempty(summary)
    clf = tmp_path / "clf_vs_features_per_subject.png"
    plot_classifier_comparison(df, clf, "per_subject")
    assert _nonempty(clf)


def test_summary_skips_empty_frame(tmp_path):
    out = tmp_path / "empty.png"
    plot_regime_summary(pd.DataFrame(), out)
    assert not out.exists()


def test_classifier_comparison_skips_unknown_regime(tmp_path):
    out = tmp_path / "missing.png"
    plot_classifier_comparison(_metrics_df(), out, "no_such_regime")
    assert not out.exists()


def test_nested_output_dir_is_created(tmp_path):
    out = tmp_path / "figures" / "det.png"
    plot_det_curves(_pools(), out)
    assert _nonempty(out)


# --- cancelability figures ---------------------------------------------------


def test_stolen_token_scores_renders(tmp_path):
    g = RNG.normal(0.7, 0.1, 500)
    i = RNG.normal(0.0, 0.1, 500)
    out = tmp_path / "stolen_token_scores.png"
    plot_stolen_token_scores(g, i, out, eer=0.07, decidability=2.9)
    assert _nonempty(out)
