"""Hypothesis probe: how to lift single_key accuracy above 0.90.

Compares, under the shared-token (single_key) regime, several leak-free CV
pipelines that differ only in (a) whether features are standardised *before*
the BioHashing projection and (b) the projection ratio m/d. The shared-token
projection is data-independent (y = R x), so it goes in the pipeline as a
FunctionTransformer and the scalers are fit per-fold (no leakage).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mwf.cv_splits import stratified_group_splitter, temporal_block_groups
from mwf.dataset import load_mimic100
from mwf.feature_transform import derive_projection
from mwf.features import extract_features_batch
from mwf.pipeline import preprocess_signals

LEVEL = 4
SHARED_TOKEN = "SHARED_TOKEN"
CACHE = Path("/tmp/aguilar_feats_l4.npz")
N_FOLDS = 5
SEED = 20260526


def _load_features() -> tuple[np.ndarray, np.ndarray]:
    if CACHE.exists():
        d = np.load(CACHE)
        return d["X"], d["y"]
    seg = load_mimic100()
    ecg, ppg = preprocess_signals(
        seg.ecg, seg.ppg, sampling_rate=seg.sampling_rate,
        snr_db=None, noise_seed=0, denoise=True,
    )
    X = extract_features_batch(ecg, ppg, level=LEVEL)
    y = seg.labels
    np.savez(CACHE, X=X, y=y)
    return X, y


def _projector(matrix: np.ndarray):
    return FunctionTransformer(lambda x: x @ matrix.T, validate=False)


def _classifiers():
    """Fast classifiers — predict only, so SVC needs no (slow) probability cal."""
    return {
        "SVM": SVC(C=275, kernel="rbf", probability=False, random_state=42),
        "LR": LogisticRegression(max_iter=15000, random_state=42),
        "RF": RandomForestClassifier(n_estimators=79, criterion="entropy", random_state=42),
    }


def _pipelines(d: int):
    """Return {name: Pipeline} variants to benchmark."""
    out = {}
    for ratio in (0.5, 1.0):
        m = max(1, round(ratio * d))
        R = derive_projection(SHARED_TOKEN, d, m).matrix  # (m, d)
        proj = _projector(R)
        # A) current: project raw features, then scale+clf
        out[f"raw_proj@{ratio}"] = ("raw", proj)
        # B) fix: scale BEFORE projection, then scale+clf
        out[f"prescale_proj@{ratio}"] = ("prescale", proj)
    return out


def _make_pipe(kind: str, proj, clf):
    if kind == "raw":
        return Pipeline([("proj", proj), ("scl", StandardScaler()), ("clf", clf)])
    return Pipeline(
        [("pre", StandardScaler()), ("proj", proj),
         ("scl", StandardScaler()), ("clf", clf)]
    )


def _cv_accuracy(X, y, kind, proj, clf) -> tuple[float, float]:
    groups = temporal_block_groups(y, segments_per_block=2)
    splitter = stratified_group_splitter(n_splits=N_FOLDS, random_state=SEED)
    accs = []
    for tr, te in splitter.split(X, y, groups=groups):
        pipe = _make_pipe(kind, clone_proj(proj), clone(clf))
        pipe.fit(X[tr], y[tr])
        accs.append(float((pipe.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs)), float(np.std(accs, ddof=1))


def clone_proj(proj):
    # FunctionTransformer holds a closure; cloning is fine (stateless).
    return proj


def main() -> None:
    t0 = time.time()
    X, y = _load_features()
    d = X.shape[1]
    print(f"features: {X.shape}  subjects={np.unique(y).size}  ({time.time()-t0:.1f}s)")

    clfs = _classifiers()
    variants = _pipelines(d)

    print(f"\n{'variant':22s} {'clf':4s} {'acc':>8s} {'std':>7s}")
    print("-" * 46)
    rows = []
    for vname, (kind, proj) in variants.items():
        for cname, clf in clfs.items():
            mean, std = _cv_accuracy(X, y, kind, proj, clf)
            flag = "  <== >0.90" if mean > 0.90 else ""
            print(f"{vname:22s} {cname:4s} {mean:8.4f} {std:7.4f}{flag}", flush=True)
            rows.append((vname, cname, mean, std))
    out_csv = Path(__file__).resolve().parents[1] / "results" / "probe_single_key.csv"
    out_csv.write_text(
        "variant,classifier,accuracy_mean,accuracy_std\n"
        + "\n".join(f"{v},{c},{m:.6f},{s:.6f}" for v, c, m, s in rows) + "\n"
    )
    print(f"\nwrote {out_csv}\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
