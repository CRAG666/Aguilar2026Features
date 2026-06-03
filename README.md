# aguilar2026features

Cancelable **multimodal (ECG + PPG)** biometric identification where the
cancelable transform protects the **feature vector** — not the waveform. It is
the *feature-domain* sibling of `Aguilar2026Wavelet` (which protects the signal
and then extracts features). The pipeline follows the canonical cancelable-
biometrics order:

```
Sensor → Preprocessing (opt. AWGN → NeuroKit clean)   ← mwf.clean (ECG neurokit / PPG elgendi)
       → Extraction (multimodal ECG+PPG feature vector) ← mwf.features (shared.features.wavelet)
       → Cancelable transform T(x, k)  ← Token k         ← ECG: BioHashing · PPG: IoM hashing
       → Protected template → cosine matcher / classifier → Decision
```

Features are extracted **once** and are token-independent; the cancelable
transform acts on the feature vector, so a single feature matrix is re-projected
per token across every regime and analysis.

## The transform — `mwf.feature_transform`

The protected template is a **per-modality hybrid** of the concatenated ECG‖PPG
feature vector `x ∈ ℝ^d` under a user token `k`. The token seeds both
sub-transforms through a SHA-256-derived RNG (`mwf.keystream`), with per-modality
salts (`::ECG`, `::PPG`) keeping their random material independent.

* **ECG block → BioHashing** (Teoh 2004). A `d_e × m` Gaussian is QR-orthonormalised
  into `R` with `R Rᵀ = I_m`; the block template is `R x_ecg`, optionally
  sign-binarised into a ±1 BioCode. `projection_ratio = m/d_e` (default `0.5`) is
  the irreversibility budget: the destroyed `(d_e − m)`-dim null space caps the
  min-norm pre-image leak at ~`√(m/d_e)`.
* **PPG block → Index-of-Max (IoM) hashing** (`mwf.iom`; Jin et al., *IEEE TIFS*
  2018). The token seeds `m` Gaussian stacks of `q` (`IOM_WINDOW = 16`) rows each;
  each hash keeps only the *index of the maximum* response, discarding magnitudes.
  Codes are returned one-hot, so cosine on the code equals the IoM collision rate
  and the existing matcher/classifiers apply unchanged. `IOM_HASHES_RATIO = 0.25`.

Each block is unit-L2-normalised before concatenation, so cosine fuses the two
modalities with equal weight. Before projecting, the feature vector is
per-feature z-scored (`FeatureScaler`, fitted on the enrolment cohort only —
leakage-free under CV/holdout); the wavelet descriptors span orders of magnitude,
and without it a shared-token projection is dominated by high-variance bands.

The PPG block gets the strongly non-invertible IoM because it is the one a linear
inversion reconstructs at `r ≈ 0.78`. IoM also makes PPG discriminability
**key-independent**: as an angle-preserving LSH, its collision rate depends on
the angle between vectors, not on the token — so genuine pairs separate from
impostors even under a *stolen* token.

## Feature extraction — `mwf.features`

Each ECG and PPG segment is decomposed with the DWT (**bior3.3**, a linear-phase
biorthogonal basis that keeps the fiducials aligned) and summarised by **13
descriptors** per subband (mean, std, var, kurtosis, skewness, energy, entropy,
max, min, median, iqr, range, mad), via the shared
`shared.features.wavelet.extract_wavelet_features`. ECG and PPG blocks are
concatenated, so `d = 2 · 13 · (level + 1)` (e.g. `130` at level 4). Data is
MIMIC-100, 6-second ECG/PPG segments at 125 Hz (`mwf.dataset`).

Two further public cohorts plug into the same pipeline via the shared `Datasets`
loaders (needs the optional `wfdb` dependency): **BIDMC** (`load_bidmc`, 53
subjects, ECG+PPG at the same 125 Hz/750-sample shape — a drop-in external
replication of the MIMIC-100 results), and **PTT-PPG** (`load_ptt_ppg(activity)`,
22 subjects recorded `sit`/`walk`/`run` at 500 Hz — separate recordings per
activity that drive the cross-activity protocol below).

## Key regimes — `mwf.pipeline.KeyMode`

* `identity` — raw multimodal features (unprotected biometric ceiling).
* `single_key` — one shared token (biometric-only recognition floor).
* `per_subject` — one token per subject (operational).

## Package layout

```
mwf/
  features.py            multimodal wavelet feature extraction (→ shared)
  feature_transform.py   hybrid transform T(x, k): ECG BioHash ‖ PPG IoM
  iom.py                 Index-of-Max hashing for the PPG block
  clean.py               NeuroKit ECG/PPG cleaning (default front-end)
  noise.py               AWGN injection for robustness runs
  pipeline.py            preprocess → extract → transform → group-aware CV (nested-CV tuning)
  classifiers.py         five reference classifiers (MLP/LR/SVM/DT/RF) + tuning grids
  dataset.py             MIMIC-100 / BIDMC / PTT-PPG loaders + Datasets adapter
  cross_session.py       cross-activity verification (enrol one condition, probe another)

  metrics.py             macro one-vs-rest accuracy/bal-acc/AUC/EER/P/R/F1/AP
  evaluation.py          metric aggregation + Nadeau-Bengio / bootstrap CIs
  operating_curves.py    DET/ROC/PR, CMC rank-k, operating points (Wilson CIs)
  verification.py        1:1 closed/open-set EER over CV folds
  scoring.py             cosine, subject centroids, z-norm, decidability

  cancelability.py       ISO/IEC 30136 renewability / diversity / Gomez-Barrero D_↔^sys
  non_invertibility.py   Wu-style 3-distribution report + Success-Attack-Rate
  stolen_token.py        worst-case (stolen-key) EER — the honest figure of merit
  significance.py        paired Nadeau-Bengio t + Benjamini-Hochberg FDR
  holdout.py             temporal (within-subject) & subject-disjoint splits

  cv_splits.py keystream.py rng.py stats_helpers.py constants.py
  batch_utils.py validation.py plots.py
scripts/run_experiment.py
tests/                   (per-module pytest suite)
```

## Usage

```bash
# from the aguilar2026features/ directory
uv sync                                   # materialise the pinned environment
pytest                                    # run the test suite

# experiments — set PYTHONHASHSEED for bit-for-bit reproducibility
PYTHONHASHSEED=42 python scripts/run_experiment.py -v          # id + verification, every dataset
PYTHONHASHSEED=42 python scripts/run_experiment.py --all -v    # full battery, every dataset
PYTHONHASHSEED=42 python scripts/run_experiment.py --datasets bidmc --all -v   # one dataset only
```

The default run evaluates **every dataset** (`mimic`, `bidmc`, `ptt`), sweeping
the three regimes over each cohort's usable DWT levels and reporting
identification + verification CV. Restrict with `--datasets`. Each analysis
block is opt-in (or all at once with `--all`, which also enables every dataset):

| Flag | Output |
|---|---|
| `--datasets {mimic,bidmc,ptt,all}` | which cohorts to evaluate (default `all`) |
| `--tune` | nested-CV hyperparameter tuning (unbiased; off → fixed reference configs) |
| `--significance` | paired classifier tests + BH-FDR (`significance.csv`) |
| `--cancelability-keys K` | ISO/IEC 30136 protocol over `K` keys (`cancelability.csv`) |
| `--non-invertibility` | Wu 3-distribution + SAR (`non_invertibility*.csv`) |
| `--stolen-token` | worst-case stolen-key EER (`stolen_token.csv`) |
| `--cross-activity` | enrol-one / probe-another EER on multi-activity cohorts (`cross_activity.csv`; PTT-PPG only) |
| `--holdout` / `--subject-holdout` | within-subject / unseen-subject holdout |
| `--det-plots` | render the figure suite under each dataset's `figures/` |

Other knobs: `--projection-ratio`, `--binarise`, `--regimes`, `--protocol`,
`--split-seeds`, `--holdout-fraction`, `--seed`. Per-row parallelism is
controlled by `AGUILAR_FEATURES_N_JOBS` / `AGUILAR_FEATURES_CV_N_JOBS`. Feature
levels are derived **per dataset** from its own segment length (1..6 at 125 Hz,
1..8 at 500 Hz), so no config bleeds across cohorts.

### Outputs

Each dataset writes to its own timestamped folder
`results/<Name>_<YYYY-MM-DD>_<HH-MM>/` (all datasets of one invocation share the
run tag), holding `metrics.csv`, `verification.csv`, `significance.csv`,
`cancelability.csv`, `non_invertibility.csv` (+ `_pools`), `stolen_token.csv`,
`cross_activity.csv` (PTT-PPG), `holdout.csv`, `subject_holdout.csv`, a
`figures/` directory, and a `run_manifest.json` (git SHA, full args, dependency
versions, `uv.lock` hash, dataset shape) tying every CSV to the exact code and
environment. A cross-dataset headline comparison
(`dataset_comparison_<run-tag>.csv`) lands in `results/shared/`.

Figures land under each dataset folder's `figures/`:

* **Recognition** — `det.png`, `roc.png`, `pr.png`, `regime_summary.png`,
  per-regime `scores_<regime>.png` and `clf_vs_features_<regime>.png`.
* **Cancelability / security** —
  `non_invertibility.png` (mated / non-mated / genuine-ref + SAR),
  `stolen_token_scores.png`.

## Statistical rigour

Identification/verification metrics use repeated `StratifiedGroupKFold` CV over
`temporal_block_groups` (no adjacent-segment leakage) with Nadeau-Bengio
corrected CIs; pooled-score EER, `D_↔^sys`, diversity and the leakage gap carry
percentile-bootstrap CIs; operating points carry Wilson CIs. `--tune` runs a
group-aware inner CV on each outer fold (nested CV), and `--significance`
FDR-controls the classifier comparison grid (Benjamini-Hochberg).

## Scope & limitations (read before citing numbers)

* **Session structure of the evaluation.** MIMIC-100 is essentially one
  continuous recording per subject. Group-aware temporal-block CV removes
  adjacent-segment leakage, but train and test still come from the **same
  acquisition session**, so the MIMIC-100 headline numbers are an **upper bound
  on within-session performance**. Two additions widen this: **BIDMC**
  replicates the protocol on an independent 53-subject cohort, and **PTT-PPG**
  supports `cross_session_verification` — enrol on one activity, verify probes
  from another (`sit`→`walk`/`run`). On PTT-PPG the within-activity EER (~0.09)
  rises to ~0.23 across activities, quantifying the **cross-condition**
  robustness gap with bootstrap CIs. Caveat: PTT-PPG's activities are recorded
  in the same visit, so this is *cross-activity / cross-condition* (physiological
  state change), **not** multi-day template ageing; a true multi-visit
  cross-session cohort remains the one outstanding external-validation gap.
* **Non-invertibility is a linear-attack bound.** The reconstructions in
  `mwf.non_invertibility` are the closed-form min-norm (ECG) /
  IoM-winner (PPG) inverses — a **lower bound on attacker capability**. A learning-
  or optimisation-based inversion (e.g. Mai et al. 2019, gradient inversion) can
  recover more; the leakage figures describe the linear family only. A token-aware
  IoM inversion still recovers the PPG *direction* with a correlation that grows
  with code length and window, so the leak cannot reach zero without destroying
  utility — the operative guarantees are unlinkability, magnitude/exact-preimage
  non-invertibility, and the key-independence of the IoM similarity.
* **Hyperparameters.** Fixed classifier configs are reference centres; report
  `--tune` (nested-CV) numbers when claiming one model beats another, or state
  that defaults were fixed a priori without data-driven selection on this cohort.
