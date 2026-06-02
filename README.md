# aguilar2026features

Cancelable **multimodal (ECG + PPG)** biometric identification where the
cancelable transform is applied to the **feature vector** — a token-keyed random
orthonormal projection (**BioHashing**) — rather than to the waveform.

It is the *feature-domain* sibling of `Aguilar2026Wavelet` (which protects the
signal and then extracts features). Here the order follows the canonical
cancelable-biometrics architecture:

```
Sensor → Preprocesamiento (AWGN→NeuroKit clean)        ← mwf.clean (ECG neurokit / PPG elgendi)
       → Extracción  (vector de características)        ← shared.features.wavelet (multimodal ECG+PPG)
       → Transformación cancelable T(x, k)  ← Token k   ← ECG: BioHashing · PPG: IoM hashing
       → Plantilla transformada
       → Base de datos / Comparador (coseno) → Decisión
```

## The transform module — `mwf.feature_transform`

The protected template is a **per-modality hybrid** of the concatenated ECG‖PPG
feature vector `x ∈ ℝ^d` under a user token `k` (SHA-256-seeded, `mwf.keystream`):

* **ECG block → BioHashing** (Teoh 2004). A `d_e × m` Gaussian is orthonormalised
  by QR into `R` with `R Rᵀ = I_m`; the block template is `R x_ecg` (optionally
  sign-binarised). `projection_ratio = m/d_e` (default `0.5`).
* **PPG block → Index-of-Max (IoM) hashing** (`mwf.iom`; Jin et al., *IEEE TIFS*
  2018). The token seeds `m` Gaussian projections; each keeps only the *index of
  the maximum* of `q` (`IOM_WINDOW`) responses — magnitudes are discarded. The
  one-hot code's inner product equals the IoM collision rate, so the existing
  cosine matcher and classifiers apply unchanged.

Each block is unit-L2-normalised, then concatenated.

Why this split — the PPG block is the one a linear inversion reconstructs at
`r≈0.78`, so it gets the strongly non-invertible IoM:

| Property | Mechanism |
|---|---|
| **Renewable / diverse** (ISO/IEC 30136) | one-bit token edit ⇒ independent `R` and IoM seeds ⇒ decorrelated template |
| **Non-invertible (PPG)** | IoM keeps only argmax indices ⇒ no linear pre-image and no magnitudes; the ~`√(m/d)` leak of a plain projection no longer applies |
| **Key-not-learnable** | IoM is a similarity-preserving LSH whose collision rate depends only on the *angle* between vectors, **not** the token ⇒ genuine pairs separate from impostors even under a *stolen* (shared) token, so the biometric — not the key — carries discriminability |

> **Privacy–utility caveat.** A best-effort, token-aware inversion of the IoM
> code still recovers the PPG feature *direction* with a correlation that grows
> with the code length `m` (`IOM_HASHES_RATIO`) and `q`. Angular-similarity
> preservation *is* directional information, so the leak cannot reach zero
> without destroying utility; the defaults sit in a regime where it stays below
> the linear baseline. The operative cancelability guarantees are unlinkability,
> magnitude/exact-preimage non-invertibility, and the key-independence above.

## Feature extraction — `mwf.features`

Per modality, `shared.features.wavelet.extract_wavelet_features` decomposes a
segment with the DWT (**bior3.3**, a linear-phase biorthogonal basis that keeps
the ECG/PPG fiducials aligned) and emits **13 descriptors** per subband (mean,
std, var, kurtosis, skewness, energy, entropy, max, min, median, iqr, range,
mad). ECG and PPG blocks are concatenated, so `d = 2 · 13 · (level + 1)`
(e.g. `130` at level 4).

## Key regimes (`mwf.pipeline.KeyMode`)

* `identity` — raw multimodal features (unprotected biometric ceiling).
* `single_key` — BioHashing with one shared token.
* `per_subject` — BioHashing with one token per subject (operational).

## Package layout

```
mwf/
  features.py           multimodal wavelet feature extraction (→ shared)
  feature_transform.py  hybrid cancelable transform T(x, k): ECG BioHash ‖ PPG IoM
  iom.py                Index-of-Max hashing for the PPG block
  inversion.py          best-effort projection inversion + leakage metrics
  cancelability.py      renewability / diversity / Gomez-Barrero D_↔^sys
  security.py           1-bit-token key sensitivity (BER + |corr|)
  pipeline.py           preprocess → extract → BioHash → CV
  clean.py              NeuroKit ECG/PPG cleaning (default front-end)
  plots.py              figure suite (DET/ROC/PR, score KDEs, metric summaries)
  verification.py       1:1 closed/open-set EER over CV folds
  metrics.py scoring.py operating_curves.py evaluation.py cv_splits.py
  dataset.py keystream.py rng.py noise.py
  stats_helpers.py classifiers.py constants.py
scripts/run_experiment.py
tests/
```

## Usage

```bash
# from the aguilar2026features/ directory
uv sync                                   # materialise the pinned environment
pytest                                    # run the test suite

# experiments
python scripts/run_experiment.py -v                       # id + verification
python scripts/run_experiment.py --all -v                 # + cancelability/keysens/inversion/DET
python scripts/run_experiment.py --max-subjects 8 --cv-folds 3 --feature-levels 4 -v
```

Outputs land under `results/`: CSVs (`metrics.csv`, `verification.csv`,
`cancelability.csv`, `key_sensitivity.csv`, `inversion.csv`, `stolen_token.csv`,
`timing.csv`, `holdout.csv`) and, under `results/figures/`, the figure suite:

* **Recognition** — `det.png`, `roc.png`, `pr.png`, `regime_summary.png`, and
  per-regime `scores_<regime>.png` / `clf_vs_features_<regime>.png`.
* **Cancelability** (IoM-specific) — `inversion_leakage.png` (IoM PPG leak vs the
  linear baseline), `stolen_token_scores.png` (worst-case genuine/impostor with
  the key neutralised), `key_sensitivity.png` (BER ≈ 0.5 avalanche + ≈ 0
  correlation under one-bit token edits).
