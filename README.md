# Aguilar2026Features — Cancelable Multimodal Biometrics via Wavelet Features

Cancelable biometric authentication system that applies token-keyed template
protection to **multimodal ECG + PPG wavelet features**.  The cancelable
transform operates on the **feature vector** (not the raw signal), making
revocation cheap and the feature extraction token-independent.

---

## High-level architecture

```
ECG waveform ─┐
              ├─► Physiological cleaning ─► Wavelet statistical
PPG waveform ─┘   (NeuroKit2 / Elgendi)    feature extraction
                                            │
                                            │  x ∈ ℝ^d  (ECG‖PPG concat)
                                            ▼
                              ┌─────────────────────────┐
                              │  Pre-projection z-score  │  (FeatureScaler, fit
                              │  standardisation         │   on enrolment cohort)
                              └────────────┬────────────┘
                                           │
                           ┌───────────────┴───────────────┐
                           │                               │
                    ECG block  x_E                  PPG block  x_P
                    (d/2 dims)                      (d/2 dims)
                           │                               │
               token + "::ECG"                 token + "::PPG"
                           │                               │
                    BioHashing                   Index-of-Max (IoM)
              R ∈ ℝ^{m×d/2}  (QR)             argmax over q Gaussians
              y_E = R x_E / ‖R x_E‖           y_P = one-hot(m) / ‖y_P‖
                           │                               │
                           └───────────────┬───────────────┘
                                           │
                                   T(x, k) ∈ ℝ^{m_E + m_P}
                              (hybrid protected template)
                                           │
                               ┌───────────┴───────────┐
                               │   Template database   │
                               └───────────────────────┘
                                           │
                              1:1 cosine verification / 1:N identification
```

The two modalities share a single user **token** `k` (an arbitrary string).
Salting — `k + "::ECG"` and `k + "::PPP"` — keeps the per-modality random
material decorrelated while a single token drives the whole template.

---

## Cancelable transform in detail

### Token-to-projection pipeline (`mwf/keystream.py`, `mwf/feature_transform.py`)

```
token  ──► SHA-256(token + salt) ──► SeedSequence ──► Generator
                                                           │
                              draw d × m Gaussian  ◄───────┘
                                       │
                                    QR(G) ──► R ∈ ℝ^{m×d}  (orthonormal rows)
```

SHA-256 provides **avalanche sensitivity**: a one-bit change in the token
yields a fully uncorrelated projection, ensuring renewability and cross-token
diversity (ISO/IEC 30136).  Little-endian byte order is enforced explicitly for
bit-for-bit reproducibility across platforms.

### ECG block — BioHashing (real-valued orthonormal projection)

```
y_E = R_E · x_E,   R_E ∈ ℝ^{m_E × d/2},   R_E R_Eᵀ = I_{m_E}
```

With `m_E = round(ratio × d/2)` (default ratio = 0.5), the protected template
retains half the original dimensions.  Because `m_E < d/2`, the projection is
many-to-one: an adversary with both `R_E` and `y_E` can only recover
`Rᵀy_E` (the min-norm pre-image), which differs from `x_E` by its null-space
component — leakage bounded by `√(m/d)`.

### PPG block — Index-of-Max hashing (`mwf/iom.py`)

```
For i = 1…m:
    draw window q Gaussian projections {g_{i,j}} seeded by token + "::PPG"
    h_i(x_P) = argmax_j  g_{i,j}ᵀ x_P          ∈ {0, …, q−1}

y_P = [one-hot(h_1) ‖ … ‖ one-hot(h_m)]  ∈ {0,1}^{m·q}
```

IoM discards the magnitude of each projection and keeps only the winning index.
This yields two properties the plain projection does not have:

| Property | BioHashing | IoM |
|---|---|---|
| Non-invertibility | partial (null-space gap) | strong (no linear pre-image) |
| Similarity preservation | cosine (angle + magnitude) | angle only (key-independent LSH) |
| Stolen-token leakage | ~√(m/d) from Rᵀy | direction averaged over winners only |

The inner product between two IoM codes counts hash collisions; collision
probability depends only on the **angle** between the original feature vectors,
not on the token — so biometric discriminability survives even in the
stolen-token scenario.

### Pre-projection standardisation (`mwf/feature_transform.py` — `FeatureScaler`)

Wavelet descriptors live on wildly different scales (sub-band energy can exceed
entropy by several orders of magnitude).  Without equalisation, a random
projection is dominated by the high-variance descriptors and buries the
discriminative low-variance ones.  A per-feature z-score — fit on the
**enrolment cohort only** within each CV fold — restores equal contribution
before the projection.  The scaler statistics are non-secret (same status as
the wavelet basis).

---

## Feature extraction (`mwf/features.py`)

```
segment (ECG, PPG, L samples)
      │
      ├─ DWT level-4, wavelet bior3.3 ──► 5 subbands × 13 statistics = 65 ECG descriptors
      │                                   [mean, std, var, kurtosis, skewness, energy,
      │                                    entropy, max, min, median, IQR, range, MAD]
      └─ DWT level-4, wavelet bior3.3 ──► 65 PPG descriptors

feature vector x = [ECG descriptors ‖ PPG descriptors] ∈ ℝ^130
```

`bior3.3` (biorthogonal 3.3) has symmetric, linear-phase reconstruction filters
that do not shift ECG/PPG fiducial points, avoiding the small phase distortion
that orthogonal Daubechies wavelets introduce in morphological statistics.

The decomposition level is a configurable hyperparameter (`PipelineConfig.feature_level`,
default 4).  `max_feature_level(segment_length, wavelet)` enforces the maximum
usable depth for a given segment.

---

## Datasets (`mwf/dataset.py`)

| Dataset | Subjects | Segment | fs | Notes |
|---|---|---|---|---|
| MIMIC-100 | 100 | 6 s | 125 Hz | single-session intra-session baseline |
| BIDMC | 53 | 6 s | 125 Hz | external cross-cohort validation |
| PTT-PPG | 22 | 6 s | 500 Hz | three activities: sit / walk / run |

Each loader returns a `BiometricSegments` frozen dataclass:

```python
@dataclass(frozen=True)
class BiometricSegments:
    ecg:            NDArray  # (N, L)
    ppg:            NDArray  # (N, L)
    labels:         NDArray  # (N,) integer subject IDs
    sampling_rate:  int
```

---

## Evaluation protocols (`mwf/`)

### Identification — `pipeline.py`

Rank-k accuracy (top-1, top-5, top-10, top-20) under three token regimes:

| `KeyMode` | Token assignment | Evaluates |
|---|---|---|
| `IDENTITY` | no transform (raw features) | baseline discriminability |
| `SINGLE_KEY` | one shared token for all subjects | worst-case privacy |
| `PER_SUBJECT` | one unique token per subject | operational use case |

Cross-validation uses `StratifiedGroupKFold` / `GroupKFold` / `LeaveOneGroupOut`
with group = subject ID to prevent identity leakage across folds.  Uncertainty
is quantified with Nadeau–Bengio corrected 95 % CIs (accounts for train-set
overlap in repeated CV).

### Verification 1:1 — `verification.py`

Closed-set and open-set EER, decidability index, and DET/ROC operating points.
Queries are scored against per-subject enrolment centroids using cosine similarity.

### Cancelability — `cancelability.py` (ISO/IEC 30136)

| Metric | Definition | Pass threshold |
|---|---|---|
| Renewability ratio | `EER(cross-key genuine) / EER(same-key genuine)` | ≤ 0.05 |
| Diversity | per-subject template correlation across different tokens | ≤ 1.5 × chance |
| Unlinkability | Gomez-Barrero D↔sys (global system-level score) | ≤ 0.05 |

Evaluated with 32 independently derived random tokens per subject.

### Non-invertibility — `non_invertibility.py`

Wu-style reporting using Pearson |r| between the reconstructed and original
feature vectors:

- **mated**: `|corr(x̂_ECG, x_ECG)|` — ECG min-norm pre-image vs original
- **non-mated**: `|corr(x̂_ECG, x'_ECG)|` — reconstruction vs other subject
- **genuine_ref**: `|corr(x_a, x_b)|` — two genuine samples of the same subject
- **leakage_gap** = mated − non-mated (figure of merit; ideal ≈ 0)
- **SAR** (Success Attack Rate): re-protected reconstruction accepted at EER threshold

PPG inversion uses a best-effort direction sum over IoM winning vectors; no
magnitude is available, so the recovered direction only partially reconstructs
`x_P`.

### Stolen-token attack — `stolen_token.py`

Worst-case scenario: the adversary knows the victim's token.  For each victim:

1. All segments (victim + impostors) are re-projected under the **victim's token**.
2. Victim's own segments form the genuine pool; others form the impostor pool.
3. Z-norm uses a held-out disjoint cohort to avoid optimistic score normalisation.

Reports EER, decidability, and operating points.  This is the floor on
verification performance when privacy has been fully compromised.

### Cross-activity — `cross_session.py`

Enrol on one activity / session, probe on another.  Applied to PTT-PPG
(sit → walk, sit → run, walk → run, and reverses) to quantify intra-subject
template stability across physiological conditions.

---

## Module map

```
mwf/
├── constants.py          PipelineConfig, all numeric defaults
├── keystream.py          SHA-256 → SeedSequence → numpy Generator
├── features.py           DWT statistical feature extraction (ECG + PPG)
├── feature_transform.py  Hybrid ECG-BioHash / PPG-IoM cancelable transform
├── iom.py                Index-of-Max hashing implementation
├── dataset.py            Dataset loaders (MIMIC-100, BIDMC, PTT-PPG)
├── pipeline.py           End-to-end identification CV pipeline
├── verification.py       1:1 verification CV
├── cancelability.py      ISO/IEC 30136 cancelability evaluation
├── non_invertibility.py  Wu-style non-invertibility reporting
├── stolen_token.py       Stolen-token worst-case protocol
├── cross_session.py      Cross-activity / cross-session verification
├── evaluation.py         CV aggregation, Nadeau-Bengio CI, MetricSummary
├── metrics.py            Classification metrics (EER, AUC, macro-OVR)
├── classifiers.py        Five reference classifiers with CV tuning grids
├── scoring.py            Cosine scoring utilities
├── operating_curves.py   DET / ROC / PR curve computation
├── plots.py              Matplotlib figure helpers
├── noise.py              AWGN injection for SNR-robustness experiments
├── clean.py              NeuroKit2 / Elgendi signal cleaning wrappers
├── cv_splits.py          Group-aware CV splitter factories
├── batch_utils.py        Parallelised batch processing helpers
├── stats_helpers.py      Bootstrap CI, Nadeau-Bengio correction
├── significance.py       Statistical significance tests
├── holdout.py            Held-out cohort management for honest z-norm
├── progress.py           tqdm progress wrappers
├── rng.py                Global RNG seed management
└── validation.py         Input validators (shape, dtype)

scripts/
├── run_experiment.py     Full Q1 evaluation battery (identification + verification
│                         + cancelability + non-invertibility + stolen-token
│                         + cross-activity + DET curves), multi-dataset
└── probe_single_key.py   Hyperparameter probe: how to reach accuracy > 0.90
                          under the shared-token regime (standardise × ratio sweep)
```

---

## Running experiments

### Quick smoke test (8 subjects, MIMIC-100 only)

```bash
python scripts/run_experiment.py \
    --datasets mimic100 \
    --max-subjects 8 \
    --cv-folds 3 \
    --feature-levels 4
```

### Full Q1 evaluation battery

```bash
python scripts/run_experiment.py --all
```

`--all` runs:
- identification CV + verification CV on MIMIC-100, BIDMC, and PTT-PPG
- cancelability (32 keys, ISO/IEC 30136)
- non-invertibility (Wu-style + SAR)
- stolen-token worst-case
- cross-activity (PTT-PPG sit / walk / run)
- DET / ROC curves

Results land in `results/<Dataset>_<date>_<hour>/` and a cross-dataset headline
summary in `results/shared/`.

### Shared-token accuracy probe

```bash
python scripts/probe_single_key.py
```

Sweeps `(standardise, projection_ratio)` combinations and writes
`results/probe_single_key.csv`.

### Parallelism

The identification/verification CV respects the `CV_N_JOBS` environment
variable (default `-1`, all cores).  Set `OMP_NUM_THREADS=1` when parallelising
at the Python level to avoid nested thread pools and ensure bit-reproducible
BLAS results.

```bash
OMP_NUM_THREADS=1 CV_N_JOBS=8 python scripts/run_experiment.py --all
```

---

## Key design decisions

**Feature-domain transform.** The cancelable transform acts on the feature
vector, not the waveform.  This keeps feature extraction token-independent and
allows arbitrary waveform preprocessing without touching the protection layer.

**Hybrid per-modality protection.** A plain BioHashing projection applied to
the full ECG‖PPG vector leaks ~√(m/d) of both modalities.  Routing the PPG
block through IoM replaces that linear leakage path with an angle-only
similarity measure that has no tractable linear pre-image.  ECG retains
real-valued BioHashing because its morphological richness benefits from the
full projection resolution.

**Equal-weight fusion.** Both protected blocks are unit-L2-normalised before
concatenation, so cosine similarity on the hybrid template weights ECG and PPG
equally regardless of their raw dimensionalities.

**Group-aware CV.** Segments are temporally correlated within a subject.
Using subject ID as the group variable in stratified-group k-fold prevents any
subject's segments from appearing in both train and test within the same fold,
eliminating identity leakage.

**Honest z-norm.** Score normalisation in the stolen-token and cross-activity
protocols uses a held-out cohort (subjects disjoint from the victim pool),
matching operational deployment where impostor statistics are estimated from an
independent reference set.
