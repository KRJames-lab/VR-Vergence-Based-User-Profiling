# VR Vergence-Based User Profiling

The experimentation code for the paper [**"VR Vergence-Based User Profiling: Clustering Binocular Coordination Patterns for Adaptive Display Design"**][paper] (IDW 2026).

## Overview

VR head-mounted displays render depth through binocular disparity on a fixed
screen, decoupling **vergence** from **accommodation**. How well a user tolerates
the resulting vergence–accommodation conflict (VAC) depends on their individual
**binocular coordination**, which varies widely across people.

We treat each participant's vergence response to repeated depth changes as a
**control-theoretic step response**, extract biomarkers from a single 60-second
task, and cluster **407 GazeBaseVR participants** into interpretable profiles. We
then test whether those vergence-derived profiles (1) generalize to independent
gaze tasks and (2) recur across sessions — a software-only basis for
profile-adaptive VR rendering.

- **Input** (per participant, session 1 VRG task): binocular gaze at 250 Hz; a
  central target alternates between near (0.35 m) and far (0.44 m) depth over
  ~60 s (23 depth transitions).
- **Output**: 6 step-response biomarkers → an 11-feature vector → a K=3 cluster
  label, plus cross-task and session-stability evidence.

> _Data source: **GazeBaseVR** (Lohr et al., 2023; [Figshare][gazebasevr-data]);
> not included in this repo. We use the **first recording round (N = 407 unique
> participants)** only._

## Repository Structure

```
.
├── src/
│   ├── pilot_biomarker_extraction.py   # 1. VRG step response → 6 biomarkers / participant
│   ├── pilot_clustering.py             # 2. 11-feature K=3 clustering (KMeans/GMM/Agg)
│   ├── pilot_cross_validation.py       # 3. cross-task validation on PUR & RAN
│   ├── step6_session_stability.py      # 4. session-to-session reproducibility (S1 → S2)
│   ├── make_fig_cluster_stepresponse.py# Fig 1: cluster-mean step responses
│   ├── make_fig_clusters.py            # Fig 2: PCA scatter + conv/div boxplots (+ Table 2)
│   └── make_fig_confusion.py           # Fig 3: S1 → S2 cluster confusion matrix
├── datasets/            # (not tracked) GazeBaseVR VRG/PUR/RAN .csv recordings
├── eda_results/         # (not tracked) intermediate CSVs + figures
└── requirements.txt
```

## Requirements

- **Python 3.10+**
- Core libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Usage

Place the GazeBaseVR recordings under
`datasets/gazebasevr/extracted_full/data/` (file names follow the dataset's
`S_<pid>_S<1|2>_<task>.csv` convention), then run the stages in order **from the
repo root**:

```bash
python src/pilot_biomarker_extraction.py   # 1. extract biomarkers   → eda_results/.../pilot_biomarkers.csv
python src/pilot_clustering.py             # 2. K=3 clustering        → pilot_biomarkers_with_clusters.csv
python src/pilot_cross_validation.py       # 3. PUR & RAN validation
python src/step6_session_stability.py      # 4. S1 → S2 stability
python src/make_fig_clusters.py            #    paper figures + Table 2 values
```

Each script writes intermediate CSVs and figures under
`eda_results/eye_tracking/full_407/`; stages 2–4 consume the CSV produced by the
stage before them.

## Method

Vergence is the horizontal angular difference between the two eyes,
`θ_v = θ_L − θ_R`. Preprocessing follows clinical-standard vergence analysis:

| Step | Operation |
| ---- | --------- |
| Outlier / artifact removal | drop samples beyond ±50° or with velocity > 1°/sample (1.4% of VRG samples), then linear interpolation |
| Low-pass filter | zero-phase **2nd-order Butterworth at 40 Hz** |
| Velocity | **two-point central difference** (Bahill et al., 1982) |
| Ideal vergence | per-participant IPD → depth-dependent target angle (~10°/8°) |

Each of the 23 depth transitions is treated as a **step input**. From the step
responses we extract **6 biomarkers**:

| Biomarker | Definition | Unit |
| --------- | ---------- |:----:|
| Latency | First 2σ deviation of the **raw** vergence after a transition | ms |
| Speed | Mean rate from onset to 90% of the step | deg/s |
| Overshoot | Peak excess beyond target / step size | % |
| Settling time | Time to stay within ideal ±1° for 200 ms | ms |
| SS error | Mean absolute tracking error in steady state (≥ 1.5 s) | deg |
| Jitter | Steady-state standard deviation | deg |

**Clustering input = 11 features**: latency, speed, overshoot, settling time and
SS error are split into convergence / divergence (= 10) plus a single jitter term.
(Peak velocity is computed but excluded from clustering.)

## Results

### Three binocular-coordination profiles (K=3, silhouette = 0.522)

Signed steady-state error separates the clusters into clinically meaningful
profiles:

| Cluster | Profile | N (%) | SS error | MAE | Jitter |
| ------- | ------- |:-----:|:--------:|:---:|:------:|
| **C0** | Normal           | 263 (65%) | +1.0° | 1.77° | 0.50° |
| **C1** | Over-converging   | 118 (29%) | +3.0° | 2.98° | 0.43° |
| **C2** | Under-converging  | 26 (6%)   | −4.9° | 5.00° | 0.62° |

The smallest cluster (C2, ~6%) is the least stable and is consistent with the
prevalence and characteristics of clinical convergence insufficiency.

### Cross-task generalization

The VRG-derived clusters separate gaze quality in **independent** smooth-pursuit
(PUR) and random-saccade (RAN) tasks: **10 of 13** metrics differ significantly
(PUR 5/6, RAN 5/7). For PUR tracking error, C2 is ~1.9× worse than C0
(3.22° vs 1.69°).

### Session stability (S1 → S2)

Reassigning session-2 profiles to the nearest session-1 cluster gives
**69.8% accuracy, Cohen's κ = 0.356 (fair)**. The dominant Normal profile is the
most reproducible (retention **81.7%**), vs 52.5% (C1) and 26.9% (C2).

> [!NOTE]
> This repository contains only the experimentation code. For the full
> methodology and detailed results, please refer to the [paper].

## License

Released under the [MIT License](LICENSE).

<!-- TODO: replace # with the published paper URL once available. -->

[paper]: #
[gazebasevr-data]: https://figshare.com/articles/dataset/GazeBaseVR_Data_Repository/21308391
