# Time-Aware Multimodal Early Warning of Neonatal Bradycardia — From Infant Heart Rate and Respiration Signals

## Overview

This repository contains the complete experimental pipeline for a **Master's in Artificial Intelligence capstone project** focused on **early prediction of neonatal bradycardia events** using **multimodal physiological time-series data**.

The system predicts whether a **bradycardia event will occur in the near future** based on recent **heart rate (HR)** and **respiration (Resp)** signals, while explicitly addressing key real-world challenges:

- Extreme class imbalance
- Temporal dependence and data leakage
- Inter-subject variability
- Clinically meaningful evaluation

All experiments are fully automated and reproducible, producing trained models, predictions, diagnostics, plots, and statistical significance tests.

**Core prediction task:**  
Given a **120-second observation window**, predict whether a bradycardia event will occur in the **next 60 seconds**, after an optional lead time.

---

## Research Contributions

- **Time-aware window sampling** with dense sampling near events and coarse sampling elsewhere
- **Distance-weighted loss** that reduces false-positive penalties near true events
- **Hybrid evaluation strategy** combining population-level learning with subject-specific temporal adaptation
- Unified comparison of **deep sequence models** and **feature-engineered traditional ML models**
- Automated **learning-curve diagnostics** and **overfitting analysis**
- Paired **statistical significance testing** across evaluation strategies

---

## Models Implemented

### Deep Learning

- **CNN-BiLSTM with Attention**
- **Temporal Convolutional Network (TCN)**
  - Dilated causal convolutions
  - Residual connections
  - Combined global state (GAP) + last-timestep representation

### Traditional Machine Learning

- **XGBoost** (dynamic `scale_pos_weight`, optional GPU acceleration)
- **Random Forest**
- **Logistic Regression**

Traditional models use a **fully vectorized feature extractor** including:

- Statistical descriptors (HR + Resp)
- Clinical HR features (rapid decelerations, time under threshold)
- Trend estimation
- HR–Resp coupling
- Frequency-domain HRV features (LF, HF, LF/HF)

---

## Evaluation Strategies

Three complementary strategies are used to assess generalization:

| Strategy     | Description                                                                         |
| ------------ | ----------------------------------------------------------------------------------- |
| **LOSO**     | Leave-One-Subject-Out (cross-subject generalization)                                |
| **Temporal** | Chronological train/val/test split within subject, with buffer zones               |
| **Hybrid**   | Train on other infants + early "warm-up" data from target infant; validate/test later |

All strategies enforce **strict temporal separation** to prevent leakage.

---

## Dataset & Data Requirements

The code expects a **WFDB-style dataset**, with the following per infant:

- **ECG:**
  - `<infant_id>_ecg.hea` / `.dat`
- **Respiration:**
  - `<infant_id>_resp.hea` / `.dat`
- **Annotations:**
  - `atr`: bradycardia event times
  - `qrsc`: R-peak locations used to compute RR intervals -> HR

The project was designed and run in **Google Colab**, copying data from Google Drive to local runtime storage for speed.

---

## Setup

### 1) Install dependencies
```bash
pip install wfdb xgboost
```

Additional libraries used:

- numpy, pandas, scipy
- scikit-learn
- matplotlib, seaborn
- torch (PyTorch)

---

### 2) Configure paths

Edit the `Config` dataclass:
```python
DRIVE_PATH   = '/content/drive/MyDrive/picsdb'
LOCAL_PATH   = '/content/picsdb_local'
RESULTS_BASE = '/content/drive/MyDrive/capstone_results'
```

If not using Colab/Drive, point `LOCAL_PATH` directly to your dataset and choose a local `RESULTS_BASE`.

---

## Method Overview

### Windowing & Labeling

- Signals are resampled to a uniform grid (`FS_GRID = 2 Hz`)
- Each sample is a **120-second window** with 2 channels: HR and Resp
- A window is labeled **positive** if a bradycardia event occurs within:
  - Start: `window_end + LEAD_TIME`
  - Duration: `HORIZON = 60s`

---

### Dynamic Sampling Near Events

To mitigate severe imbalance:

- **Dense stride (2s)** near events within `POS_REGION_RADIUS`
- **Coarse stride (10s)** elsewhere
- Each window's **distance to nearest event** is computed
- Windows are retained if:
  - Inside dense region, or
  - Aligned with the coarse grid

---

### Time-Aware Loss Weighting

For deep models, per-sample weights combine:

- **Class weighting** (positives up-weighted, capped)
- **Time-aware FP de-emphasis** for negatives near events:
  - Minimum penalty at distance 0 (`MIN_FP_WEIGHT`)
  - Linear ramp to full penalty at `POS_REGION_RADIUS`

This is combined with a **WeightedRandomSampler** targeting a desired positive fraction per epoch.

---

## Training Details

- Gradient clipping
- Early stopping with patience
- Mixed-precision training (AMP)
- Learning-rate scheduling (`ReduceLROnPlateau`)
- Deterministic seeding (including DataLoader workers)
- Automatic checkpointing and resume support (`progress.json`)

---

## Running Experiments

This project is provided as a single Jupyter Notebook (`.ipynb`). Open it in Google Colab and run the cells top-to-bottom.

The pipeline performs:

1. Data processing and caching
2. Training across all models and strategies
3. Evaluation, diagnostics, and plotting
4. Statistical significance testing

> **Note:** Deep models are GPU-intensive. XGBoost may use GPU if available. Use CPU for ML.

---

## Outputs

All results are stored under a single experiment directory.

**Important:** The pipeline writes **strategy-specific** subfolders for models and predictions (LOSO / Temporal / Hybrid), matching the code.
```
capstone_results/
└── Merged_Optimized_Diagnostics/
    ├── progress.json
    ├── all_results.csv
    ├── summary_stats.csv
    ├── cnn_learning_diagnostics.csv
    ├── traditional_learning_diagnostics.csv
    ├── figures/
    │   ├── learning_curves/
    │   ├── learning_curves_traditional/
    │   ├── aggregate/
    │   ├── feature_importance.png
    │   └── overfitting_analysis.png
    ├── data/
    │   ├── infant_data_cache_time_aware.npz
    │   └── infant_metadata_time_aware.csv
    ├── loso/
    │   ├── models/
    │   └── predictions/
    ├── temporal/
    │   ├── models/
    │   └── predictions/
    └── hybrid/
        ├── models/
        └── predictions/
```

Generated artifacts include:

- Per-subject prediction files (`.npz`)
- Learning curves with automated fit diagnosis
- Aggregate learning curves (deep models)
- Overfitting analysis plots (train vs val AUROC)
- Global feature importance (Random Forest)
- Paired Wilcoxon signed-rank tests

---

## Evaluation Metrics

- **AUROC**, **AUPRC**, **Accuracy**
- Sensitivity at fixed specificity (90%, 95%)
- Train–validation AUROC gaps for generalization analysis

These metrics are chosen to better reflect **clinical tradeoffs**, not just raw discrimination.

---

## Reproducibility

The code enforces reproducibility via:

- Fixed random seeds (`random`, `numpy`, `torch`)
- Deterministic cuDNN settings when applicable
- Cached intermediate datasets
- Checkpoint-based resume system (`progress.json`)

---

## Known Assumptions & Notes

- Dataset naming must follow the loader's conventions:
  - ECG: `<infant_id>_ecg`
  - Resp: `<infant_id>_resp`
- Explicit buffer zones prevent temporal leakage in Temporal/Hybrid strategies
- Some runs are skipped if insufficient positives or test samples exist (and are marked complete to avoid blocking long batch runs)
