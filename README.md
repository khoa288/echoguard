# EchoGuard — Energy-Aware Always-On Emergency Sound Detection (3-Stage Cascade)

EchoGuard is an on-device, energy-aware emergency sound detection pipeline built on **ESC-50** and evaluated in **continuous synthetic soundscapes**. It implements a strict cascade:

- **Stage 1 (DSP gate)**: ultra-cheap event trigger (energy/flux/band changes) that opens a 5s compute window  
- **Stage 2 (binary danger/safe)**: lightweight classifier to filter safe events and decide escalation  
- **Stage 3 (50-class ESC)**: EfficientAT **DyMN** classifier; emergency decision is derived by summing probabilities over a fixed emergency class set

This repo contains the full notebooks, training scripts for Stage-2, and the consolidated result table.

**Project code + experiments** are designed for reproducibility: fixed scenario definitions, a consistent decision grid, and explicit compute accounting (MACs/s, Stage3 duty, EUI).

---

## Repository Contents

- **`EchoGuard_Full_Pipeline.ipynb`**  
  End-to-end pipeline: soundscape generation, Stage1/Stage2/Stage3 cascades, evaluation on 6 scenarios, compute accounting, and export of the final table (`results_summary.csv`).

- **`EchoGuard_Stage2.ipynb`**  
  Stage-2 experiments (binary danger/safe): training/eval of deep models (e.g., BC-ResNet variants) and classical ML baselines.

- **`train_esc50_binary.py`**  
  Train Stage-2 **deep** binary classifiers on ESC-50 folds (log-mel frontend + SpecAugment + SGD schedule, etc.).

- **`train_esc50_binary_ml.py`**  
  Train Stage-2 **classical ML** baselines (Logistic Regression / Linear SVM) using pooled log-mel features.

- **`results_summary.csv`**  
  Consolidated results table (**125 rows**) covering:
  - **120 soundscape runs**: 6 scenarios × {S3, S1+S3, S2+S3, S1+S2+S3} over all stage variants
  - **5 single-clip runs**: clip-level evaluation for sanity checking

## Pretrained Checkpoints (Required)

To run `EchoGuard_Full_Pipeline.ipynb`, download **all checkpoints** from the latest GitHub Release and place them **in the same folder** (no subdirectories).

### Included models
**Stage 2 (binary danger/safe):**
- `bcresnet1.pt`  — BC-ResNet (τ = 1)
- `bcresnet2.pt`  — BC-ResNet (τ = 2)
- `bcresnet3.pt`  — BC-ResNet (τ = 3)
- `bcresnet8.pt`  — BC-ResNet (τ = 8)

**Stage 3 (50-class ESC, EfficientAT):**
- `dymn10_as.pt` — DyMN-10 pretrained on AudioSet, fine-tuned on ESC-50 (fold 1)
- `mn04_as.pt`  — MobileNet (mn04) pretrained on AudioSet, fine-tuned on ESC-50 (fold 1)

The notebook will load checkpoints by filename depending on the selected
cascade configuration.

## EchoGuard-Bandit

EchoGuard-Bandit reframes the fixed cascade as **safety-constrained contextual
selection among complete inference paths**. It does not treat S1, S2, and S3 as
interchangeable classifiers. The initial action registry contains A0–A14:

- stop safely after S1 context;
- run one of BC1/BC2/BC3/BC8 and stop;
- run MN04 or DyMN10 directly;
- run a BC-ResNet and conditionally escalate to MN04 or DyMN10.

The model roles are deliberately strict:

| Stage | Models | Output |
|---|---|---|
| S2 | `bcresnet1.pt`, `bcresnet2.pt`, `bcresnet3.pt`, `bcresnet8.pt` | Binary danger probability |
| S3 | `mn04_as.pt`, `dymn10_as.pt` | 50 ESC-50 probabilities, aggregated into emergency probability |

`mn04_as.pt` is **not** a released binary Stage-2 model.

### Package layout

```text
echoguard/
  labels.py                  # canonical emergency categories
  stage1.py                  # cheap DSP context
  costs.py                   # versioned primitive MAC costs
  models/stage2.py           # BC checkpoint identities/adapters
  models/stage3.py           # MN04/DyMN10 expert summaries
  bandit/
    arms.py                  # deterministic A0-A14 paths
    config.py                # versioned run configuration
    schema.py                # counterfactual log contract
    features.py              # leakage-safe causal context
    counterfactual_logger.py # all-model log construction
    outcomes.py              # path outcomes and event timing
    oracle.py                # cheapest-correct upper bound
    policies.py              # fixed, MAB, and LinUCB policies
    safety.py                # conservative and calibrated shields
    eval_policy.py           # chronological replay
```

### Counterfactual workflow

The expensive pipeline pass must produce one primitive row per 0.5-second
decision window containing S1 context, all four BC danger probabilities, and
both Stage-3 expert summaries. It must run each checkpoint once per window.
Action outcomes are then computed without rerunning audio models:

```bash
python scripts/01_build_counterfactual_log.py \
  --input path/to/primitive_window_log.parquet

python scripts/02_compute_oracle_policy.py \
  --log bandit_logs/window_counterfactuals.parquet

python scripts/03_train_contextual_policy.py \
  --log bandit_logs/window_counterfactuals.parquet

python scripts/04_eval_bandit_policies.py \
  --log bandit_logs/window_counterfactuals.parquet

python scripts/05_plot_bandit_results.py
```

Parquet support requires `pyarrow`; plotting requires `matplotlib`:

```bash
python -m pip install -e ".[dev,parquet,plots]"
```

Full logs and generated results are ignored by Git. Small fixtures and selected
summary tables can be committed intentionally.

### Leakage and evaluation rules

- Level-1 decisions use only S1 context and causal history.
- S2 probabilities become available only after selecting a BC model.
- S3 confidence is unavailable until an expert has been invoked.
- Scenario names and ground truth are never policy inputs.
- Rolling features reset at stream boundaries and use only prior windows.
- Train/calibration/test partitions must be grouped by stream or scenario seed,
  not randomly split by adjacent windows.
- Time-to-detect is computed by chronological event replay, not independently
  per row.

The scenario-specific baseline and ground-truth oracle are non-deployable upper
bounds. Conservative replay and calibration shields are research mechanisms,
not guarantees of real-world emergency safety.

### Experiments

Reproducible experiment definitions are under `configs/bandit/`:

- `six_scenarios.json`: independent quiet, sleep, busy, kitchen, outdoor, and commute streams;
- `nonstationary.json`: a concatenated stream without exposing scenario labels;
- `ablations.json`: BC/expert portfolio and safety-layer ablations.

Primary reporting should use compute at matched recall, S3 duty, missed events,
false alarms per hour, mean and tail TTD, and regret versus the oracle. ESC-50
and synthetic soundscapes remain important limitations; the package does not
claim deployment readiness or state-of-the-art emergency detection.

## ESC-50 Dataset Setup

EchoGuard expects the **ESC-50 dataset resampled to 32 kHz**. Follow the instructions in the [PaSST](https://github.com/kkoutini/PaSST/tree/main/esc50) repository to get the ESC50 dataset.

You should end up with a folder `esc50` containing the two folders:

* `meta`: contains `meta.csv`
* `audio_32k`: contains all .wav files

Then update the path in the notebook:

```python
ESC50_32K_ROOT = "path/to/esc50"
```
