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
