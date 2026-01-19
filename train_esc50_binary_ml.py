# train_esc50_binary_ml.py
# Classical ML baselines for ESC-50 Danger/Safe (32k preprocessed)
# Features: log-mel (40 bins) -> mean+std pooling over time (80-dim)
# Models: Logistic Regression, Linear SVM (calibrated)
#
# Usage:
#   python train_esc50_binary_ml.py --esc50_root "$ESC50_ROOT" --output_dir runs_ml \
#     --model lr --test_fold 1 --seed 0 --wandb_mode online
#
# Optional:
#   --model svm
#   --C_list 0.1 1 3 10 (grid on val; best by AUROC)

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List

import torch
import torchaudio
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib

try:
    import wandb
except Exception:
    wandb = None


DANGER_CATEGORIES = {
    "siren",
    "car_horn",
    "glass_breaking",
    "thunderstorm",
    "crying_baby",
    "dog",
    "door_wood_knock",
    "clock_alarm",
}


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_esc50_metadata(esc50_root: str) -> pd.DataFrame:
    csv_path = os.path.join(esc50_root, "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_root, "audio_32k")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"esc50.csv not found: {csv_path}")
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"audio_32k not found: {audio_dir}")
    df = pd.read_csv(csv_path)
    for col in ["filename", "fold", "category"]:
        if col not in df.columns:
            raise ValueError(f"esc50.csv missing column: {col}")
    df = df.copy()
    df["path"] = df["filename"].astype(str).apply(lambda fn: os.path.join(audio_dir, fn))
    miss = df.loc[~df["path"].apply(os.path.isfile)]
    if len(miss) > 0:
        raise FileNotFoundError(f"Missing audio file example: {miss.iloc[0]['path']} (count={len(miss)})")
    df["fold"] = df["fold"].astype(int)
    df["label"] = df["category"].astype(str).apply(lambda c: 1 if c in DANGER_CATEGORIES else 0).astype(int)
    return df


class LogMelStatsExtractor:
    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 40,
        win_ms: float = 30.0,
        hop_ms: float = 10.0,
        n_fft: int = 1024,
        clip_seconds: float = 5.0,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        cmvn: bool = True,
    ):
        self.sr = int(sample_rate)
        self.clip_len = int(round(self.sr * clip_seconds))
        self.cmvn = bool(cmvn)
        win_length = int(round(self.sr * win_ms / 1000.0))
        hop_length = int(round(self.sr * hop_ms / 1000.0))
        fmax = float(self.sr / 2.0 if fmax is None else fmax)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=int(n_fft),
            win_length=win_length,
            hop_length=hop_length,
            n_mels=int(n_mels),
            f_min=float(fmin),
            f_max=fmax,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )

    @staticmethod
    def _to_mono(wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            return wav.unsqueeze(0)
        if wav.size(0) == 1:
            return wav
        return wav.mean(dim=0, keepdim=True)

    def _pad_or_trim(self, wav: torch.Tensor) -> torch.Tensor:
        T = wav.size(-1)
        if T < self.clip_len:
            return torch.nn.functional.pad(wav, (0, self.clip_len - T))
        if T > self.clip_len:
            start = (T - self.clip_len) // 2
            return wav[:, start : start + self.clip_len]
        return wav

    def extract_one(self, path: str) -> np.ndarray:
        wav, sr = torchaudio.load(path)  # [C,T]
        wav = self._to_mono(wav).float()  # [1,T]
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        wav = self._pad_or_trim(wav).clamp(-1.0, 1.0)  # [1,T]

        m = self.melspec(wav)  # [1,M,T']
        m = torch.log(m + 1e-6).squeeze(0)  # [M,T']

        if self.cmvn:
            m = (m - m.mean()) / (m.std() + 1e-5)

        # time pooling: mean+std over frames
        mu = m.mean(dim=1)                # [M]
        sd = m.std(dim=1)                 # [M]
        feat = torch.cat([mu, sd], dim=0) # [2M]
        return feat.numpy().astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_pred = (y_prob >= thr).astype(np.int64)

    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)})
    return out


def build_model(model_name: str, C: float, seed: int):
    # StandardScaler is important; pooled log-mel features scale varies
    if model_name == "lr":
        clf = LogisticRegression(
            C=float(C),
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_name == "svm":
        # LinearSVC is strong; calibrate to get probabilities (needed for AUROC/AUPRC)
        base = LinearSVC(C=float(C), class_weight="balanced", random_state=seed)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    raise ValueError("model must be one of: lr, svm")


def predict_proba_1(pipeline, X: np.ndarray) -> np.ndarray:
    # Return P(y=1)
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    # Some pipelines might expose decision_function
    if hasattr(pipeline, "decision_function"):
        z = pipeline.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    # Fallback
    y = pipeline.predict(X).astype(np.float64)
    return y


@dataclass
class Config:
    esc50_root: str
    output_dir: str
    model: str             # lr | svm
    test_fold: int = 1
    seed: int = 0
    val_ratio: float = 0.1

    sample_rate: int = 32000
    clip_seconds: float = 5.0
    n_mels: int = 40
    win_ms: float = 30.0
    hop_ms: float = 10.0
    n_fft: int = 1024
    cmvn: bool = True

    C_list: Tuple[float, ...] = (0.1, 1.0, 3.0, 10.0)

    wandb_project: str = "ESC50-DangerSafe-ML"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # online|offline|disabled
    run_name: Optional[str] = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="runs_ml")
    ap.add_argument("--model", type=str, choices=["lr", "svm"], default="lr")
    ap.add_argument("--test_fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--sample_rate", type=int, default=32000)
    ap.add_argument("--clip_seconds", type=float, default=5.0)
    ap.add_argument("--n_mels", type=int, default=40)
    ap.add_argument("--win_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--no_cmvn", action="store_true")

    ap.add_argument("--C_list", nargs="+", type=float, default=[0.1, 1.0, 3.0, 10.0])

    ap.add_argument("--wandb_project", type=str, default="ESC50-DangerSafe-ML")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="online")
    ap.add_argument("--run_name", type=str, default=None)

    args = ap.parse_args()

    cfg = Config(
        esc50_root=args.esc50_root,
        output_dir=args.output_dir,
        model=args.model,
        test_fold=args.test_fold,
        seed=args.seed,
        val_ratio=args.val_ratio,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        n_mels=args.n_mels,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
        n_fft=args.n_fft,
        cmvn=not args.no_cmvn,
        C_list=tuple(args.C_list),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        run_name=args.run_name,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_global_seed(cfg.seed)

    df = load_esc50_metadata(cfg.esc50_root)
    df_test = df[df["fold"] == cfg.test_fold].reset_index(drop=True)
    df_trainval = df[df["fold"] != cfg.test_fold].reset_index(drop=True)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=cfg.val_ratio, random_state=cfg.seed)
    idx_train, idx_val = next(splitter.split(df_trainval["path"].values, df_trainval["label"].values))
    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val = df_trainval.iloc[idx_val].reset_index(drop=True)

    extractor = LogMelStatsExtractor(
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        win_ms=cfg.win_ms,
        hop_ms=cfg.hop_ms,
        n_fft=cfg.n_fft,
        clip_seconds=cfg.clip_seconds,
        cmvn=cfg.cmvn,
    )

    # Cache features for speed/reproducibility
    cache_path = os.path.join(cfg.output_dir, f"features_fold{cfg.test_fold}_seed{cfg.seed}.npz")
    if os.path.isfile(cache_path):
        pack = np.load(cache_path, allow_pickle=False)
        X_train, y_train = pack["X_train"], pack["y_train"]
        X_val, y_val = pack["X_val"], pack["y_val"]
        X_test, y_test = pack["X_test"], pack["y_test"]
    else:
        def featurize(df_part: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            X_list, y_list = [], []
            for _, r in tqdm(df_part.iterrows(), total=len(df_part), desc="featurize"):
                X_list.append(extractor.extract_one(r["path"]))
                y_list.append(int(r["label"]))
            return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64)

        X_train, y_train = featurize(df_train)
        X_val, y_val = featurize(df_val)
        X_test, y_test = featurize(df_test)

        np.savez_compressed(
            cache_path,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
        )

    wb_run = None
    if cfg.wandb_mode != "disabled":
        if wandb is None:
            raise RuntimeError("wandb not installed but wandb_mode != disabled")
        wkwargs = dict(project=cfg.wandb_project, config=asdict(cfg), mode=cfg.wandb_mode)
        if cfg.wandb_entity:
            wkwargs["entity"] = cfg.wandb_entity
        if cfg.run_name:
            wkwargs["name"] = cfg.run_name
        wb_run = wandb.init(**wkwargs)

    # Grid search C on val using AUROC (fallback F1)
    best = {"C": None, "val_score": -1.0, "pipeline": None, "val_metrics": None}
    for C in cfg.C_list:
        pipe = build_model(cfg.model, C=C, seed=cfg.seed)
        pipe.fit(X_train, y_train)
        val_prob = predict_proba_1(pipe, X_val)
        val_metrics = compute_metrics(y_val, val_prob)
        val_score = val_metrics["auroc"]
        if not np.isfinite(val_score):
            val_score = val_metrics["f1"]

        if wb_run is not None:
            wandb.log({f"val/auroc_C{C}": val_metrics["auroc"], f"val/auprc_C{C}": val_metrics["auprc"], f"val/f1_C{C}": val_metrics["f1"]})

        if float(val_score) > best["val_score"]:
            best.update({"C": float(C), "val_score": float(val_score), "pipeline": pipe, "val_metrics": val_metrics})

    pipe = best["pipeline"]
    test_prob = predict_proba_1(pipe, X_test)
    test_metrics = compute_metrics(y_test, test_prob)

    out_dir = os.path.join(cfg.output_dir, f"ml_{cfg.model}_fold{cfg.test_fold}_seed{cfg.seed}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(os.path.join(out_dir, "val_best.json"), "w") as f:
        json.dump({"best_C": best["C"], "val_metrics": best["val_metrics"]}, f, indent=2)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    joblib.dump(pipe, os.path.join(out_dir, "model.joblib"))

    # If Logistic Regression: also export tiny weights for easy mobile use
    if cfg.model == "lr":
        scaler: StandardScaler = pipe.named_steps["scaler"]
        clf: LogisticRegression = pipe.named_steps["clf"]
        export = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": clf.coef_.reshape(-1).tolist(),
            "intercept": float(clf.intercept_.reshape(-1)[0]),
            "threshold": 0.5,
        }
        with open(os.path.join(out_dir, "lr_weights.json"), "w") as f:
            json.dump(export, f, indent=2)

    summary = {
        "best_C": best["C"],
        **{f"test/{k}": float(v) for k, v in test_metrics.items() if k in ["acc","bacc","precision","recall","f1","auroc","auprc"]},
    }
    print(summary)

    if wb_run is not None:
        wandb.summary.update(summary)
        wb_run.finish()

    print(f"Saved to: {out_dir}")
    print("Artifacts: model.joblib (and lr_weights.json if model=lr)")


if __name__ == "__main__":
    main()
