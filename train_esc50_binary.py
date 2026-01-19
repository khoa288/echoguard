# train_esc50_binary.py
# One-file, reproducible training script for ESC-50 Danger/Safe binary classification (32 kHz preprocessed).
# Key fix: log-mel frontend is forced to FP32 (no autocast) to avoid NaNs on Colab/A100.
# Adds: checkpoint saving (best/last/every N epochs) + optional zipping for easy download.
#
# Expected dataset layout:
#   <ESC50_ROOT>/
#     audio_32k/*.wav
#     meta/esc50.csv
#
# Smoke:
#   python train_esc50_binary.py --esc50_root "$ESC50_ROOT" --output_dir runs_debug \
#     --models bcresnet dscnn --taus 1 --dscnn_width_mults 1.0 --folds 1 --seeds 0 --epochs 5 --wandb_mode disabled \
#     --save_every 1 --zip_runs
#
# Paper-style (recommend save_every 10 to keep size manageable):
#   python train_esc50_binary.py --esc50_root "$ESC50_ROOT" --output_dir runs_paper \
#     --models bcresnet dscnn --taus 1 1.5 2 3 6 8 --dscnn_width_mults 1.0 --folds 1 2 3 4 5 --seeds 0 \
#     --epochs 200 --wandb_mode online --run_name_prefix "esc50_bin_" --save_every 10 --zip_runs

import os
import math
import json
import time
import uuid
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None


# -------------------------
# Reproducibility helpers
# -------------------------
def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -------------------------
# ESC-50 binary labels
# -------------------------
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


def load_esc50_metadata(esc50_root: str) -> pd.DataFrame:
    csv_path = os.path.join(esc50_root, "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_root, "audio_32k")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"esc50.csv not found at: {csv_path}")
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"audio_32k folder not found at: {audio_dir}")

    df = pd.read_csv(csv_path)
    required = {"filename", "fold", "category"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"esc50.csv must contain columns {sorted(list(required))}; got {list(df.columns)}")

    df = df.copy()
    df["path"] = df["filename"].astype(str).apply(lambda fn: os.path.join(audio_dir, fn))
    missing = df.loc[~df["path"].apply(os.path.isfile)]
    if len(missing) > 0:
        raise FileNotFoundError(
            f"Found {len(missing)} missing audio files under audio_32k/ (example: {missing.iloc[0]['path']})"
        )

    df["label"] = df["category"].astype(str).apply(lambda c: 1 if c in DANGER_CATEGORIES else 0).astype(int)
    df["fold"] = df["fold"].astype(int)
    return df


class WavePathDataset(Dataset):
    def __init__(
        self,
        paths: List[str],
        labels: List[int],
        sample_rate: int = 32000,
        clip_seconds: float = 5.0,
        train_crop: bool = True,
    ):
        super().__init__()
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.sr = int(sample_rate)
        self.clip_len = int(round(sample_rate * clip_seconds))
        self.train_crop = bool(train_crop)

    @staticmethod
    def _to_mono(wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            return wav.unsqueeze(0)
        if wav.size(0) == 1:
            return wav
        return wav.mean(dim=0, keepdim=True)

    def _pad_or_trim(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [1,T]
        T = wav.size(-1)
        if T == self.clip_len:
            return wav
        if T < self.clip_len:
            return F.pad(wav, (0, self.clip_len - T))
        # T > clip_len
        if self.train_crop:
            start = random.randint(0, T - self.clip_len)
        else:
            start = (T - self.clip_len) // 2
        return wav[:, start : start + self.clip_len]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = int(self.labels[idx])

        wav, sr = torchaudio.load(path)  # [C,T]
        wav = self._to_mono(wav)         # [1,T]
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        wav = self._pad_or_trim(wav)
        wav = wav.clamp(-1.0, 1.0).float()
        return wav.squeeze(0), torch.tensor(label, dtype=torch.long)


# -------------------------
# Audio front-end (log-mel + optional SpecAugment)
# -------------------------
class AudioFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 40,
        win_ms: float = 30.0,
        hop_ms: float = 10.0,
        n_fft: int = 1024,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        cmvn: bool = True,
        specaug: bool = True,
        freq_mask_param: int = 10,
        time_mask_param: int = 80,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.sr = int(sample_rate)
        self.win_length = int(round(self.sr * win_ms / 1000.0))
        self.hop_length = int(round(self.sr * hop_ms / 1000.0))
        self.n_fft = int(n_fft)
        self.cmvn = bool(cmvn)
        self.use_specaug = bool(specaug)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=int(n_mels),
            f_min=float(fmin),
            f_max=float(self.sr / 2.0 if fmax is None else fmax),
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        self.freq_maskers = nn.ModuleList(
            [torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param) for _ in range(num_freq_masks)]
        )
        self.time_maskers = nn.ModuleList(
            [torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param) for _ in range(num_time_masks)]
        )

    @staticmethod
    def _cmvn(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        mean = x.mean(dim=(-1, -2), keepdim=True)
        std = x.std(dim=(-1, -2), keepdim=True)
        return (x - mean) / (std + eps)

    def forward(self, wav: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """
        wav: [B,T] float32 in [-1,1]
        returns: [B,1,M,T']
        """
        x = self.melspec(wav)           # [B,M,T']
        x = torch.log(x + 1e-6)
        x = x.unsqueeze(1)              # [B,1,M,T']

        if self.cmvn:
            x = self._cmvn(x)

        if augment and self.use_specaug:
            out = []
            for i in range(x.size(0)):
                xi = x[i, 0]  # [M,T']
                for fm in self.freq_maskers:
                    xi = fm(xi)
                for tm in self.time_maskers:
                    xi = tm(xi)
                out.append(xi)
            x = torch.stack(out, dim=0).unsqueeze(1)  # [B,1,M,T']
        return x


# -------------------------
# BC-ResNet -> binary logit
# -------------------------
class SubSpectralNorm(nn.Module):
    def __init__(self, num_features, spec_groups=16, affine="Sub", batch=True, dim=2):
        super().__init__()
        self.spec_groups = spec_groups
        self.affine_all = False
        affine_norm = False
        if affine == "Sub":
            affine_norm = True
        elif affine == "All":
            self.affine_all = True
            self.weight = nn.Parameter(torch.ones((1, num_features, 1, 1)))
            self.bias = nn.Parameter(torch.zeros((1, num_features, 1, 1)))
        if batch:
            self.ssnorm = nn.BatchNorm2d(num_features * spec_groups, affine=affine_norm)
        else:
            self.ssnorm = nn.InstanceNorm2d(num_features * spec_groups, affine=affine_norm)
        self.sub_dim = dim

    def forward(self, x):
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3).contiguous()
        b, c, h, w = x.size()
        if h % self.spec_groups != 0:
            return x
        x = x.view(b, c * self.spec_groups, h // self.spec_groups, w)
        x = self.ssnorm(x)
        x = x.view(b, c, h, w)
        if self.affine_all:
            x = x * self.weight + self.bias
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3).contiguous()
        return x


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_plane,
        out_plane,
        idx,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        swish=False,
        BN=True,
        ssn=False,
    ):
        super().__init__()
        self.idx = idx

        def get_padding(k, use_dil):
            rate = 1
            padding_len = (k - 1) // 2
            if use_dil and k > 1:
                rate = int(2 ** self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        if isinstance(kernel_size, (list, tuple)):
            padding = []
            rate = []
            for k_size in kernel_size:
                p, r = get_padding(k_size, use_dilation)
                padding.append(p)
                rate.append(r)
        else:
            padding, rate = get_padding(kernel_size, use_dilation)

        layers = [nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, rate, groups, bias=False)]
        if ssn:
            layers.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BCResBlock(nn.Module):
    def __init__(self, in_plane, out_plane, idx, stride):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)

        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, 1, 1))
            in_plane = out_plane
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))

        self.f1 = nn.Sequential(
            ConvBNReLU(
                out_plane,
                out_plane,
                idx,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)
        x = self.f1(x)
        x = x + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        x = F.relu(x, True)
        return x


def BCBlockStage(num_layers, last_channel, cur_channel, idx, use_stride):
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
    return stage


class BCResNetBinary(nn.Module):
    def __init__(self, tau: float):
        super().__init__()
        base_c = int(round(float(tau) * 8))
        self.n = [2, 2, 4, 4]
        self.c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]
        self.s = [1, 2]
        self._build_network()

    def _build_network(self):
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
        )
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride))

        self.classifier = nn.Sequential(
            nn.Conv2d(self.c[-2], self.c[-2], (5, 5), bias=False, groups=self.c[-2], padding=(0, 2)),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], 1, 1),
        )

    def forward(self, x):
        x = self.cnn_head(x)
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
        x = self.classifier(x).view(x.size(0))
        return x


# -------------------------
# DS-CNN baseline (depthwise separable conv)
# -------------------------
class DSConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: Tuple[int, int] = (1, 1), k: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=stride, padding=pad, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = F.relu(self.dw_bn(self.dw(x)), inplace=True)
        x = F.relu(self.pw_bn(self.pw(x)), inplace=True)
        return self.drop(x)


class DSCNNBinary(nn.Module):
    def __init__(self, width_mult: float = 1.0, dropout: float = 0.1):
        super().__init__()

        def c(ch):
            return max(8, int(round(ch * width_mult)))

        self.stem = nn.Sequential(
            nn.Conv2d(1, c(32), kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU(True),
        )
        self.blocks = nn.Sequential(
            DSConvBlock(c(32), c(64), stride=(1, 1), k=3, dropout=dropout),
            DSConvBlock(c(64), c(64), stride=(1, 2), k=3, dropout=dropout),
            DSConvBlock(c(64), c(128), stride=(1, 1), k=3, dropout=dropout),
            DSConvBlock(c(128), c(128), stride=(2, 2), k=3, dropout=dropout),
            DSConvBlock(c(128), c(256), stride=(1, 1), k=3, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c(256), 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x).view(x.size(0))


# -------------------------
# Metrics
# -------------------------
def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
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

    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)

    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_pred = (y_prob >= threshold).astype(np.int64)

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


@torch.no_grad()
def evaluate(model: nn.Module, frontend: nn.Module, loader: DataLoader, device: torch.device, amp: bool):
    model.eval()
    frontend.eval()

    all_prob, all_true = [], []

    use_cuda = (device.type == "cuda")
    fp32_ctx = torch.amp.autocast(device_type="cuda", enabled=False) if use_cuda else torch.autocast("cpu", enabled=False)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp) if use_cuda else torch.autocast("cpu", enabled=False)

    for wav, y in loader:
        wav = wav.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True)

        with fp32_ctx:
            feats = frontend(wav, augment=False)
        if not torch.isfinite(feats).all():
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        with amp_ctx:
            logits = model(feats)
            prob = torch.sigmoid(logits)
        if not torch.isfinite(prob).all():
            prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)

        all_prob.append(prob.detach().float().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    return compute_binary_metrics(y_true, y_prob), y_true, y_prob


# -------------------------
# Training
# -------------------------
@dataclass
class RunConfig:
    esc50_root: str
    output_dir: str
    model: str  # bcresnet|dscnn
    tau: float = 1.0
    dscnn_width_mult: float = 1.0
    test_fold: int = 1
    seed: int = 0

    val_ratio: float = 0.1

    sample_rate: int = 32000
    clip_seconds: float = 5.0
    batch_size: int = 128
    num_workers: int = 8

    n_mels: int = 40
    win_ms: float = 30.0
    hop_ms: float = 10.0
    n_fft: int = 1024
    cmvn: bool = True
    specaug: bool = True
    freq_mask_param: int = 10
    time_mask_param: int = 80
    num_freq_masks: int = 2
    num_time_masks: int = 2

    epochs: int = 200
    warmup_epochs: int = 5
    base_lr: float = 0.1
    weight_decay: float = 1e-3
    momentum: float = 0.9
    grad_clip: float = 5.0
    mixup_alpha: float = 0.0
    amp: bool = True
    deterministic: bool = False

    # checkpoint saving
    save_every: int = 0     # 0 = disable; 1 = every epoch; N = every N epochs
    save_last: bool = True
    zip_runs: bool = False

    wandb_project: str = "ESC50-DangerSafe"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # online|offline|disabled
    run_name: Optional[str] = None


def build_model(cfg: RunConfig) -> nn.Module:
    if cfg.model == "bcresnet":
        return BCResNetBinary(cfg.tau)
    if cfg.model == "dscnn":
        return DSCNNBinary(cfg.dscnn_width_mult)
    raise ValueError(cfg.model)


def make_lr_schedule(total_steps: int, warmup_steps: int):
    def lr_mult(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_mult


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y.float(), y.float(), 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    return x_mix, y.float(), y[perm].float(), lam


def bce_mixup_loss(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float, pos_weight: torch.Tensor):
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)


def save_checkpoint(path: str, model: nn.Module, frontend: nn.Module, optimizer, scaler, epoch: int, best_val: float, cfg: RunConfig):
    obj = {
        "model": model.state_dict(),
        "frontend": frontend.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch": epoch,
        "best_val": float(best_val),
        "cfg": asdict(cfg),
    }
    torch.save(obj, path)


def maybe_zip_folder(folder_path: str, zip_path: str):
    import zipfile
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, folder_path)
                zf.write(full, arcname=rel)


def train_one(cfg: RunConfig) -> Dict[str, float]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_esc50_metadata(cfg.esc50_root)
    df_test = df[df["fold"] == int(cfg.test_fold)].reset_index(drop=True)
    df_trainval = df[df["fold"] != int(cfg.test_fold)].reset_index(drop=True)

    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(cfg.val_ratio), random_state=int(cfg.seed))
    idx_train, idx_val = next(splitter.split(df_trainval["path"].values, df_trainval["label"].values))
    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val = df_trainval.iloc[idx_val].reset_index(drop=True)

    train_set = WavePathDataset(df_train["path"].tolist(), df_train["label"].tolist(), cfg.sample_rate, cfg.clip_seconds, train_crop=True)
    val_set = WavePathDataset(df_val["path"].tolist(), df_val["label"].tolist(), cfg.sample_rate, cfg.clip_seconds, train_crop=False)
    test_set = WavePathDataset(df_test["path"].tolist(), df_test["label"].tolist(), cfg.sample_rate, cfg.clip_seconds, train_crop=False)

    y_train = np.array(df_train["label"].values, dtype=np.int64)
    pos = max(1, int(y_train.sum()))
    neg = max(1, int((y_train == 0).sum()))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    persistent = (cfg.num_workers > 0)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=seed_worker, generator=g,
        persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=seed_worker, persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=seed_worker, persistent_workers=persistent
    )

    frontend = AudioFrontend(
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        win_ms=cfg.win_ms,
        hop_ms=cfg.hop_ms,
        n_fft=cfg.n_fft,
        cmvn=cfg.cmvn,
        specaug=cfg.specaug,
        freq_mask_param=cfg.freq_mask_param,
        time_mask_param=cfg.time_mask_param,
        num_freq_masks=cfg.num_freq_masks,
        num_time_masks=cfg.num_time_masks,
    ).to(device)

    model = build_model(cfg).to(device)

    lr = cfg.base_lr * (cfg.batch_size / 100.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = len(train_loader) * cfg.warmup_epochs
    lr_mult = make_lr_schedule(total_steps, warmup_steps)

    use_cuda = (device.type == "cuda")
    fp32_ctx = torch.amp.autocast(device_type="cuda", enabled=False) if use_cuda else torch.autocast("cpu", enabled=False)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.amp) if use_cuda else torch.autocast("cpu", enabled=False)

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and use_cuda))
    base_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    wb_run = None
    if cfg.wandb_mode != "disabled":
        if wandb is None:
            raise RuntimeError("wandb is not installed but wandb_mode != disabled.")
        wandb_kwargs = dict(project=cfg.wandb_project, config=asdict(cfg), mode=cfg.wandb_mode)
        if cfg.wandb_entity:
            wandb_kwargs["entity"] = cfg.wandb_entity
        if cfg.run_name:
            wandb_kwargs["name"] = cfg.run_name
        wb_run = wandb.init(**wandb_kwargs)

    run_tag = (wb_run.id if wb_run is not None else uuid.uuid4().hex[:10])
    ckpt_dir = os.path.join(cfg.output_dir, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    best_val = -1.0
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_path = os.path.join(ckpt_dir, "last.pt")

    global_step = 0
    skipped_batches = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        frontend.train()

        epoch_loss = 0.0
        n_seen = 0

        pbar = tqdm(train_loader, desc=f"{cfg.model} fold{cfg.test_fold} seed{cfg.seed} ep{epoch}", leave=False)
        for wav, y in pbar:
            wav = wav.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)

            # waveform aug
            gain_db = (torch.rand(wav.size(0), device=device) * 12.0 - 6.0)
            gain = torch.pow(10.0, gain_db / 20.0).unsqueeze(1)
            wav = (wav * gain).clamp(-1.0, 1.0)

            max_shift = int(0.1 * cfg.sample_rate)
            shift = torch.randint(-max_shift, max_shift + 1, (wav.size(0),), device=device)
            if max_shift > 0:
                wav_shifted = []
                for i in range(wav.size(0)):
                    s = int(shift[i].item())
                    if s == 0:
                        wav_shifted.append(wav[i])
                    elif s > 0:
                        wav_shifted.append(torch.cat([wav[i, s:], torch.zeros(s, device=device)], dim=0))
                    else:
                        s = -s
                        wav_shifted.append(torch.cat([torch.zeros(s, device=device), wav[i, :-s]], dim=0))
                wav = torch.stack(wav_shifted, dim=0)

            wav = (wav + 0.005 * torch.randn_like(wav)).clamp(-1.0, 1.0)

            if cfg.mixup_alpha > 0:
                wav, y_a, y_b, lam = mixup_batch(wav, y, cfg.mixup_alpha)

            optimizer.zero_grad(set_to_none=True)

            # Frontend FP32
            with fp32_ctx:
                feats = frontend(wav, augment=True)
            if not torch.isfinite(feats).all():
                feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

            # Model AMP optional
            with amp_ctx:
                logits = model(feats)
                if cfg.mixup_alpha > 0:
                    loss = bce_mixup_loss(logits, y_a, y_b, lam, pos_weight=pos_weight)
                else:
                    loss = base_loss(logits, y.float())

            if not torch.isfinite(loss):
                skipped_batches += 1
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            global_step += 1
            mult = lr_mult(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * mult

            bs = wav.size(0)
            epoch_loss += float(loss.detach().cpu().item()) * bs
            n_seen += bs
            pbar.set_postfix({"loss": epoch_loss / max(1, n_seen), "skipped": skipped_batches})

        train_loss = epoch_loss / max(1, n_seen)

        val_metrics, _, _ = evaluate(model, frontend, val_loader, device, cfg.amp)
        val_score = val_metrics["auroc"]
        if not np.isfinite(val_score):
            val_score = val_metrics["f1"]

        cur_lr = optimizer.param_groups[0]["lr"]
        log_dict = {
            "epoch": epoch,
            "lr": float(cur_lr),
            "train/loss": float(train_loss),
            "train/skipped_batches": int(skipped_batches),
            **{f"val/{k}": float(v) for k, v in val_metrics.items() if k in ["acc", "bacc", "precision", "recall", "f1", "auroc", "auprc"]},
        }

        if wb_run is not None:
            wandb.log(log_dict, step=global_step)
        else:
            if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
                print(log_dict)

        # best
        if float(val_score) > best_val:
            best_val = float(val_score)
            save_checkpoint(best_path, model, frontend, optimizer, scaler, epoch, best_val, cfg)

        # periodic
        if cfg.save_every and cfg.save_every > 0 and (epoch % cfg.save_every == 0):
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"), model, frontend, optimizer, scaler, epoch, best_val, cfg)

        # last (each epoch, cheap)
        if cfg.save_last:
            save_checkpoint(last_path, model, frontend, optimizer, scaler, epoch, best_val, cfg)

    # Test best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    frontend.load_state_dict(ckpt["frontend"])

    test_metrics, y_true, y_prob = evaluate(model, frontend, test_loader, device, cfg.amp)
    with open(os.path.join(ckpt_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print({"best_epoch": int(ckpt["epoch"]), **{f"test/{k}": float(v) for k, v in test_metrics.items() if k in ["acc","bacc","precision","recall","f1","auroc","auprc"]}})

    if wb_run is not None:
        wandb.summary.update({f"test/{k}": float(v) for k, v in test_metrics.items()})
        try:
            y_pred = (np.nan_to_num(y_prob, nan=0.0) >= 0.5).astype(np.int64)
            wandb.log({"test/conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred, class_names=["safe", "danger"])})
        except Exception:
            pass
        wb_run.finish()

    # optional zip per-run folder (checkpoints + config + metrics)
    if cfg.zip_runs:
        zip_path = os.path.join(cfg.output_dir, f"{run_tag}.zip")
        maybe_zip_folder(ckpt_dir, zip_path)

    return {
        "model": cfg.model,
        "tau": float(cfg.tau),
        "dscnn_width_mult": float(cfg.dscnn_width_mult),
        "test_fold": int(cfg.test_fold),
        "seed": int(cfg.seed),
        **{f"test_{k}": float(v) for k, v in test_metrics.items() if k in ["acc", "bacc", "precision", "recall", "f1", "auroc", "auprc"]},
        "run_dir": ckpt_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs_esc50")

    parser.add_argument("--models", nargs="+", default=["bcresnet", "dscnn"], choices=["bcresnet", "dscnn"])
    parser.add_argument("--taus", nargs="+", type=float, default=[1, 1.5, 2, 3, 6, 8])
    parser.add_argument("--dscnn_width_mults", nargs="+", type=float, default=[1.0])

    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])

    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--clip_seconds", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--base_lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--n_mels", type=int, default=40)
    parser.add_argument("--win_ms", type=float, default=30.0)
    parser.add_argument("--hop_ms", type=float, default=10.0)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--no_cmvn", action="store_true")
    parser.add_argument("--no_specaug", action="store_true")
    parser.add_argument("--freq_mask_param", type=int, default=10)
    parser.add_argument("--time_mask_param", type=int, default=80)
    parser.add_argument("--num_freq_masks", type=int, default=2)
    parser.add_argument("--num_time_masks", type=int, default=2)

    # checkpoint flags
    parser.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0=disable).")
    parser.add_argument("--no_save_last", action="store_true", help="Disable saving last.pt each epoch.")
    parser.add_argument("--zip_runs", action="store_true", help="Zip each run folder into output_dir/<runid>.zip")

    parser.add_argument("--wandb_project", type=str, default="ESC50-DangerSafe")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--run_name_prefix", type=str, default="")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []

    for model_name in args.models:
        if model_name == "bcresnet":
            exps = [{"model": "bcresnet", "tau": float(t)} for t in args.taus]
        else:
            exps = [{"model": "dscnn", "dscnn_width_mult": float(w)} for w in args.dscnn_width_mults]

        for exp in exps:
            for fold in args.folds:
                for seed in args.seeds:
                    run_name = None
                    if args.run_name_prefix:
                        if exp["model"] == "bcresnet":
                            run_name = f"{args.run_name_prefix}bcresnet_tau{exp['tau']}_fold{fold}_seed{seed}"
                        else:
                            run_name = f"{args.run_name_prefix}dscnn_w{exp['dscnn_width_mult']}_fold{fold}_seed{seed}"

                    cfg = RunConfig(
                        esc50_root=args.esc50_root,
                        output_dir=args.output_dir,
                        model=exp["model"],
                        tau=float(exp.get("tau", 1.0)),
                        dscnn_width_mult=float(exp.get("dscnn_width_mult", 1.0)),
                        test_fold=int(fold),
                        seed=int(seed),
                        val_ratio=float(args.val_ratio),
                        sample_rate=int(args.sample_rate),
                        clip_seconds=float(args.clip_seconds),
                        batch_size=int(args.batch_size),
                        num_workers=int(args.num_workers),
                        n_mels=int(args.n_mels),
                        win_ms=float(args.win_ms),
                        hop_ms=float(args.hop_ms),
                        n_fft=int(args.n_fft),
                        cmvn=not args.no_cmvn,
                        specaug=not args.no_specaug,
                        freq_mask_param=int(args.freq_mask_param),
                        time_mask_param=int(args.time_mask_param),
                        num_freq_masks=int(args.num_freq_masks),
                        num_time_masks=int(args.num_time_masks),
                        epochs=int(args.epochs),
                        warmup_epochs=int(args.warmup_epochs),
                        base_lr=float(args.base_lr),
                        weight_decay=float(args.weight_decay),
                        momentum=float(args.momentum),
                        grad_clip=float(args.grad_clip),
                        mixup_alpha=float(args.mixup_alpha),
                        amp=not args.no_amp,
                        deterministic=bool(args.deterministic),
                        save_every=int(args.save_every),
                        save_last=not bool(args.no_save_last),
                        zip_runs=bool(args.zip_runs),
                        wandb_project=str(args.wandb_project),
                        wandb_entity=args.wandb_entity,
                        wandb_mode=str(args.wandb_mode),
                        run_name=run_name,
                    )

                    print(f"\n=== RUN {cfg.model} | tau={cfg.tau} dscnn_w={cfg.dscnn_width_mult} | test_fold={cfg.test_fold} | seed={cfg.seed} ===")
                    row = train_one(cfg)
                    rows.append(row)

                    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "all_runs_raw.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "all_runs_raw.csv"), index=False)

    # Force numeric dtype for all test_* metrics (fixes NaN-from-dtype issues in aggregation)
    metric_cols = [c for c in df.columns if c.startswith("test_")]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    group_cols = ["model", "tau", "dscnn_width_mult"]
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.to_csv(os.path.join(args.output_dir, "summary_mean_std.csv"), index=False)

    print("\n========== FINAL SUMMARY (mean±std across folds/seeds) ==========")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
        print(agg)

    print(f"\nSaved:\n- {os.path.join(args.output_dir, 'all_runs_raw.csv')}\n- {os.path.join(args.output_dir, 'summary_mean_std.csv')}")
    print(f"Run folders (each contains best.pt/last.pt/epoch_*.pt/config.json/test_metrics.json): {args.output_dir}/<runid>/")


if __name__ == "__main__":
    main()
