from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

from raw_eeg_final import SpectralDomainSVC, extract_domain_features, extract_raw_amplitude_features, load_eeg


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
LABEL_CSV = ROOT / "data" / "train_labels.csv"
OUT = ROOT / "diagnostics" / "multiview_conv_embed.csv"
SEEDS = [2028, 2029, 2030, 2026, 2027]


@dataclass
class Config:
    batch_size: int = 128
    epochs: int = 70
    lr: float = 7e-4
    weight_decay: float = 1.5e-3
    dropout: float = 0.30
    workers: int = 4
    mode: str = "multi"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_data() -> tuple[np.ndarray, np.ndarray]:
    labels = pd.read_csv(LABEL_CSV, usecols=["eeg_file", "label"])
    y = (labels["label"] == "target").astype(np.int64).to_numpy()
    raw = load_eeg(TRAIN_DIR, labels["eeg_file"].tolist()).astype(np.float32)
    return raw, y


def fit_stats(raw: np.ndarray, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    x = raw[train_idx]
    channel_mean = x.mean(axis=(0, 2), keepdims=True)
    channel_std = x.std(axis=(0, 2), keepdims=True) + 1e-6
    robust_center = np.median(x, axis=(0, 2), keepdims=True)
    robust_scale = np.percentile(np.abs(x - robust_center), 75, axis=(0, 2), keepdims=True) + 1e-6
    return {
        "channel_mean": channel_mean.astype(np.float32),
        "channel_std": channel_std.astype(np.float32),
        "robust_center": robust_center.astype(np.float32),
        "robust_scale": robust_scale.astype(np.float32),
    }


def make_views(raw: np.ndarray, stats: dict[str, np.ndarray], mode: str) -> np.ndarray:
    x = raw.astype(np.float32, copy=False)
    train_z = (x - stats["channel_mean"]) / stats["channel_std"]
    robust_z = (x - stats["robust_center"]) / stats["robust_scale"]
    epoch_z = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)
    base = x - x[:, :, :50].mean(axis=2, keepdims=True)
    base = base / (base.std(axis=(1, 2), keepdims=True) + 1e-6)
    deriv = np.diff(train_z, axis=2, prepend=train_z[:, :, :1])
    if mode == "channel":
        views = [train_z]
    elif mode == "multi":
        views = [train_z, epoch_z, deriv]
    elif mode == "robust_multi":
        views = [train_z, robust_z, epoch_z, deriv]
    elif mode == "baseline_multi":
        views = [train_z, base, deriv]
    else:
        raise ValueError(mode)
    return np.concatenate(views, axis=1).astype(np.float32)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    z = x.copy()
    n_base = 59
    if rng.random() < 0.60:
        z += rng.normal(0, rng.uniform(0.003, 0.016), z.shape).astype(np.float32)
    if rng.random() < 0.30:
        z *= rng.uniform(0.90, 1.10)
    if rng.random() < 0.25:
        ch = rng.choice(n_base, size=int(rng.integers(1, 6)), replace=False)
        for offset in range(0, z.shape[0], n_base):
            z[offset + ch] = 0
    if rng.random() < 0.25:
        start = int(rng.integers(0, 250))
        width = int(rng.integers(8, 30))
        z[:, start : min(z.shape[1], start + width)] = 0
    return z


class EEGDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, train: bool, seed: int):
        self.X = X[idx].astype(np.float32)
        self.y = y[idx].astype(np.int64)
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.train:
            x = augment(x, np.random.default_rng(self.seed + i * 4271))
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MultiViewConv(nn.Module):
    def __init__(self, in_channels: int, dropout: float):
        super().__init__()
        self.stem = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(in_channels, 64, k, padding=k // 2, bias=False), nn.BatchNorm1d(64), nn.GELU())
                for k in (5, 11, 21)
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(192, 160, 1, bias=False),
            nn.BatchNorm1d(160),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            ResBlock(160, 9, 1, dropout),
            ResBlock(160, 9, 2, dropout),
            nn.AvgPool1d(2),
            ResBlock(160, 7, 4, dropout),
            ResBlock(160, 5, 8, dropout),
            nn.AvgPool1d(2),
            ResBlock(160, 5, 12, dropout),
        )
        self.proj = nn.Sequential(nn.Linear(160 * 5, 192), nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(192, 2)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([stem(x) for stem in self.stem], dim=1)
        z = self.mix(z)
        z = self.blocks(z)
        avg = F.adaptive_avg_pool1d(z, 1).squeeze(-1)
        mx = F.adaptive_max_pool1d(z, 1).squeeze(-1)
        q1 = z[:, :, : z.shape[-1] // 3].mean(dim=-1)
        q2 = z[:, :, z.shape[-1] // 3 : 2 * z.shape[-1] // 3].mean(dim=-1)
        q3 = z[:, :, 2 * z.shape[-1] // 3 :].mean(dim=-1)
        return self.proj(torch.cat([avg, mx, q1, q2, q3], dim=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        return self.head(feat), feat


def train_net(X: np.ndarray, y: np.ndarray, tr: np.ndarray, seed: int, cfg: Config) -> MultiViewConv:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = EEGDS(X, y, tr, True, seed)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)
    model = MultiViewConv(X.shape[1], cfg.dropout).to(device)
    counts = np.bincount(y[tr], minlength=2)
    weights = torch.tensor(len(tr) / (2.0 * counts), dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, _ = model(xb)
                loss = ce(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
    return model


def extract(model: MultiViewConv, X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    ds = EEGDS(X, y, idx, False, 0)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    feats, probs, truth = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits, feat = model(xb)
            probs.append(torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy())
            feats.append(feat.float().cpu().numpy())
            truth.append(yb.numpy())
    return np.vstack(feats), np.concatenate(probs), np.concatenate(truth)


def score_row(seed: int, mode: str, method: str, y_true: np.ndarray, prob: np.ndarray) -> dict[str, object]:
    return {
        "seed": seed,
        "mode": mode,
        "method": method,
        "accuracy": accuracy_score(y_true, prob >= 0.5),
        "balanced_accuracy": balanced_accuracy_score(y_true, prob >= 0.5),
        "auc": roc_auc_score(y_true, prob),
    }


def run() -> None:
    raw, y = load_data()
    raw_features = extract_raw_amplitude_features(raw)
    domain_features = extract_domain_features(raw)
    rows = []
    cfgs = [
        Config(mode="multi", epochs=55),
        Config(mode="baseline_multi", epochs=55),
        Config(mode="robust_multi", epochs=65),
    ]
    for cfg in cfgs:
        print(f"\nCONFIG {cfg}", flush=True)
        for seed in SEEDS:
            tr, va = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(raw, y))
            stats = fit_stats(raw, tr)
            X = make_views(raw, stats, cfg.mode)
            net = train_net(X, y, tr, seed, cfg)
            ztr, ptr, ytr = extract(net, X, y, tr)
            zva, pnn, yva = extract(net, X, y, va)
            heads = {
                "net": pnn,
                "z_log": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.18, class_weight="balanced", max_iter=3000, random_state=seed))]),
                "z_svc": Pipeline([("sc", StandardScaler()), ("clf", SVC(C=0.9, gamma="scale", probability=True, class_weight="balanced", random_state=seed))]),
                "zp_log": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.18, class_weight="balanced", max_iter=3000, random_state=seed))]),
                "z_et": ExtraTreesClassifier(n_estimators=600, max_features=0.55, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=seed),
            }
            probs = {"net": pnn}
            for name, head in heads.items():
                if name == "net":
                    continue
                if name == "zp_log":
                    head.fit(np.column_stack([ztr, ptr]), ytr)
                    probs[name] = head.predict_proba(np.column_stack([zva, pnn]))[:, 1]
                else:
                    head.fit(ztr, ytr)
                    probs[name] = head.predict_proba(zva)[:, 1]
            svc = SpectralDomainSVC(seed=seed, n_clusters=8, domain_blend=0.5)
            svc.fit(raw_features[tr], domain_features[tr], y[tr])
            svc_prob = svc.predict_proba(raw_features[va], domain_features[va])[:, 1]
            probs["svc"] = svc_prob
            for base_name in ["net", "z_log", "z_svc", "zp_log", "z_et"]:
                for w in [0.10, 0.20, 0.35]:
                    probs[f"fusion_{base_name}_{w:.2f}"] = (1.0 - w) * svc_prob + w * probs[base_name]
            for name, prob in probs.items():
                row = score_row(seed, cfg.mode, name, yva, prob)
                rows.append(row)
            best = max((r for r in rows if r["seed"] == seed and r["mode"] == cfg.mode), key=lambda r: r["accuracy"])
            print(
                seed,
                f"best={best['method']} acc={best['accuracy']:.4f} auc={best['auc']:.4f}",
                f"svc={accuracy_score(yva, svc_prob >= 0.5):.4f}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    summary = (
        df.groupby(["mode", "method"])
        .agg(mean_acc=("accuracy", "mean"), min_acc=("accuracy", "min"), max_acc=("accuracy", "max"), mean_auc=("auc", "mean"))
        .sort_values(["min_acc", "mean_acc"], ascending=False)
    )
    print("\nSUMMARY")
    print(summary.head(80).to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    run()
