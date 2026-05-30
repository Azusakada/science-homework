from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
LABEL_CSV = ROOT / "data" / "train_labels.csv"
OUT_DIR = ROOT / "diagnostics"

METHOD = "conv_fusion_z_log_0.20"
FUSION_WEIGHT = 0.20
DEFAULT_SEEDS = (2026, 1145, 114399)


@dataclass(frozen=True)
class ConvConfig:
    batch_size: int = 128
    epochs: int = 55
    lr: float = 7e-4
    weight_decay: float = 1.5e-3
    dropout: float = 0.30
    workers: int = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_eeg(data_dir: str | Path, files: Iterable[str]) -> np.ndarray:
    data_dir = Path(data_dir)
    return np.stack([np.load(data_dir / file_name).astype(np.float32).squeeze() * 1e6 for file_name in files])


def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    labels = pd.read_csv(LABEL_CSV, usecols=["eeg_file", "label"])
    y = (labels["label"] == "target").astype(np.int64).to_numpy()
    raw = load_eeg(TRAIN_DIR, labels["eeg_file"].tolist()).astype(np.float32)
    return raw, y, labels


def session_indices(labels: pd.DataFrame, train_session: str, test_session: str) -> tuple[np.ndarray, np.ndarray]:
    train_mask = labels["eeg_file"].str.contains(train_session, regex=False).to_numpy()
    test_mask = labels["eeg_file"].str.contains(test_session, regex=False).to_numpy()
    if np.any(train_mask & test_mask):
        raise RuntimeError(f"overlap between {train_session} and {test_session}")
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(f"empty split: {train_session}={len(train_idx)}, {test_session}={len(test_idx)}")
    return train_idx, test_idx


def extract_raw_amplitude_features(raw: np.ndarray) -> np.ndarray:
    x = raw.astype(np.float32, copy=False)
    smooth = signal.savgol_filter(x, window_length=15, polyorder=3, axis=2, mode="interp")
    pieces = []
    for width in [6, 10, 15, 20, 30]:
        n_bins = x.shape[2] // width
        pieces.append(smooth[:, :, : n_bins * width].reshape(len(x), 59, n_bins, width).mean(axis=3).reshape(len(x), -1))

    windows = [(start, min(282, start + 40)) for start in range(0, 282, 20)]
    windows += [(50, 120), (80, 160), (100, 180), (120, 220), (150, 240), (180, 260), (200, 282), (220, 282), (240, 282)]
    window_parts = []
    for start, stop in windows:
        window = smooth[:, :, start:stop]
        window_parts.extend(
            [
                window.mean(axis=2),
                window.std(axis=2),
                window.max(axis=2),
                window.min(axis=2),
                np.sqrt(np.mean(window * window, axis=2)),
            ]
        )
    pieces.append(np.concatenate(window_parts, axis=1))

    fft = np.abs(np.fft.rfft(x, axis=2))
    band_parts = []
    for start, stop in [(1, 3), (3, 6), (6, 10), (10, 16), (16, 24), (24, 36), (36, 55), (55, 80), (80, 110), (110, 142)]:
        band_parts.append(np.log1p(np.mean(fft[:, :, start:stop] ** 2, axis=2)))
    pieces.append(np.concatenate(band_parts, axis=1))

    stat_parts = [
        x.mean(axis=2),
        x.std(axis=2),
        stats.skew(x, axis=2, bias=False),
        stats.kurtosis(x, axis=2, bias=False),
        x.max(axis=2),
        x.min(axis=2),
        np.ptp(x, axis=2),
    ]
    pieces.append(np.nan_to_num(np.concatenate(stat_parts, axis=1)))
    return np.nan_to_num(np.hstack(pieces).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_domain_features(raw: np.ndarray) -> np.ndarray:
    fft = np.abs(np.fft.rfft(raw.astype(np.float32, copy=False), axis=2))
    pieces = []
    for start, stop in [(1, 4), (4, 8), (8, 14), (14, 30), (30, 60), (60, 110), (110, 142)]:
        pieces.append(np.log1p(np.mean(fft[:, :, start:stop] ** 2, axis=2)))
    return np.nan_to_num(np.hstack(pieces).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def make_svc(seed: int, n_features: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(800, n_features))),
            ("clf", SVC(C=1.4, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
        ]
    )


class SpectralDomainSVC:
    def __init__(self, seed: int = 2026, n_clusters: int = 8, domain_blend: float = 0.5):
        self.seed = seed
        self.n_clusters = n_clusters
        self.domain_blend = domain_blend

    def fit(self, raw_features: np.ndarray, domain_features: np.ndarray, y: np.ndarray) -> SpectralDomainSVC:
        self.global_model_ = make_svc(self.seed, raw_features.shape[1])
        self.global_model_.fit(raw_features, y)

        self.domain_scaler_ = StandardScaler()
        z = self.domain_scaler_.fit_transform(domain_features)
        self.domain_pca_ = PCA(n_components=min(10, z.shape[1]), random_state=self.seed)
        z = self.domain_pca_.fit_transform(z)
        self.clusterer_ = KMeans(n_clusters=self.n_clusters, n_init=40, random_state=self.seed)
        clusters = self.clusterer_.fit_predict(z)

        self.cluster_models_: dict[int, Pipeline] = {}
        for cluster_id in range(self.n_clusters):
            idx = np.where(clusters == cluster_id)[0]
            if len(idx) >= 160 and min(np.bincount(y[idx], minlength=2)) >= 50:
                model = make_svc(self.seed + 100 + cluster_id, raw_features.shape[1])
                model.fit(raw_features[idx], y[idx])
                self.cluster_models_[cluster_id] = model
        return self

    def _clusters(self, domain_features: np.ndarray) -> np.ndarray:
        z = self.domain_scaler_.transform(domain_features)
        z = self.domain_pca_.transform(z)
        return self.clusterer_.predict(z)

    def predict_proba(self, raw_features: np.ndarray, domain_features: np.ndarray) -> np.ndarray:
        global_prob = self.global_model_.predict_proba(raw_features)
        clusters = self._clusters(domain_features)
        cluster_prob = global_prob.copy()
        for cluster_id, model in self.cluster_models_.items():
            mask = clusters == cluster_id
            if np.any(mask):
                cluster_prob[mask] = model.predict_proba(raw_features[mask])
        return (1.0 - self.domain_blend) * global_prob + self.domain_blend * cluster_prob


def fit_stats(raw: np.ndarray, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    x = raw[train_idx]
    channel_mean = x.mean(axis=(0, 2), keepdims=True)
    channel_std = x.std(axis=(0, 2), keepdims=True) + 1e-6
    return {
        "channel_mean": channel_mean.astype(np.float32),
        "channel_std": channel_std.astype(np.float32),
    }


def make_views(raw: np.ndarray, stats_: dict[str, np.ndarray]) -> np.ndarray:
    x = raw.astype(np.float32, copy=False)
    train_z = (x - stats_["channel_mean"]) / stats_["channel_std"]
    epoch_z = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-6)
    deriv = np.diff(train_z, axis=2, prepend=train_z[:, :, :1])
    return np.concatenate([train_z, epoch_z, deriv], axis=1).astype(np.float32)


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


class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, train: bool, seed: int):
        self.X = X[idx].astype(np.float32)
        self.y = y[idx].astype(np.int64)
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def train_net(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, seed: int, cfg: ConvConfig) -> MultiViewConv:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = EEGDataset(X, y, train_idx, True, seed)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)
    model = MultiViewConv(X.shape[1], cfg.dropout).to(device)
    counts = np.bincount(y[train_idx], minlength=2)
    weights = torch.tensor(len(train_idx) / (2.0 * counts), dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for _ in range(1, cfg.epochs + 1):
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


def extract_embeddings(model: MultiViewConv, X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    ds = EEGDataset(X, y, idx, False, 0)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    feats = []
    truth = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            _, feat = model(xb)
            feats.append(feat.float().cpu().numpy())
            truth.append(yb.numpy())
    return np.vstack(feats), np.concatenate(truth)


def fit_predict_single_best(
    raw_features: np.ndarray,
    domain_features: np.ndarray,
    raw: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    seed: int,
    cfg: ConvConfig,
) -> np.ndarray:
    stats_ = fit_stats(raw, fit_idx)
    X = make_views(raw, stats_)
    net = train_net(X, y, fit_idx, seed, cfg)
    z_fit, y_fit = extract_embeddings(net, X, y, fit_idx)
    z_eval, _ = extract_embeddings(net, X, y, eval_idx)

    svc = SpectralDomainSVC(seed=seed, n_clusters=8, domain_blend=0.5)
    svc.fit(raw_features[fit_idx], domain_features[fit_idx], y[fit_idx])
    svc_prob = svc.predict_proba(raw_features[eval_idx], domain_features[eval_idx])[:, 1].astype(np.float32)

    z_log = Pipeline(
        [
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=0.18, class_weight="balanced", max_iter=3000, random_state=seed)),
        ]
    )
    z_log.fit(z_fit, y_fit)
    z_log_prob = z_log.predict_proba(z_eval)[:, 1].astype(np.float32)

    return ((1.0 - FUSION_WEIGHT) * svc_prob + FUSION_WEIGHT * z_log_prob).astype(np.float32)


def run_direction(
    raw: np.ndarray,
    y: np.ndarray,
    labels: pd.DataFrame,
    raw_features: np.ndarray,
    domain_features: np.ndarray,
    seed: int,
    train_session: str,
    test_session: str,
) -> dict[str, object]:
    train_idx, test_idx = session_indices(labels, train_session, test_session)
    y_train = y[train_idx]
    y_test = y[test_idx]
    split_name = f"{train_session}_train_{test_session}_test"
    print(f"seed={seed} {split_name}: train={len(train_idx)} test={len(test_idx)}", flush=True)

    inner_cfg = ConvConfig(epochs=42)
    final_cfg = ConvConfig(epochs=55)
    oof_prob = np.zeros(len(train_idx), dtype=np.float32)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 4401)
    for fold, (fit_rel, eval_rel) in enumerate(inner.split(raw_features[train_idx], y_train), start=1):
        fit_idx = train_idx[fit_rel]
        eval_idx = train_idx[eval_rel]
        oof_prob[eval_rel] = fit_predict_single_best(raw_features, domain_features, raw, y, fit_idx, eval_idx, seed * 31 + fold, inner_cfg)
        print(f"  inner_fold={fold}/3", flush=True)

    test_prob = fit_predict_single_best(raw_features, domain_features, raw, y, train_idx, test_idx, seed, final_cfg)
    test_pred = test_prob >= 0.5
    row = {
        "split": split_name,
        "run_seed": seed,
        "method": METHOD,
        "selected": METHOD,
        "oof_accuracy": accuracy_score(y_train, oof_prob >= 0.5),
        "accuracy": accuracy_score(y_test, test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
        "pred_rate": float(test_pred.mean()),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
    print(f"  {METHOD} accuracy={row['accuracy']:.6f} oof_accuracy={row['oof_accuracy']:.6f}", flush=True)
    return row


def output_path(seed: int, train_session: str, test_session: str, many_seeds: bool) -> Path:
    if seed == 2026 and not many_seeds:
        return OUT_DIR / f"session_{train_session}_train_{test_session}_test.csv"
    return OUT_DIR / f"session_seed{seed}_{train_session}_train_{test_session}_test.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run only {METHOD} for sess1/sess2 transfer.")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the original conv training flow")

    raw, y, labels = load_data()
    raw_features = extract_raw_amplitude_features(raw)
    domain_features = extract_domain_features(raw)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    many_seeds = len(seeds) != 1
    for seed in seeds:
        for train_session, test_session in [("sess1", "sess2"), ("sess2", "sess1")]:
            row = run_direction(raw, y, labels, raw_features, domain_features, seed, train_session, test_session)
            rows.append(row)
            pd.DataFrame([row]).to_csv(output_path(seed, train_session, test_session, many_seeds), index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "single_best_session_transfer_runs.csv", index=False)
    mean_row = {
        "method": METHOD,
        "mean_accuracy": summary["accuracy"].mean(),
        "min_accuracy": summary["accuracy"].min(),
        "max_accuracy": summary["accuracy"].max(),
        "n_runs": len(summary),
    }
    pd.DataFrame([mean_row]).to_csv(OUT_DIR / "single_best_session_transfer_summary.csv", index=False)
    print("\nSUMMARY")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame([mean_row]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
