from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from raw_eeg_final import SpectralDomainSVC


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache_strict_eeg"
OUT = ROOT / "diagnostics" / "spec_gpu_oof.csv"
DETAIL = ROOT / "diagnostics" / "spec_gpu_oof_detail.csv"


@dataclass(frozen=True)
class ViewConfig:
    name: str
    feature: str
    k: int
    transform: str = "std"
    hidden: int = 192
    epochs: int = 90
    dropout: float = 0.34
    lr: float = 8e-4
    weight_decay: float = 8e-4
    batch_size: int = 192
    noise: float = 0.018
    mixup: float = 0.18


CONFIGS = {
    "spec649": ViewConfig("spec649", "bank_std_spec", 649, hidden=160, epochs=95, dropout=0.30, noise=0.015),
    "spec649_qt": ViewConfig("spec649_qt", "bank_std_spec", 649, transform="qt", hidden=160, epochs=95, dropout=0.32, noise=0.012),
    "signal1600": ViewConfig("signal1600", "bank_std_signal", 1600, hidden=256, epochs=90, dropout=0.36, noise=0.018),
    "phase900": ViewConfig("phase900", "phase_phase", 900, hidden=192, epochs=85, dropout=0.34, noise=0.016),
    "corr1600": ViewConfig("corr1600", "phase_corr", 1600, hidden=224, epochs=85, dropout=0.34, noise=0.016),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_features() -> tuple[dict[str, np.ndarray], np.ndarray]:
    features: dict[str, np.ndarray] = {}
    _, raw, y = joblib.load(CACHE / "raw_amp_features_v1.joblib")
    grid, domain, y2 = joblib.load(CACHE / "raw_amp_grid_features_v1.joblib")
    _, bank, y3 = joblib.load(CACHE / "feature_bank_v2.joblib")
    _, phase, y4 = joblib.load(CACHE / "phase_feature_bank_v1.joblib")
    tf = joblib.load(CACHE / "timefreq_local_features_v1.joblib")
    for yy in [y2, y3, y4]:
        if not np.array_equal(y, yy):
            raise RuntimeError("label mismatch")
    features["raw"] = raw.astype(np.float32, copy=False)
    features["grid"] = grid.astype(np.float32, copy=False)
    features["domain"] = domain.astype(np.float32, copy=False)
    for key, value in bank.items():
        features[f"bank_{key}"] = value.astype(np.float32, copy=False)
    for key, value in phase.items():
        features[f"phase_{key}"] = value.astype(np.float32, copy=False)
    for key, value in tf.items():
        features[f"tf_{key}"] = value.astype(np.float32, copy=False)
    return features, y.astype(np.int64, copy=False)


def prepare_view(
    features: dict[str, np.ndarray],
    y: np.ndarray,
    fit_idx: np.ndarray,
    pred_idx: np.ndarray,
    cfg: ViewConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x = features[cfg.feature]
    selector = SelectKBest(f_classif, k=min(cfg.k, x.shape[1]))
    scaler = StandardScaler()
    x_fit = selector.fit_transform(x[fit_idx], y[fit_idx])
    x_pred = selector.transform(x[pred_idx])
    x_fit = scaler.fit_transform(x_fit)
    x_pred = scaler.transform(x_pred)
    if cfg.transform == "qt":
        qt = QuantileTransformer(n_quantiles=min(700, len(fit_idx)), output_distribution="normal", random_state=0)
        x_fit = qt.fit_transform(x_fit)
        x_pred = qt.transform(x_pred)
    elif cfg.transform != "std":
        raise ValueError(cfg.transform)
    return (
        np.nan_to_num(x_fit.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(x_pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
    )


class SpecMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_gpu_predict_many(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_preds: list[np.ndarray],
    seed: int,
    cfg: ViewConfig,
) -> list[np.ndarray]:
    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    counts = np.bincount(y_fit, minlength=2).astype(np.float32)
    class_weight = len(y_fit) / (2.0 * np.maximum(counts, 1.0))
    sample_weight = class_weight[y_fit].astype(np.float32)
    sampler = WeightedRandomSampler(sample_weight.astype(np.float64), len(sample_weight), replacement=True)
    ds = TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit.astype(np.int64)))
    loader = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=2, pin_memory=True, drop_last=False)
    model = SpecMLP(x_fit.shape[1], cfg.hidden, cfg.dropout).to(device)
    weights = torch.tensor(class_weight, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda")
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if cfg.mixup > 0 and xb.shape[0] > 2:
                lam = np.random.beta(cfg.mixup, cfg.mixup)
                perm = torch.randperm(xb.shape[0], device=device)
                xb = lam * xb + (1.0 - lam) * xb[perm]
                y_one = F.one_hot(yb, 2).float()
                y_soft = lam * y_one + (1.0 - lam) * y_one[perm]
            else:
                y_soft = F.one_hot(yb, 2).float()
            xb = xb + torch.randn_like(xb) * cfg.noise
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                logp = F.log_softmax(logits.float(), dim=1)
                loss = -(y_soft * logp * weights[None, :]).sum(dim=1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_pred in x_preds:
            pred_loader = DataLoader(TensorDataset(torch.from_numpy(x_pred)), batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
            probs = []
            for (xb,) in pred_loader:
                logits = model(xb.to(device, non_blocking=True))
                probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            outputs.append(np.concatenate(probs).astype(np.float32))
    return outputs


def train_gpu_prob(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_pred: np.ndarray,
    seed: int,
    cfg: ViewConfig,
) -> np.ndarray:
    return train_gpu_predict_many(x_fit, y_fit, [x_pred], seed, cfg)[0]


def make_linear_heads(seed: int, cfg: ViewConfig) -> dict[str, object]:
    return {
        "log_c04": LogisticRegression(C=0.04, class_weight="balanced", max_iter=5000, random_state=seed),
        "log_c08": LogisticRegression(C=0.08, class_weight="balanced", max_iter=5000, random_state=seed),
        "ridge_a5": RidgeClassifier(alpha=5.0, class_weight="balanced"),
        "svc_c08": SVC(C=0.8, gamma="scale", probability=True, class_weight="balanced", random_state=seed),
    }


def proba_or_score(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(np.float32)
    score = model.decision_function(x).astype(np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0)))).astype(np.float32)


def fit_view_oof_val(
    features: dict[str, np.ndarray],
    y: np.ndarray,
    outer_train: np.ndarray,
    outer_val: np.ndarray,
    cfg: ViewConfig,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    oof: dict[str, np.ndarray] = {f"{cfg.name}_gpu": np.zeros(len(outer_train), dtype=np.float32)}
    head_names = [f"{cfg.name}_{name}" for name in make_linear_heads(seed, cfg)]
    for name in head_names:
        oof[name] = np.zeros(len(outer_train), dtype=np.float32)
    val_parts: dict[str, list[np.ndarray]] = {key: [] for key in oof}
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 741)
    for fold, (fit_rel, pred_rel) in enumerate(inner.split(features[cfg.feature][outer_train], y[outer_train]), start=1):
        fit_idx = outer_train[fit_rel]
        pred_idx = outer_train[pred_rel]
        x_fit, x_pred = prepare_view(features, y, fit_idx, pred_idx, cfg)
        _, x_val = prepare_view(features, y, fit_idx, outer_val, cfg)
        gpu_oof, gpu_val = train_gpu_predict_many(x_fit, y[fit_idx], [x_pred, x_val], seed * 101 + fold, cfg)
        oof[f"{cfg.name}_gpu"][pred_rel] = gpu_oof
        val_parts[f"{cfg.name}_gpu"].append(gpu_val)
        for head_key, head in make_linear_heads(seed * 101 + fold, cfg).items():
            head.fit(x_fit, y[fit_idx])
            name = f"{cfg.name}_{head_key}"
            oof[name][pred_rel] = proba_or_score(head, x_pred)
            val_parts[name].append(proba_or_score(head, x_val))
        print(f"    {cfg.name} fold={fold} done", flush=True)
    val = {name: np.mean(parts, axis=0).astype(np.float32) for name, parts in val_parts.items()}
    return oof, val


def fit_base(features: dict[str, np.ndarray], y: np.ndarray, outer_train: np.ndarray, outer_val: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    raw = features["raw"]
    domain = features["domain"]
    oof = np.zeros(len(outer_train), dtype=np.float32)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 811)
    for fold, (fit_rel, pred_rel) in enumerate(inner.split(raw[outer_train], y[outer_train]), start=1):
        fit_idx = outer_train[fit_rel]
        pred_idx = outer_train[pred_rel]
        model = SpectralDomainSVC(seed=seed * 17 + fold, n_clusters=8, domain_blend=0.5)
        model.fit(raw[fit_idx], domain[fit_idx], y[fit_idx])
        oof[pred_rel] = model.predict_proba(raw[pred_idx], domain[pred_idx])[:, 1].astype(np.float32)
    final = SpectralDomainSVC(seed=seed, n_clusters=8, domain_blend=0.5)
    final.fit(raw[outer_train], domain[outer_train], y[outer_train])
    val = final.predict_proba(raw[outer_val], domain[outer_val])[:, 1].astype(np.float32)
    return oof, val


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))).astype(np.float32)


def add_fusions(oof: dict[str, np.ndarray], val: dict[str, np.ndarray]) -> None:
    atoms = [name for name in list(oof) if name != "base"]
    bo = logit(oof["base"])
    bm, bs = float(bo.mean()), float(bo.std() + 1e-6)
    for name in atoms:
        for weight in [0.03, 0.05, 0.08, 0.12, 0.18, 0.28, 0.35, 0.40]:
            key = f"fusion_{name}_{weight:.2f}"
            oof[key] = ((1.0 - weight) * oof["base"] + weight * oof[name]).astype(np.float32)
            val[key] = ((1.0 - weight) * val["base"] + weight * val[name]).astype(np.float32)
        co = logit(oof[name])
        cm, cs = float(co.mean()), float(co.std() + 1e-6)
        for weight in [0.08, 0.12, 0.18, 0.28, 0.35]:
            key = f"zfusion_{name}_{weight:.2f}"
            oof[key] = sigmoid((1.0 - weight) * ((bo - bm) / bs) + weight * ((co - cm) / cs))
            val[key] = sigmoid((1.0 - weight) * ((logit(val["base"]) - bm) / bs) + weight * ((logit(val[name]) - cm) / cs))


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    for threshold in np.linspace(0.35, 0.65, 61):
        acc = accuracy_score(y_true, prob >= threshold)
        if acc > best[1]:
            best = (float(threshold), float(acc))
    return best


def metric(
    seed: int,
    config: str,
    method: str,
    selected: str,
    y_train: np.ndarray,
    y_val: np.ndarray,
    oof_prob: np.ndarray,
    val_prob: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    pred = val_prob >= threshold
    return {
        "seed": seed,
        "config": config,
        "method": method,
        "selected": selected,
        "threshold": threshold,
        "oof_accuracy": accuracy_score(y_train, oof_prob >= threshold),
        "oof_auc": roc_auc_score(y_train, oof_prob),
        "accuracy": accuracy_score(y_val, pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, pred),
        "auc": roc_auc_score(y_val, val_prob),
        "pred_rate": float(pred.mean()),
    }


def run(seeds: list[int], configs: list[ViewConfig]) -> None:
    features, y = load_features()
    rows: list[dict[str, object]] = []
    detail: list[dict[str, object]] = []
    if OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
    if DETAIL.exists():
        detail = pd.read_csv(DETAIL).to_dict("records")
    done = {(int(r["seed"]), str(r["config"])) for r in rows if str(r.get("method")) == "base:fixed"}
    for seed in seeds:
        outer_train, outer_val = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(features["raw"], y))
        y_train = y[outer_train]
        y_val = y[outer_val]
        for cfg in configs:
            if (seed, cfg.name) in done:
                print(f"skip seed={seed} cfg={cfg.name}", flush=True)
                continue
            print(f"\nseed={seed} cfg={cfg}", flush=True)
            oof: dict[str, np.ndarray] = {}
            val: dict[str, np.ndarray] = {}
            oof["base"], val["base"] = fit_base(features, y, outer_train, outer_val, seed)
            view_oof, view_val = fit_view_oof_val(features, y, outer_train, outer_val, cfg, seed)
            oof.update(view_oof)
            val.update(view_val)
            add_fusions(oof, val)
            seed_rows: list[dict[str, object]] = []
            for name in sorted(oof):
                seed_rows.append(metric(seed, cfg.name, f"{name}:fixed", name, y_train, y_val, oof[name], val[name], 0.5))
                threshold, _ = best_threshold(y_train, oof[name])
                seed_rows.append(metric(seed, cfg.name, f"{name}:oof_thr", name, y_train, y_val, oof[name], val[name], threshold))
                detail.append(
                    {
                        "seed": seed,
                        "config": cfg.name,
                        "method": name,
                        "fixed_oof_accuracy": accuracy_score(y_train, oof[name] >= 0.5),
                        "oof_auc": roc_auc_score(y_train, oof[name]),
                    }
                )
            rows.extend(seed_rows)
            OUT.parent.mkdir(exist_ok=True)
            pd.DataFrame(rows).drop_duplicates(["seed", "config", "method"], keep="last").to_csv(OUT, index=False)
            pd.DataFrame(detail).drop_duplicates(["seed", "config", "method"], keep="last").to_csv(DETAIL, index=False)
            print(pd.DataFrame(seed_rows).sort_values(["accuracy", "auc"], ascending=False).head(20).to_string(index=False), flush=True)
    df = pd.DataFrame(rows).drop_duplicates(["seed", "config", "method"], keep="last")
    df.to_csv(OUT, index=False)
    summary = (
        df.groupby(["config", "method"])
        .agg(n=("seed", "nunique"), mean_acc=("accuracy", "mean"), min_acc=("accuracy", "min"), max_acc=("accuracy", "max"), mean_auc=("auc", "mean"))
        .sort_values(["n", "min_acc", "mean_acc"], ascending=False)
    )
    print("\nSUMMARY")
    print(summary.head(80).to_string(float_format=lambda value: f"{value:.4f}"), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2028")
    parser.add_argument("--configs", default="spec649")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    configs = [CONFIGS[item] for item in args.configs.split(",") if item]
    run(seeds, configs)


if __name__ == "__main__":
    main()
