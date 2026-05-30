from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyriemann.estimation import ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from scipy import signal, stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC

from raw_eeg_final import LABEL_TO_INDEX, SpectralDomainSVC, extract_domain_features, extract_raw_amplitude_features, load_eeg


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "train"
LABEL_CSV = ROOT / "data" / "train_labels.csv"
CACHE = ROOT / ".cache_strict_eeg"
OUT = ROOT / "diagnostics" / "erp_riemann_oof.csv"
DETAIL = ROOT / "diagnostics" / "erp_riemann_oof_detail.csv"


@dataclass(frozen=True)
class FeatConfig:
    name: str
    kind: str
    window: tuple[int, int] = (0, 282)
    norm: str = "train_channel"
    components: int = 4
    estimator: str = "oas"
    bands: tuple[tuple[float, float], ...] = ((0.01, 0.08), (0.08, 0.18), (0.18, 0.45))
    k: int = 900


CONFIGS = {
    "erp_late": FeatConfig("erp_late", "erp_stats", (70, 282), "train_channel", k=1100),
    "erp_midlate": FeatConfig("erp_midlate", "erp_stats", (45, 250), "baseline_epoch", k=1200),
    "xdawn4_late": FeatConfig("xdawn4_late", "xdawn_ts", (60, 260), "baseline_epoch", components=4, k=900),
    "xdawn6_all": FeatConfig("xdawn6_all", "xdawn_ts", (0, 282), "train_channel", components=6, k=1200),
    "erp_cov4": FeatConfig("erp_cov4", "erp_cov", (50, 260), "baseline_epoch", components=4, k=1000),
    "fb_xdawn": FeatConfig("fb_xdawn", "fb_xdawn", (50, 270), "baseline_epoch", components=4, k=1600),
}


def load_xy() -> tuple[np.ndarray, np.ndarray]:
    labels = pd.read_csv(LABEL_CSV, usecols=["eeg_file", "label"])
    y = labels["label"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    raw = load_eeg(TRAIN_DIR, labels["eeg_file"].tolist()).astype(np.float32)
    return raw, y


def fit_norm(raw: np.ndarray, idx: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    x = raw[idx].astype(np.float32, copy=False)
    if mode == "train_channel":
        return {
            "mean": x.mean(axis=(0, 2), keepdims=True).astype(np.float32),
            "std": (x.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32),
        }
    if mode == "baseline_epoch":
        return {"mean": np.zeros((1, raw.shape[1], 1), dtype=np.float32), "std": np.ones((1, raw.shape[1], 1), dtype=np.float32)}
    if mode == "raw":
        return {"mean": np.zeros((1, raw.shape[1], 1), dtype=np.float32), "std": np.ones((1, raw.shape[1], 1), dtype=np.float32)}
    raise ValueError(mode)


def apply_norm(raw: np.ndarray, stats_: dict[str, np.ndarray], mode: str, window: tuple[int, int]) -> np.ndarray:
    x = raw.astype(np.float32, copy=False)
    if mode == "baseline_epoch":
        z = x - x[:, :, :50].mean(axis=2, keepdims=True)
        z = z / (z.std(axis=2, keepdims=True) + 1e-6)
    elif mode == "train_channel":
        z = (x - stats_["mean"]) / stats_["std"]
    elif mode == "raw":
        z = x
    else:
        raise ValueError(mode)
    a, b = window
    return np.nan_to_num(z[:, :, a:b].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def erp_features(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    smooth = signal.savgol_filter(x32, 15, 3, axis=2, mode="interp")
    pieces: list[np.ndarray] = []
    for width in [3, 6, 9, 15]:
        n = smooth.shape[2] // width
        if n > 0:
            pieces.append(smooth[:, :, : n * width].reshape(len(x32), x32.shape[1], n, width).mean(axis=3).reshape(len(x32), -1))
    rel_windows = []
    total = smooth.shape[2]
    for start in np.linspace(0, max(1, total - 24), 8).round().astype(int):
        rel_windows.append((int(start), int(min(total, start + max(18, total // 5)))))
    rel_windows += [(0, total // 2), (total // 2, total), (max(0, total - 70), total)]
    for a, b in rel_windows:
        if b <= a + 2:
            continue
        w = smooth[:, :, a:b]
        pieces.extend([w.mean(axis=2), w.std(axis=2), w.max(axis=2), w.min(axis=2), np.sqrt(np.mean(w * w, axis=2))])
    pieces.append(np.nan_to_num(stats.skew(smooth, axis=2, bias=False)))
    pieces.append(np.nan_to_num(stats.kurtosis(smooth, axis=2, bias=False)))
    return np.nan_to_num(np.hstack(pieces).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class XdawnTangent(BaseEstimator, TransformerMixin):
    def __init__(self, cfg: FeatConfig):
        self.cfg = cfg

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.est_ = XdawnCovariances(nfilter=self.cfg.components, estimator=self.cfg.estimator, xdawn_estimator="scm")
        cov = self.est_.fit_transform(X, y)
        self.ts_ = TangentSpace(metric="riemann")
        self.ts_.fit(cov, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        cov = self.est_.transform(X)
        return np.nan_to_num(self.ts_.transform(cov).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class ERPCovTangent(BaseEstimator, TransformerMixin):
    def __init__(self, cfg: FeatConfig):
        self.cfg = cfg

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.est_ = ERPCovariances(estimator=self.cfg.estimator, svd=self.cfg.components)
        cov = self.est_.fit_transform(X, y)
        self.ts_ = TangentSpace(metric="riemann")
        self.ts_.fit(cov, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        cov = self.est_.transform(X)
        return np.nan_to_num(self.ts_.transform(cov).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def bandpass(x: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    sos = signal.butter(3, band, btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=2).astype(np.float64)


def make_feature(
    raw: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    pred_idx: np.ndarray,
    cfg: FeatConfig,
) -> tuple[np.ndarray, np.ndarray]:
    norm_stats = fit_norm(raw, fit_idx, cfg.norm)
    x_fit = apply_norm(raw[fit_idx], norm_stats, cfg.norm, cfg.window)
    x_pred = apply_norm(raw[pred_idx], norm_stats, cfg.norm, cfg.window)
    if cfg.kind == "erp_stats":
        return erp_features(x_fit), erp_features(x_pred)
    if cfg.kind == "xdawn_ts":
        tr = XdawnTangent(cfg).fit(x_fit, y[fit_idx])
        return tr.transform(x_fit), tr.transform(x_pred)
    if cfg.kind == "erp_cov":
        tr = ERPCovTangent(cfg).fit(x_fit, y[fit_idx])
        return tr.transform(x_fit), tr.transform(x_pred)
    if cfg.kind == "fb_xdawn":
        fit_parts = []
        pred_parts = []
        for band in cfg.bands:
            xb_fit = bandpass(x_fit, band)
            xb_pred = bandpass(x_pred, band)
            tr = XdawnTangent(cfg).fit(xb_fit, y[fit_idx])
            fit_parts.append(tr.transform(xb_fit))
            pred_parts.append(tr.transform(xb_pred))
            fit_parts.append(erp_features(xb_fit)[:, : 2 * xb_fit.shape[1]])
            pred_parts.append(erp_features(xb_pred)[:, : 2 * xb_pred.shape[1]])
        return np.hstack(fit_parts).astype(np.float32), np.hstack(pred_parts).astype(np.float32)
    raise ValueError(cfg.kind)


def make_heads(seed: int, n: int) -> dict[str, object]:
    k1 = min(700, n)
    k2 = min(1200, n)
    return {
        "log_c03": Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_classif, k=k2)), ("clf", LogisticRegression(C=0.03, class_weight="balanced", max_iter=5000, random_state=seed))]),
        "log_c08": Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_classif, k=k2)), ("clf", LogisticRegression(C=0.08, class_weight="balanced", max_iter=5000, random_state=seed))]),
        "ridge_a8": Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_classif, k=k2)), ("clf", RidgeClassifier(alpha=8.0, class_weight="balanced"))]),
        "lda": Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_classif, k=k1)), ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]),
        "svc_c07": Pipeline([("sc", StandardScaler()), ("sel", SelectKBest(f_classif, k=k1)), ("clf", SVC(C=0.7, gamma="scale", probability=True, class_weight="balanced", random_state=seed))]),
        "et_l8": ExtraTreesClassifier(n_estimators=600, max_features=0.55, min_samples_leaf=8, class_weight="balanced", n_jobs=-1, random_state=seed),
    }


def proba_or_score(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(np.float32)
    score = model.decision_function(x).astype(np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0)))).astype(np.float32)


def fit_cfg_oof_val(raw: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, cfg: FeatConfig, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    names: list[str] | None = None
    oof: dict[str, np.ndarray] = {}
    val_parts: dict[str, list[np.ndarray]] = {}
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 6103)
    for fold, (fit_rel, pred_rel) in enumerate(inner.split(raw[train_idx], y[train_idx]), start=1):
        fit_idx = train_idx[fit_rel]
        pred_idx = train_idx[pred_rel]
        x_fit, x_pred = make_feature(raw, y, fit_idx, pred_idx, cfg)
        _, x_val = make_feature(raw, y, fit_idx, val_idx, cfg)
        heads = make_heads(seed * 101 + fold, x_fit.shape[1])
        if names is None:
            names = [f"{cfg.name}_{name}" for name in heads]
            oof = {name: np.zeros(len(train_idx), dtype=np.float32) for name in names}
            val_parts = {name: [] for name in names}
        for head_name, head in heads.items():
            name = f"{cfg.name}_{head_name}"
            model = clone(head)
            model.fit(x_fit, y[fit_idx])
            oof[name][pred_rel] = proba_or_score(model, x_pred)
            val_parts[name].append(proba_or_score(model, x_val))
        print(f"    {cfg.name} fold={fold}/5", flush=True)
    assert names is not None
    val = {name: np.mean(parts, axis=0).astype(np.float32) for name, parts in val_parts.items()}
    return oof, val


def fit_base_oof_val(raw_features: np.ndarray, domain_features: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(train_idx), dtype=np.float32)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 719)
    for fold, (fit_rel, pred_rel) in enumerate(inner.split(raw_features[train_idx], y[train_idx]), start=1):
        fit_idx = train_idx[fit_rel]
        pred_idx = train_idx[pred_rel]
        model = SpectralDomainSVC(seed=seed * 17 + fold, n_clusters=8, domain_blend=0.5)
        model.fit(raw_features[fit_idx], domain_features[fit_idx], y[fit_idx])
        oof[pred_rel] = model.predict_proba(raw_features[pred_idx], domain_features[pred_idx])[:, 1].astype(np.float32)
    final = SpectralDomainSVC(seed=seed, n_clusters=8, domain_blend=0.5)
    final.fit(raw_features[train_idx], domain_features[train_idx], y[train_idx])
    val = final.predict_proba(raw_features[val_idx], domain_features[val_idx])[:, 1].astype(np.float32)
    return oof, val


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))).astype(np.float32)


def add_fusions(oof: dict[str, np.ndarray], val: dict[str, np.ndarray]) -> None:
    base_o = oof["base"]
    base_v = val["base"]
    bo = logit(base_o)
    bm, bs = float(bo.mean()), float(bo.std() + 1e-6)
    atoms = [name for name in list(oof) if name != "base"]
    for name in atoms:
        for weight in [0.03, 0.05, 0.08, 0.12, 0.18, 0.28, 0.40]:
            key = f"fusion_{name}_{weight:.2f}"
            oof[key] = ((1.0 - weight) * base_o + weight * oof[name]).astype(np.float32)
            val[key] = ((1.0 - weight) * base_v + weight * val[name]).astype(np.float32)
        ao = logit(oof[name])
        am, ass = float(ao.mean()), float(ao.std() + 1e-6)
        for weight in [0.05, 0.10, 0.18, 0.28]:
            key = f"zfusion_{name}_{weight:.2f}"
            oof[key] = sigmoid((1.0 - weight) * ((bo - bm) / bs) + weight * ((ao - am) / ass))
            val[key] = sigmoid((1.0 - weight) * ((logit(base_v) - bm) / bs) + weight * ((logit(val[name]) - am) / ass))


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    for threshold in np.linspace(0.38, 0.62, 49):
        acc = accuracy_score(y_true, prob >= threshold)
        if acc > best[1]:
            best = (float(threshold), float(acc))
    return best


def metric(seed: int, config: str, method: str, selected: str, y_train: np.ndarray, y_val: np.ndarray, oof_prob: np.ndarray, val_prob: np.ndarray, threshold: float) -> dict[str, object]:
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


def run(seeds: list[int], configs: list[FeatConfig], resume: bool) -> None:
    raw, y = load_xy()
    raw_cache = CACHE / "raw_amp_features_v1.joblib"
    domain_cache = CACHE / "domain_features_v1.joblib"
    if raw_cache.exists():
        _, raw_features, yy = joblib.load(raw_cache)
        if not np.array_equal(y, yy):
            raise RuntimeError("label mismatch")
    else:
        raw_features = extract_raw_amplitude_features(raw)
    if domain_cache.exists():
        domain_features = joblib.load(domain_cache)
    else:
        domain_features = extract_domain_features(raw)
        joblib.dump(domain_features, domain_cache, compress=3)
    raw_features = raw_features.astype(np.float32, copy=False)
    domain_features = domain_features.astype(np.float32, copy=False)
    rows: list[dict[str, object]] = []
    detail: list[dict[str, object]] = []
    if resume and OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
    if resume and DETAIL.exists():
        detail = pd.read_csv(DETAIL).to_dict("records")
    done = {(int(r["seed"]), str(r["config"])) for r in rows if str(r.get("method")) == "base:fixed"}
    for seed in seeds:
        train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(raw_features, y))
        y_train = y[train_idx]
        y_val = y[val_idx]
        for cfg in configs:
            if (seed, cfg.name) in done:
                print(f"skip seed={seed} cfg={cfg.name}", flush=True)
                continue
            print(f"\nseed={seed} cfg={cfg}", flush=True)
            oof: dict[str, np.ndarray] = {}
            val: dict[str, np.ndarray] = {}
            oof["base"], val["base"] = fit_base_oof_val(raw_features, domain_features, y, train_idx, val_idx, seed)
            cfg_oof, cfg_val = fit_cfg_oof_val(raw, y, train_idx, val_idx, cfg, seed)
            oof.update(cfg_oof)
            val.update(cfg_val)
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
            pool = [name for name in oof if name == "base" or name.startswith("fusion_") or name.startswith("zfusion_")]
            best_fixed = max(pool, key=lambda name: (accuracy_score(y_train, oof[name] >= 0.5), roc_auc_score(y_train, oof[name]), name))
            best_auc = max(pool, key=lambda name: (roc_auc_score(y_train, oof[name]), accuracy_score(y_train, oof[name] >= 0.5), name))
            for selected, pick_name in [(best_fixed, "pick_fixed"), (best_auc, "pick_auc")]:
                seed_rows.append(metric(seed, cfg.name, f"{pick_name}:fixed", selected, y_train, y_val, oof[selected], val[selected], 0.5))
                threshold, _ = best_threshold(y_train, oof[selected])
                seed_rows.append(metric(seed, cfg.name, f"{pick_name}:oof_thr", selected, y_train, y_val, oof[selected], val[selected], threshold))
            rows.extend(seed_rows)
            OUT.parent.mkdir(exist_ok=True)
            pd.DataFrame(rows).drop_duplicates(["seed", "config", "method"], keep="last").to_csv(OUT, index=False)
            pd.DataFrame(detail).drop_duplicates(["seed", "config", "method"], keep="last").to_csv(DETAIL, index=False)
            print(pd.DataFrame(seed_rows).sort_values(["accuracy", "auc"], ascending=False).head(24).to_string(index=False), flush=True)
    df = pd.DataFrame(rows).drop_duplicates(["seed", "config", "method"], keep="last")
    df.to_csv(OUT, index=False)
    summary = (
        df.groupby(["config", "method"])
        .agg(n=("seed", "nunique"), mean_acc=("accuracy", "mean"), min_acc=("accuracy", "min"), max_acc=("accuracy", "max"), mean_auc=("auc", "mean"), mean_oof=("oof_accuracy", "mean"))
        .sort_values(["n", "min_acc", "mean_acc"], ascending=False)
    )
    print("\nSUMMARY")
    print(summary.head(120).to_string(float_format=lambda value: f"{value:.4f}"), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2028")
    parser.add_argument("--configs", default="erp_late")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    configs = [CONFIGS[item] for item in args.configs.split(",") if item]
    run(seeds, configs, resume=not args.no_resume)


if __name__ == "__main__":
    main()
