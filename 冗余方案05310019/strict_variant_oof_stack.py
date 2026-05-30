from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
RAW_CACHE = ROOT / ".cache_strict_eeg" / "raw_amp_features_v1.joblib"
DOMAIN_CACHE = ROOT / ".cache_strict_eeg" / "domain_features_v1.joblib"
OUT = ROOT / "diagnostics" / "variant_oof_stack.csv"
SEEDS = [2026, 2027, 2028, 2029, 2030]


@dataclass(frozen=True)
class Variant:
    name: str
    k: int = 1000
    c: float = 1.2
    n_clusters: int = 5
    blend: float = 0.5
    local_min_size: int = 160
    local_min_class: int = 50


VARIANTS = [
    Variant("base", k=1000, c=1.2, n_clusters=5, blend=0.5),
    Variant("c8_k800_c12", k=800, c=1.2, n_clusters=8, blend=0.5),
    Variant("c8_k800_c14", k=800, c=1.4, n_clusters=8, blend=0.5),
    Variant("c8_k800_c16", k=800, c=1.6, n_clusters=8, blend=0.5),
    Variant("c8_k900_c12", k=900, c=1.2, n_clusters=8, blend=0.5),
    Variant("c8_k900_c14", k=900, c=1.4, n_clusters=8, blend=0.5),
    Variant("c8_k600_c16", k=600, c=1.6, n_clusters=8, blend=0.5),
    Variant("c8_k800_c14_loose", k=800, c=1.4, n_clusters=8, blend=0.5, local_min_size=120, local_min_class=35),
    Variant("c8_k800_c14_b65", k=800, c=1.4, n_clusters=8, blend=0.65),
    Variant("c7_k700_c16", k=700, c=1.6, n_clusters=7, blend=0.5),
    Variant("c6_k1000_c12", k=1000, c=1.2, n_clusters=6, blend=0.5),
    Variant("global_k800_c14", k=800, c=1.4, n_clusters=8, blend=0.0),
]


def make_svc(seed: int, variant: Variant, n_features: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(variant.k, n_features))),
            ("clf", SVC(C=variant.c, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
        ]
    )


class Model:
    def __init__(self, seed: int, variant: Variant):
        self.seed = seed
        self.variant = variant

    def fit(self, X: np.ndarray, D: np.ndarray, y: np.ndarray):
        v = self.variant
        self.global_ = make_svc(self.seed, v, X.shape[1])
        self.global_.fit(X, y)
        self.scaler_ = StandardScaler()
        z = self.scaler_.fit_transform(D)
        self.pca_ = PCA(n_components=min(10, z.shape[1]), random_state=self.seed)
        z = self.pca_.fit_transform(z)
        self.km_ = KMeans(n_clusters=v.n_clusters, n_init=30, random_state=self.seed)
        cl = self.km_.fit_predict(z)
        self.locals_: dict[int, Pipeline] = {}
        for cid in range(v.n_clusters):
            idx = np.flatnonzero(cl == cid)
            if len(idx) >= v.local_min_size and min(np.bincount(y[idx], minlength=2)) >= v.local_min_class:
                m = make_svc(self.seed + 100 + cid, v, X.shape[1])
                m.fit(X[idx], y[idx])
                self.locals_[cid] = m
        return self

    def predict_proba(self, X: np.ndarray, D: np.ndarray) -> np.ndarray:
        gp = self.global_.predict_proba(X)
        if self.variant.blend <= 1e-12:
            return gp
        z = self.pca_.transform(self.scaler_.transform(D))
        cl = self.km_.predict(z)
        lp = gp.copy()
        for cid, m in self.locals_.items():
            mask = cl == cid
            if np.any(mask):
                lp[mask] = m.predict_proba(X[mask])
        return (1.0 - self.variant.blend) * gp + self.variant.blend * lp


def best_threshold(y: np.ndarray, prob: np.ndarray, radius: float = 0.16) -> tuple[float, float]:
    best = (0.5, -1.0)
    for thr in np.linspace(max(0.05, 0.5 - radius), min(0.95, 0.5 + radius), 65):
        acc = accuracy_score(y, prob >= thr)
        if acc > best[1]:
            best = (float(thr), float(acc))
    return best


def metric_row(seed: int, name: str, y_true: np.ndarray, prob: np.ndarray, threshold: float, oof_acc: float | None) -> dict[str, object]:
    pred = prob >= threshold
    return {
        "seed": seed,
        "method": name,
        "threshold": threshold,
        "oof_accuracy": oof_acc,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "auc": roc_auc_score(y_true, prob),
    }


def stack_features(P: np.ndarray) -> np.ndarray:
    margin = np.abs(P - 0.5)
    return np.column_stack(
        [
            P,
            P.mean(axis=1),
            P.std(axis=1),
            np.median(P, axis=1),
            P.min(axis=1),
            P.max(axis=1),
            margin.mean(axis=1),
            margin.min(axis=1),
            (P >= 0.5).mean(axis=1),
        ]
    ).astype(np.float32)


def run() -> None:
    _, X, y = joblib.load(RAW_CACHE)
    D = joblib.load(DOMAIN_CACHE)
    X = X.astype(np.float32, copy=False)
    D = D.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)
    rows = []
    for seed in SEEDS:
        tr, va = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y))
        print(f"\nSEED {seed}", flush=True)
        oof_cols = []
        val_cols = []
        variant_oof_scores = []
        for vi, variant in enumerate(VARIANTS):
            oof = np.zeros(len(tr), dtype=np.float32)
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 300 + vi)
            for fold, (itr, iva) in enumerate(inner.split(X[tr], y[tr]), start=1):
                fit_idx = tr[itr]
                eval_idx = tr[iva]
                model = Model(seed * 17 + vi * 101 + fold, variant)
                model.fit(X[fit_idx], D[fit_idx], y[fit_idx])
                oof[iva] = model.predict_proba(X[eval_idx], D[eval_idx])[:, 1]
            full = Model(seed * 19 + vi * 101, variant)
            full.fit(X[tr], D[tr], y[tr])
            val_prob = full.predict_proba(X[va], D[va])[:, 1].astype(np.float32)
            oof_cols.append(oof)
            val_cols.append(val_prob)
            oof_score = accuracy_score(y[tr], oof >= 0.5)
            variant_oof_scores.append(oof_score)
            row = metric_row(seed, variant.name, y[va], val_prob, 0.5, oof_score)
            rows.append(row)
            print(variant.name, f"oof={oof_score:.4f}", f"val={row['accuracy']:.4f}", f"auc={row['auc']:.4f}", flush=True)

        O = np.column_stack(oof_cols)
        V = np.column_stack(val_cols)
        y_tr = y[tr]
        y_va = y[va]
        scores = np.asarray(variant_oof_scores)
        order = np.argsort(scores)[::-1]

        for k in [3, 5, 8, len(VARIANTS)]:
            cols = order[:k]
            for mode in ["mean", "median", "weighted"]:
                if mode == "mean":
                    o_prob = O[:, cols].mean(axis=1)
                    v_prob = V[:, cols].mean(axis=1)
                elif mode == "median":
                    o_prob = np.median(O[:, cols], axis=1)
                    v_prob = np.median(V[:, cols], axis=1)
                else:
                    w = np.maximum(scores[cols] - 0.5, 0.0)
                    w = w / w.sum() if w.sum() > 1e-8 else np.ones(len(cols)) / len(cols)
                    o_prob = O[:, cols] @ w
                    v_prob = V[:, cols] @ w
                for thr_mode in ["fixed", "oof_thr"]:
                    thr, oof_acc = (0.5, accuracy_score(y_tr, o_prob >= 0.5)) if thr_mode == "fixed" else best_threshold(y_tr, o_prob)
                    rows.append(metric_row(seed, f"{mode}_top{k}_{thr_mode}", y_va, v_prob, thr, oof_acc))

        Z_oof = stack_features(O)
        Z_val = stack_features(V)
        stackers = {
            "stack_log_c03": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=0.03, class_weight="balanced", max_iter=3000, random_state=seed))]),
            "stack_log_c08": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=0.08, class_weight="balanced", max_iter=3000, random_state=seed))]),
            "stack_l1_c03": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=0.03, penalty="l1", solver="liblinear", class_weight="balanced", max_iter=3000, random_state=seed))]),
            "stack_lda": Pipeline([("scale", StandardScaler()), ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]),
        }
        for name, stacker in stackers.items():
            stacker.fit(Z_oof, y_tr)
            o_prob = stacker.predict_proba(Z_oof)[:, 1]
            v_prob = stacker.predict_proba(Z_val)[:, 1]
            for thr_mode in ["fixed", "oof_thr"]:
                thr, oof_acc = (0.5, accuracy_score(y_tr, o_prob >= 0.5)) if thr_mode == "fixed" else best_threshold(y_tr, o_prob)
                row = metric_row(seed, f"{name}_{thr_mode}", y_va, v_prob, thr, oof_acc)
                rows.append(row)
                print(row["method"], f"oof={oof_acc:.4f}", f"val={row['accuracy']:.4f}", f"auc={row['auc']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    summary = (
        df.groupby("method")
        .agg(mean_acc=("accuracy", "mean"), min_acc=("accuracy", "min"), max_acc=("accuracy", "max"), mean_auc=("auc", "mean"))
        .sort_values(["min_acc", "mean_acc"], ascending=False)
    )
    print("\nSUMMARY")
    print(summary.head(40).to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    run()
