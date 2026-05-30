from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABEL_TO_INDEX = {"background": 0, "target": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


def load_eeg(data_dir: str | Path, files: Iterable[str]) -> np.ndarray:
    data_dir = Path(data_dir)
    return np.stack([np.load(data_dir / file_name).astype(np.float32).squeeze() * 1e6 for file_name in files])


def extract_raw_amplitude_features(raw: np.ndarray) -> np.ndarray:
    x = raw.astype(np.float32, copy=False)
    smooth = signal.savgol_filter(x, window_length=15, polyorder=3, axis=2, mode="interp")
    pieces = []
    for width in [6, 10, 15, 20, 30]:
        n_bins = x.shape[2] // width
        pieces.append(smooth[:, :, : n_bins * width].reshape(len(x), 59, n_bins, width).mean(axis=3).reshape(len(x), -1))

    windows = []
    for start in range(0, 282, 20):
        windows.append((start, min(282, start + 40)))
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

    def fit(self, raw_features: np.ndarray, domain_features: np.ndarray, y: np.ndarray):
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
        self.classes_ = np.asarray([0, 1])
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

    def predict(self, raw_features: np.ndarray, domain_features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(raw_features, domain_features)[:, 1] >= 0.5).astype(int)


def validate(raw: np.ndarray, y: np.ndarray, seeds: Iterable[int] = (2026, 2027, 2028, 2029, 2030)) -> pd.DataFrame:
    raw_features = extract_raw_amplitude_features(raw)
    domain_features = extract_domain_features(raw)
    rows = []
    for seed in seeds:
        train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(raw_features, y))
        model = SpectralDomainSVC(seed=seed, domain_blend=0.5)
        model.fit(raw_features[train_idx], domain_features[train_idx], y[train_idx])
        prob = model.predict_proba(raw_features[val_idx], domain_features[val_idx])[:, 1]
        pred = prob >= 0.5
        rows.append(
            {
                "seed": seed,
                "accuracy": accuracy_score(y[val_idx], pred),
                "balanced_accuracy": balanced_accuracy_score(y[val_idx], pred),
                "auc": roc_auc_score(y[val_idx], prob),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
        )
    return pd.DataFrame(rows)


def train_and_save(train_dir: str | Path, label_csv: str | Path, model_path: str | Path, seed: int = 2026, run_cv: bool = True) -> pd.DataFrame:
    labels = pd.read_csv(label_csv)
    files = labels["eeg_file"].tolist()
    y = labels["label"].map(LABEL_TO_INDEX).to_numpy()
    raw = load_eeg(train_dir, files)
    cv = validate(raw, y) if run_cv else pd.DataFrame()
    raw_features = extract_raw_amplitude_features(raw)
    domain_features = extract_domain_features(raw)
    model = SpectralDomainSVC(seed=seed, domain_blend=0.5)
    model.fit(raw_features, domain_features, y)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "seed": seed, "cv": cv, "kind": "spectral_domain_svc"}, model_path)
    return cv


def predict_saved(model_path: str | Path, test_dir: str | Path) -> pd.DataFrame:
    payload = joblib.load(model_path)
    files = sorted(path.name for path in Path(test_dir).glob("*.npy"))
    raw = load_eeg(test_dir, files)
    raw_features = extract_raw_amplitude_features(raw)
    domain_features = extract_domain_features(raw)
    pred = payload["model"].predict(raw_features, domain_features)
    return pd.DataFrame({"eeg_file": files, "prediction": [INDEX_TO_LABEL[int(i)] for i in pred]})
