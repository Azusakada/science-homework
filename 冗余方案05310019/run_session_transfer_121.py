from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from strict_erp_riemann_oof import CONFIGS as ERP_CONFIGS
from strict_erp_riemann_oof import fit_cfg_oof_val as fit_erp_oof_val
from strict_multiview_conv_embed import load_data
from strict_spec_gpu_oof import load_features
from strict_super_pool_oof import (
    PER_SEED_DIR,
    add_fixed_recipes,
    add_oof_ranked_recipes,
    add_prefixed,
    add_stackers,
    conv_oof_val,
    spec_oof_val,
)


ROOT = Path(__file__).resolve().parent
LABEL_CSV = ROOT / "data" / "train_labels.csv"


def fixed_metric(split_name: str, method: str, y_train: np.ndarray, y_test: np.ndarray, oof_prob: np.ndarray, test_prob: np.ndarray) -> dict[str, object]:
    pred = test_prob >= 0.5
    return {
        "split": split_name,
        "method": method,
        "selected": method,
        "oof_accuracy": accuracy_score(y_train, oof_prob >= 0.5),
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "pred_rate": float(pred.mean()),
    }


def session_indices(train_session: str, test_session: str) -> tuple[np.ndarray, np.ndarray]:
    labels = pd.read_csv(LABEL_CSV, usecols=["eeg_file"])
    train_mask = labels["eeg_file"].str.contains(train_session, regex=False).to_numpy()
    test_mask = labels["eeg_file"].str.contains(test_session, regex=False).to_numpy()
    if np.any(train_mask & test_mask):
        raise RuntimeError(f"overlap between {train_session} and {test_session}")
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(f"empty split: {train_session}={len(train_idx)}, {test_session}={len(test_idx)}")
    return train_idx, test_idx


def run_one(raw: np.ndarray, y: np.ndarray, features: dict[str, np.ndarray], train_session: str, test_session: str, seed: int) -> pd.DataFrame:
    split_name = f"{train_session}_train_{test_session}_test"
    train_idx, test_idx = session_indices(train_session, test_session)
    y_train = y[train_idx]
    y_test = y[test_idx]
    oof: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}
    print(f"\nseed={seed} {split_name}: train={len(train_idx)} test={len(test_idx)}", flush=True)

    print("  spec", flush=True)
    so, sv = spec_oof_val(features, y, train_idx, test_idx, seed)
    add_prefixed(oof, test, "spec", so, sv)

    print("  erp_late", flush=True)
    eo, ev = fit_erp_oof_val(raw, y, train_idx, test_idx, ERP_CONFIGS["erp_late"], seed)
    add_prefixed(oof, test, "erp", eo, ev)

    print("  conv", flush=True)
    co, cv, cdetail = conv_oof_val(raw, y, train_idx, test_idx, seed)
    add_prefixed(oof, test, "conv", co, cv)
    print(f"  conv_best_oof={cdetail['conv_best_oof']} conv_svc_oof={cdetail['conv_svc_oof']:.6f}", flush=True)

    add_stackers(oof, test, y_train, seed)
    add_fixed_recipes(oof, test)
    rank_detail = add_oof_ranked_recipes(oof, test, y_train)
    print(f"  oof_rank_pool_size={rank_detail['oof_rank_pool_size']} final_candidates={len(oof)}", flush=True)

    rows = [fixed_metric(split_name, name, y_train, y_test, oof[name], test[name]) for name in sorted(oof)]
    df = pd.DataFrame(rows).sort_values("method")
    df.insert(1, "run_seed", seed)
    if len(df) != 121:
        raise RuntimeError(f"{split_name} produced {len(df)} rows, expected 121")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2026")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    raw, y = load_data()
    features, y_features = load_features()
    if not np.array_equal(y, y_features):
        raise RuntimeError("label mismatch")

    PER_SEED_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        jobs = [
            ("sess1", "sess2", PER_SEED_DIR / f"session_seed{seed}_sess1_train_sess2_test.csv"),
            ("sess2", "sess1", PER_SEED_DIR / f"session_seed{seed}_sess2_train_sess1_test.csv"),
        ]
        if seed == 2026 and len(seeds) == 1:
            jobs = [
                ("sess1", "sess2", PER_SEED_DIR / "session_sess1_train_sess2_test.csv"),
                ("sess2", "sess1", PER_SEED_DIR / "session_sess2_train_sess1_test.csv"),
            ]
        for train_session, test_session, out_path in jobs:
            df = run_one(raw, y, features, train_session, test_session, seed)
            df.to_csv(out_path, index=False)
            best = df.sort_values(["accuracy", "method"], ascending=[False, True]).iloc[0]
            print(f"  wrote {out_path} rows={len(df)} best={best.method} acc={best.accuracy:.6f}", flush=True)


if __name__ == "__main__":
    main()
