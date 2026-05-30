from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from strict_conv_oof_gate import build_probs as build_conv_probs
from strict_erp_riemann_oof import CONFIGS as ERP_CONFIGS
from strict_erp_riemann_oof import fit_cfg_oof_val as fit_erp_oof_val
from strict_multiview_conv_embed import Config as ConvConfig
from strict_multiview_conv_embed import load_data
from strict_spec_gpu_oof import CONFIGS as SPEC_CONFIGS
from strict_spec_gpu_oof import add_fusions as add_spec_fusions
from strict_spec_gpu_oof import fit_base, fit_view_oof_val, load_features
from raw_eeg_final import extract_domain_features, extract_raw_amplitude_features


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "diagnostics" / "super_pool_oof.csv"
DETAIL = ROOT / "diagnostics" / "super_pool_oof_detail.csv"
PER_SEED_DIR = ROOT / "diagnostics" / "per_seed"


def logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def sigmoid(score: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(score, -50.0, 50.0)))).astype(np.float32)


def proba_or_score(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(np.float32)
    return sigmoid(model.decision_function(x).astype(np.float32))


def metric(
    seed: int,
    method: str,
    selected: str,
    y_train: np.ndarray,
    y_val: np.ndarray,
    oof_prob: np.ndarray,
    val_prob: np.ndarray,
) -> dict[str, object]:
    pred = val_prob >= 0.5
    return {
        "seed": seed,
        "method": method,
        "selected": selected,
        "oof_accuracy": accuracy_score(y_train, oof_prob >= 0.5),
        "accuracy": accuracy_score(y_val, pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, pred),
        "pred_rate": float(pred.mean()),
    }


def add_prefixed(dst_oof: dict[str, np.ndarray], dst_val: dict[str, np.ndarray], prefix: str, oof: dict[str, np.ndarray], val: dict[str, np.ndarray]) -> None:
    for name, prob in oof.items():
        if name in val:
            key = f"{prefix}_{name}"
            dst_oof[key] = prob.astype(np.float32, copy=False)
            dst_val[key] = val[name].astype(np.float32, copy=False)


def conv_oof_val(raw: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    raw_features = extract_raw_amplitude_features(raw)
    domain_features = extract_domain_features(raw)
    inner_cfg = ConvConfig(mode="multi", epochs=42)
    final_cfg = ConvConfig(mode="multi", epochs=55)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 4401)
    method_names: list[str] | None = None
    oof: dict[str, np.ndarray] = {}
    for fold, (fit_rel, eval_rel) in enumerate(inner.split(raw_features[train_idx], y[train_idx]), start=1):
        fit_idx = train_idx[fit_rel]
        eval_idx = train_idx[eval_rel]
        fold_probs = build_conv_probs(raw_features, domain_features, raw, y, fit_idx, eval_idx, seed * 31 + fold, inner_cfg)
        if method_names is None:
            method_names = sorted(fold_probs)
            oof = {name: np.zeros(len(train_idx), dtype=np.float32) for name in method_names}
        for name in method_names:
            oof[name][eval_rel] = fold_probs[name]
        print(f"    conv fold={fold}/3", flush=True)
    assert method_names is not None
    val = build_conv_probs(raw_features, domain_features, raw, y, train_idx, val_idx, seed, final_cfg)

    y_train = y[train_idx]
    svc_score = accuracy_score(y_train, oof["svc"] >= 0.5)
    whitelist = [name for name in method_names if name == "svc" or name.startswith("fusion_")]
    best_white = max(whitelist, key=lambda name: (accuracy_score(y_train, oof[name] >= 0.5), name))
    for margin in [0.008, 0.012, 0.016]:
        chosen = best_white if accuracy_score(y_train, oof[best_white] >= 0.5) >= svc_score + margin else "svc"
        oof[f"gate{margin:.3f}"] = oof[chosen]
        val[f"gate{margin:.3f}"] = val[chosen]
    return oof, val, {"conv_best_oof": best_white, "conv_svc_oof": svc_score}


def spec_oof_val(features: dict[str, np.ndarray], y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    oof: dict[str, np.ndarray] = {}
    val: dict[str, np.ndarray] = {}
    oof["base"], val["base"] = fit_base(features, y, train_idx, val_idx, seed)
    view_oof, view_val = fit_view_oof_val(features, y, train_idx, val_idx, SPEC_CONFIGS["spec649"], seed)
    oof.update(view_oof)
    val.update(view_val)
    add_spec_fusions(oof, val)
    return oof, val


def add_fixed_recipes(oof: dict[str, np.ndarray], val: dict[str, np.ndarray]) -> None:
    recipe_defs = {
        "recipe_conv_spec_erp": [
            "conv_gate0.012",
            "spec_zfusion_spec649_gpu_0.18",
            "erp_fusion_erp_late_ridge_a8_0.08",
        ],
        "recipe_four_core": [
            "conv_gate0.012",
            "spec_zfusion_spec649_gpu_0.18",
            "erp_fusion_erp_late_ridge_a8_0.08",
        ],
        "recipe_robust_core": [
            "spec_zfusion_spec649_ridge_a5_0.18",
            "erp_fusion_erp_late_ridge_a8_0.08",
            "conv_gate0.012",
        ],
    }
    for name, members in recipe_defs.items():
        members = [member for member in members if member in oof and member in val]
        if len(members) < 2:
            continue
        O = np.column_stack([oof[member] for member in members])
        V = np.column_stack([val[member] for member in members])
        oof[f"{name}_mean"] = O.mean(axis=1).astype(np.float32)
        val[f"{name}_mean"] = V.mean(axis=1).astype(np.float32)
        oof[f"{name}_median"] = np.median(O, axis=1).astype(np.float32)
        val[f"{name}_median"] = np.median(V, axis=1).astype(np.float32)
        oz = np.column_stack([logit(oof[member]) for member in members])
        vz = np.column_stack([logit(val[member]) for member in members])
        mean = oz.mean(axis=0, keepdims=True)
        std = oz.std(axis=0, keepdims=True) + 1e-6
        oof[f"{name}_zmean"] = sigmoid(((oz - mean) / std).mean(axis=1))
        val[f"{name}_zmean"] = sigmoid(((vz - mean) / std).mean(axis=1))


def add_oof_ranked_recipes(oof: dict[str, np.ndarray], val: dict[str, np.ndarray], y_train: np.ndarray) -> dict[str, object]:
    base_pool = sorted(name for name in oof if name in val)
    scored = sorted(base_pool, key=lambda name: (accuracy_score(y_train, oof[name] >= 0.5), name), reverse=True)
    detail: dict[str, object] = {
        "oof_rank_pool_size": len(base_pool),
        "oof_rank_pool_members": "|".join(base_pool),
    }
    for k in [3, 5, 8, 12]:
        members = scored[: min(k, len(scored))]
        detail[f"oof_rank_mean{k}_members"] = "|".join(members)
        if len(members) < 2:
            continue
        O = np.column_stack([oof[member] for member in members])
        V = np.column_stack([val[member] for member in members])
        oof[f"oof_rank_mean{k}"] = O.mean(axis=1).astype(np.float32)
        val[f"oof_rank_mean{k}"] = V.mean(axis=1).astype(np.float32)
        oof[f"oof_rank_median{k}"] = np.median(O, axis=1).astype(np.float32)
        val[f"oof_rank_median{k}"] = np.median(V, axis=1).astype(np.float32)
    return detail


def add_stackers(oof: dict[str, np.ndarray], val: dict[str, np.ndarray], y_train: np.ndarray, seed: int) -> None:
    pool = sorted(name for name in oof if name in val)
    if len(pool) < 4:
        return
    Xo = np.column_stack([oof[name] for name in pool])
    Xv = np.column_stack([val[name] for name in pool])
    extra_o = np.column_stack(
        [
            Xo.mean(axis=1),
            np.median(Xo, axis=1),
            Xo.std(axis=1),
            np.abs(Xo - 0.5).mean(axis=1),
            (Xo >= 0.5).mean(axis=1),
        ]
    )
    extra_v = np.column_stack(
        [
            Xv.mean(axis=1),
            np.median(Xv, axis=1),
            Xv.std(axis=1),
            np.abs(Xv - 0.5).mean(axis=1),
            (Xv >= 0.5).mean(axis=1),
        ]
    )
    Xo = np.column_stack([Xo, extra_o]).astype(np.float32)
    Xv = np.column_stack([Xv, extra_v]).astype(np.float32)
    stackers = {
        "stack_log_c003": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.003, class_weight="balanced", max_iter=5000, random_state=seed))]),
        "stack_log_c01": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000, random_state=seed))]),
        "stack_lda": Pipeline([("sc", StandardScaler()), ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]),
    }
    for name, model in stackers.items():
        oof_pred = np.zeros(len(y_train), dtype=np.float32)
        val_pred_folds: list[np.ndarray] = []
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 9227)
        for fit_rel, eval_rel in inner.split(Xo, y_train):
            fold_model = clone(model)
            fold_model.fit(Xo[fit_rel], y_train[fit_rel])
            oof_pred[eval_rel] = proba_or_score(fold_model, Xo[eval_rel])
            val_pred_folds.append(proba_or_score(fold_model, Xv))
        final_model = clone(model)
        final_model.fit(Xo, y_train)
        val_pred_folds.append(proba_or_score(final_model, Xv))
        oof[name] = oof_pred
        val[name] = np.mean(np.column_stack(val_pred_folds), axis=1).astype(np.float32)


def run(seeds: list[int], resume: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    raw, y = load_data()
    features, y_features = load_features()
    if not np.array_equal(y, y_features):
        raise RuntimeError("label mismatch")
    rows: list[dict[str, object]] = []
    detail: list[dict[str, object]] = []
    if resume and OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
    if resume and DETAIL.exists():
        detail = pd.read_csv(DETAIL).to_dict("records")
    done = {int(row["seed"]) for row in detail if str(row.get("stage")) == "done"}
    for seed in seeds:
        if seed in done:
            print(f"skip seed={seed}", flush=True)
            continue
        train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(raw, y))
        y_train = y[train_idx]
        y_val = y[val_idx]
        oof: dict[str, np.ndarray] = {}
        val: dict[str, np.ndarray] = {}
        print(f"\nSEED {seed}", flush=True)

        print("  spec", flush=True)
        so, sv = spec_oof_val(features, y, train_idx, val_idx, seed)
        add_prefixed(oof, val, "spec", so, sv)

        print("  erp_late", flush=True)
        eo, ev = fit_erp_oof_val(raw, y, train_idx, val_idx, ERP_CONFIGS["erp_late"], seed)
        add_prefixed(oof, val, "erp", eo, ev)

        print("  conv", flush=True)
        co, cv, cdetail = conv_oof_val(raw, y, train_idx, val_idx, seed)
        add_prefixed(oof, val, "conv", co, cv)

        add_stackers(oof, val, y_train, seed)
        add_fixed_recipes(oof, val)
        rank_detail = add_oof_ranked_recipes(oof, val, y_train)

        seed_rows: list[dict[str, object]] = []
        for name in sorted(oof):
            seed_rows.append(metric(seed, name, name, y_train, y_val, oof[name], val[name]))

        rows.extend(seed_rows)
        detail.append(
            {
                "seed": seed,
                "stage": "done",
                "n_candidates": len(oof),
                **cdetail,
                **rank_detail,
                "final_oof_diagnostic_size": len(oof),
                "final_oof_diagnostic_members": "|".join(sorted(oof)),
                "best_val": max(seed_rows, key=lambda row: row["accuracy"])["method"],
                "best_oof": max(seed_rows, key=lambda row: row["oof_accuracy"])["method"],
            }
        )
        OUT.parent.mkdir(exist_ok=True)
        PER_SEED_DIR.mkdir(parents=True, exist_ok=True)
        seed_df = pd.DataFrame(seed_rows).sort_values("method")
        seed_df.to_csv(PER_SEED_DIR / f"seed_{seed}.csv", index=False)
        pd.DataFrame(rows).drop_duplicates(["seed", "method"], keep="last").to_csv(OUT, index=False)
        pd.DataFrame(detail).to_csv(DETAIL, index=False)
        print(seed_df.sort_values(["accuracy", "method"], ascending=[False, True]).head(32).to_string(index=False), flush=True)

    df = pd.DataFrame(rows).drop_duplicates(["seed", "method"], keep="last")
    df.to_csv(OUT, index=False)
    if not df.empty:
        summary = (
            df.groupby("method")
            .agg(n=("seed", "nunique"), mean_acc=("accuracy", "mean"), min_acc=("accuracy", "min"), max_acc=("accuracy", "max"), mean_oof=("oof_accuracy", "mean"))
            .sort_values(["n", "min_acc", "mean_acc"], ascending=False)
        )
        print("\nSUMMARY")
        print(summary.head(120).to_string(float_format=lambda value: f"{value:.4f}"), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2028,2029,2030")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    run(seeds, resume=not args.no_resume)


if __name__ == "__main__":
    main()
