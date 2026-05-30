from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from raw_eeg_final import SpectralDomainSVC, extract_domain_features, extract_raw_amplitude_features
from strict_multiview_conv_embed import (
    Config,
    extract,
    fit_stats,
    load_data,
    make_views,
    train_net,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "diagnostics" / "conv_oof_gate.csv"
DETAIL = ROOT / "diagnostics" / "conv_oof_gate_details.csv"
SEEDS = [2026, 2027, 2028, 2029, 2030]


def make_heads(seed: int) -> dict[str, object]:
    return {
        "z_log": Pipeline(
            [("sc", StandardScaler()), ("clf", LogisticRegression(C=0.18, class_weight="balanced", max_iter=3000, random_state=seed))]
        ),
        "z_svc": Pipeline(
            [("sc", StandardScaler()), ("clf", SVC(C=0.9, gamma="scale", probability=True, class_weight="balanced", random_state=seed))]
        ),
        "zp_log": Pipeline(
            [("sc", StandardScaler()), ("clf", LogisticRegression(C=0.18, class_weight="balanced", max_iter=3000, random_state=seed))]
        ),
        "z_et": ExtraTreesClassifier(
            n_estimators=600,
            max_features=0.55,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def build_probs(
    raw_features: np.ndarray,
    domain_features: np.ndarray,
    raw: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    seed: int,
    cfg: Config,
) -> dict[str, np.ndarray]:
    stats = fit_stats(raw, fit_idx)
    X = make_views(raw, stats, cfg.mode)
    net = train_net(X, y, fit_idx, seed, cfg)
    z_fit, p_fit, y_fit = extract(net, X, y, fit_idx)
    z_eval, p_eval, _ = extract(net, X, y, eval_idx)

    svc = SpectralDomainSVC(seed=seed, n_clusters=8, domain_blend=0.5)
    svc.fit(raw_features[fit_idx], domain_features[fit_idx], y[fit_idx])
    svc_prob = svc.predict_proba(raw_features[eval_idx], domain_features[eval_idx])[:, 1]

    probs: dict[str, np.ndarray] = {"svc": svc_prob.astype(np.float32), "net": p_eval.astype(np.float32)}
    for name, head in make_heads(seed).items():
        if name == "zp_log":
            head.fit(np.column_stack([z_fit, p_fit]), y_fit)
            probs[name] = head.predict_proba(np.column_stack([z_eval, p_eval]))[:, 1].astype(np.float32)
        else:
            head.fit(z_fit, y_fit)
            probs[name] = head.predict_proba(z_eval)[:, 1].astype(np.float32)

    for base in ["net", "z_log", "z_svc", "zp_log", "z_et"]:
        for weight in [0.10, 0.20, 0.35]:
            probs[f"fusion_{base}_{weight:.2f}"] = ((1.0 - weight) * probs["svc"] + weight * probs[base]).astype(np.float32)
    return probs


def fixed_metric(seed: int, method: str, selected: str, y_true: np.ndarray, prob: np.ndarray, oof_acc: float, svc_oof_acc: float) -> dict[str, object]:
    pred = prob >= 0.5
    return {
        "seed": seed,
        "method": method,
        "selected": selected,
        "oof_accuracy": oof_acc,
        "svc_oof_accuracy": svc_oof_acc,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "auc": roc_auc_score(y_true, prob),
    }


def run() -> None:
    raw, y = load_data()
    raw_features = extract_raw_amplitude_features(raw)
    domain_features = extract_domain_features(raw)
    outer_rows = []
    detail_rows = []
    inner_cfg = Config(mode="multi", epochs=42)
    final_cfg = Config(mode="multi", epochs=55)

    for seed in SEEDS:
        outer_train, outer_val = next(
            StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(raw_features, y)
        )
        print(f"\nSEED {seed}", flush=True)

        method_names: list[str] | None = None
        oof_probs: dict[str, np.ndarray] = {}
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 4401)
        for fold, (fit_rel, eval_rel) in enumerate(inner.split(raw_features[outer_train], y[outer_train]), start=1):
            fit_idx = outer_train[fit_rel]
            eval_idx = outer_train[eval_rel]
            fold_probs = build_probs(raw_features, domain_features, raw, y, fit_idx, eval_idx, seed * 31 + fold, inner_cfg)
            if method_names is None:
                method_names = sorted(fold_probs)
                oof_probs = {name: np.zeros(len(outer_train), dtype=np.float32) for name in method_names}
            for name in method_names:
                oof_probs[name][eval_rel] = fold_probs[name]
            print(f"  fold={fold} done", flush=True)

        assert method_names is not None
        y_outer_train = y[outer_train]
        oof_scores = {name: accuracy_score(y_outer_train, oof_probs[name] >= 0.5) for name in method_names}
        svc_oof = oof_scores["svc"]
        for name in method_names:
            detail_rows.append(
                {
                    "seed": seed,
                    "method": name,
                    "oof_accuracy": oof_scores[name],
                    "oof_auc": roc_auc_score(y_outer_train, oof_probs[name]),
                }
            )

        final_probs = build_probs(raw_features, domain_features, raw, y, outer_train, outer_val, seed, final_cfg)
        y_val = y[outer_val]
        for name in method_names:
            row = fixed_metric(seed, name, name, y_val, final_probs[name], oof_scores[name], svc_oof)
            outer_rows.append(row)

        # Conservative gate: only trust convolution fusion when train-only OOF clears the SVC by a real margin.
        ranked = sorted(method_names, key=lambda name: (oof_scores[name], -len(name)), reverse=True)
        best_name = ranked[0]
        for margin in [0.000, 0.004, 0.008, 0.012, 0.016, 0.020]:
            chosen = best_name if oof_scores[best_name] >= svc_oof + margin else "svc"
            row = fixed_metric(seed, f"gate_margin_{margin:.3f}", chosen, y_val, final_probs[chosen], oof_scores[chosen], svc_oof)
            outer_rows.append(row)

        # Fixed whitelist avoids selecting weak standalone conv heads even when OOF is noisy.
        whitelist = [name for name in method_names if name == "svc" or name.startswith("fusion_")]
        best_white = max(whitelist, key=lambda name: oof_scores[name])
        for margin in [0.000, 0.004, 0.008, 0.012, 0.016, 0.020]:
            chosen = best_white if oof_scores[best_white] >= svc_oof + margin else "svc"
            row = fixed_metric(seed, f"gate_fusion_margin_{margin:.3f}", chosen, y_val, final_probs[chosen], oof_scores[chosen], svc_oof)
            outer_rows.append(row)

        best_direct = max((r for r in outer_rows if r["seed"] == seed and r["method"] in method_names), key=lambda r: r["accuracy"])
        gate_rows = [r for r in outer_rows if r["seed"] == seed and r["method"].startswith("gate")]
        best_gate = max(gate_rows, key=lambda r: r["accuracy"])
        print(
            f"  oof_svc={svc_oof:.4f} oof_best={best_name}:{oof_scores[best_name]:.4f} "
            f"direct_best={best_direct['method']}:{best_direct['accuracy']:.4f} "
            f"best_gate={best_gate['method']}->{best_gate['selected']}:{best_gate['accuracy']:.4f}",
            flush=True,
        )
        OUT.parent.mkdir(exist_ok=True)
        pd.DataFrame(outer_rows).to_csv(OUT, index=False)
        pd.DataFrame(detail_rows).to_csv(DETAIL, index=False)

    df = pd.DataFrame(outer_rows)
    df.to_csv(OUT, index=False)
    pd.DataFrame(detail_rows).to_csv(DETAIL, index=False)
    summary = (
        df.groupby("method")
        .agg(mean_acc=("accuracy", "mean"), min_acc=("accuracy", "min"), max_acc=("accuracy", "max"), mean_auc=("auc", "mean"))
        .sort_values(["min_acc", "mean_acc"], ascending=False)
    )
    print("\nSUMMARY")
    print(summary.head(80).to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nDETAIL")
    detail = pd.DataFrame(detail_rows)
    print(detail.pivot_table(index="seed", columns="method", values="oof_accuracy").round(4).to_string())


if __name__ == "__main__":
    run()
