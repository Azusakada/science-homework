"""在留出会话上（双向 LSO）收集 out-of-fold 概率，挑选诚实的全局决策阈值。

做法：训 sess1 预测 sess2、训 sess2 预测 sess1，每个训练样本都在"被留出"时拿到一次
target 类概率（含与推理一致的零填充时移 TTA）。汇总后扫阈值，取使整体准确率最高者。
阈值仅在留出会话上确定，绝不接触 test。结果写入 models/ensemble_meta.json。

用法： python tune_threshold.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments import ExpDataset, split_by_session
from model import Net
from utils import get_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
TRAIN_DIR = DATA_ROOT / "train"
LABEL_CSV = DATA_ROOT / "train_labels.csv"
CONFIG_PATH = PROJECT_ROOT / "best_config.json"

LABEL_TO_INDEX = {"background": 0, "target": 1}


def shift_pad(x: torch.Tensor, s: int) -> torch.Tensor:
    if s == 0:
        return x
    out = torch.zeros_like(x)
    if s > 0:
        out[..., s:] = x[..., :-s]
    else:
        out[..., :s] = x[..., -s:]
    return out


def tta_probs(model, eeg, shifts):
    return torch.stack([F.softmax(model(shift_pad(eeg, s)), dim=1) for s in shifts]).mean(0)


def train_fold(cfg, val_session, device):
    set_seed(42)
    tr_idx, _ = split_by_session(val_session)
    ds = ExpDataset(
        data_dir=TRAIN_DIR, label_csv=LABEL_CSV, augment=True,
        shift_mode=cfg["shift_mode"], noise_std=cfg["noise_std"],
        max_time_shift=cfg["max_time_shift"], time_mask_ratio=cfg["time_mask_ratio"],
        channel_mask_ratio=cfg["channel_mask_ratio"], selected_indices=tr_idx,
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda")
    sample_eeg, _ = ds[0]
    model = Net(input_shape=tuple(sample_eeg.shape), dropout=cfg["dropout"]).to(device)
    if cfg["use_class_weight"]:
        ls = ds.samples["label"]
        bg = (ls == "background").sum(); tg = (ls == "target").sum()
        w = torch.tensor([len(ls)/(2.0*bg), len(ls)/(2.0*tg)], dtype=torch.float32, device=device)
    else:
        w = None
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for _ in range(cfg["epochs_star"]):
        model.train()
        for eeg, labels in loader:
            eeg = eeg.to(device); labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(eeg), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            if cfg["max_norm_spatial"] is not None:
                with torch.no_grad():
                    sw = model.features[2].weight
                    torch.renorm(sw, p=2, dim=0, maxnorm=cfg["max_norm_spatial"], out=sw)
    return model


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    device = get_device()
    shifts = cfg.get("tta_shifts", [-8, -4, 0, 4, 8])

    all_probs, all_labels = [], []
    for val_session in ("sess2", "sess1"):
        model = train_fold(cfg, val_session, device)
        model.eval()
        _, val_idx = split_by_session(val_session)
        val_ds = ExpDataset(data_dir=TRAIN_DIR, label_csv=LABEL_CSV, augment=False,
                            selected_indices=val_idx)
        loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda")
        with torch.no_grad():
            for eeg, labels in loader:
                eeg = eeg.to(device)
                p = tta_probs(model, eeg, shifts)[:, 1].cpu().numpy()
                all_probs.append(p)
                # ExpDataset(has_label=True) 已把标签映射为整数张量(0/1)，直接用
                all_labels.append(labels.numpy())
        print(f"collected OOF for val={val_session}", flush=True)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    acc_05 = float(((probs >= 0.5).astype(int) == labels).mean())
    best_t, best_acc = 0.5, acc_05
    for t in np.arange(0.30, 0.701, 0.01):
        acc = float(((probs >= t).astype(int) == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)

    print("\n" + "=" * 60)
    print(f"OOF 样本数={len(labels)} | acc@0.5={acc_05:.4f} | "
          f"best_t={best_t:.2f} -> acc={best_acc:.4f}")
    print("=" * 60)
    # 阈值很靠近 0.5 才采用，避免过拟合留出会话；增益<0.5pt 则回退 0.5
    use_t = best_t if (best_acc - acc_05) >= 0.005 and abs(best_t - 0.5) <= 0.12 else 0.5
    print(f"采用阈值 = {use_t:.2f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = MODEL_DIR / "ensemble_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"threshold": use_t, "tta_shifts": shifts,
                 "oof_acc_at_0.5": acc_05, "oof_best_t": best_t, "oof_best_acc": best_acc})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"saved threshold meta -> {meta_path}")


if __name__ == "__main__":
    main()
