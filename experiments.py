"""跨会话 (Leave-Session-Out) 实验台：对照基准 0.6535 扫描各改动。

不修改生产文件。通过子类化 EEGDataset 加入可配置的增强模式，通过训练循环加入
max-norm、可开关的类别权重等。每个配置用多个 seed 跑双向 LSO，报告均值与标准差，
从而在噪声（单折 ±2~3pt）下做出可靠判定。

用法：
    python experiments.py            # 跑预设 sweep
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_data import EEGDataset
from model import Net
from utils import get_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_ROOT / "train"
LABEL_CSV = DATA_ROOT / "train_labels.csv"


class ExpDataset(EEGDataset):
    """在 EEGDataset 基础上增加可配置的时移增强模式。"""

    def __init__(self, *args, shift_mode: str = "roll", **kwargs):
        self.shift_mode = shift_mode
        super().__init__(*args, **kwargs)

    def _augment_eeg(self, eeg: np.ndarray) -> np.ndarray:
        # 时移：roll(循环) / zeropad(零填充非循环) / none
        if self.max_time_shift > 0 and self.shift_mode != "none":
            shift = np.random.randint(-self.max_time_shift, self.max_time_shift + 1)
            if shift != 0:
                if self.shift_mode == "roll":
                    eeg = np.roll(eeg, shift=shift, axis=-1)
                elif self.shift_mode == "zeropad":
                    out = np.zeros_like(eeg)
                    if shift > 0:
                        out[..., shift:] = eeg[..., :-shift]
                    else:
                        out[..., :shift] = eeg[..., -shift:]
                    eeg = out

        if self.time_mask_ratio > 0 and np.random.rand() < 0.5:
            time_mask = max(1, int(eeg.shape[-1] * self.time_mask_ratio * np.random.uniform(0.5, 1.0)))
            start = np.random.randint(0, eeg.shape[-1] - time_mask + 1)
            eeg[..., start : start + time_mask] = 0.0

        if self.channel_mask_ratio > 0 and np.random.rand() < 0.3:
            channel_mask = max(1, int(eeg.shape[-2] * self.channel_mask_ratio))
            channels = np.random.choice(eeg.shape[-2], size=channel_mask, replace=False)
            eeg[:, channels, :] = 0.0

        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, size=eeg.shape).astype(np.float32)
            eeg = eeg + noise

        return eeg


def split_by_session(val_session: str) -> Tuple[List[int], List[int]]:
    df = pd.read_csv(LABEL_CSV)
    sessions = df["eeg_file"].astype(str).str.extract(r"(sess\d+)")[0]
    val = np.where(sessions.values == val_session)[0].tolist()
    tr = np.where(sessions.values != val_session)[0].tolist()
    return sorted(tr), sorted(val)


DEFAULTS = dict(
    shift_mode="roll",
    max_time_shift=10,
    noise_std=0.01,
    time_mask_ratio=0.06,
    channel_mask_ratio=0.08,
    weight_decay=1e-4,
    use_class_weight=True,
    label_smoothing=0.05,
    dropout=0.35,
    max_norm_spatial=None,   # float or None
    epochs=80,
    lr=1e-3,
    batch_size=64,
    grad_clip=1.0,
    lr_patience=5,
    early_stop=15,
)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total = correct = 0
    with torch.no_grad():
        for eeg, labels in loader:
            eeg = eeg.to(device); labels = labels.to(device)
            logits = model(eeg)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / total, correct / total


def train_one_fold(cfg: dict, val_session: str, seed: int, device) -> Tuple[float, int]:
    set_seed(seed)
    tr_idx, val_idx = split_by_session(val_session)

    train_ds = ExpDataset(
        data_dir=TRAIN_DIR, label_csv=LABEL_CSV, augment=True,
        shift_mode=cfg["shift_mode"], noise_std=cfg["noise_std"],
        max_time_shift=cfg["max_time_shift"], time_mask_ratio=cfg["time_mask_ratio"],
        channel_mask_ratio=cfg["channel_mask_ratio"], selected_indices=tr_idx,
    )
    val_ds = ExpDataset(
        data_dir=TRAIN_DIR, label_csv=LABEL_CSV, augment=False, selected_indices=val_idx,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")

    sample_eeg, _ = train_ds[0]
    model = Net(input_shape=tuple(sample_eeg.shape), dropout=cfg["dropout"]).to(device)

    if cfg["use_class_weight"]:
        ls = train_ds.samples["label"]
        bg = (ls == "background").sum(); tg = (ls == "target").sum()
        w = torch.tensor([len(ls) / (2.0 * bg), len(ls) / (2.0 * tg)],
                         dtype=torch.float32, device=device)
    else:
        w = None
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=cfg["lr_patience"])

    best_acc, best_loss, best_epoch, no_improve = -1.0, float("inf"), -1, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for eeg, labels in train_loader:
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
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss, best_epoch, no_improve = val_acc, val_loss, epoch, 0
        else:
            no_improve += 1
        if no_improve >= cfg["early_stop"]:
            break
    return best_acc, best_epoch


def run_config(name: str, overrides: dict, seeds: List[int], device) -> dict:
    cfg = {**DEFAULTS, **overrides}
    per_seed = []
    fold_accs = {"sess2": [], "sess1": []}
    fold_epochs = {"sess2": [], "sess1": []}
    for s in seeds:
        a2, e2 = train_one_fold(cfg, "sess2", s, device)   # train sess1, val sess2
        a1, e1 = train_one_fold(cfg, "sess1", s, device)   # train sess2, val sess1
        fold_accs["sess2"].append(a2); fold_accs["sess1"].append(a1)
        fold_epochs["sess2"].append(e2); fold_epochs["sess1"].append(e1)
        per_seed.append((a2 + a1) / 2.0)
    res = {
        "name": name,
        "overrides": overrides,
        "mean": float(np.mean(per_seed)),
        "std": float(np.std(per_seed)),
        "val_sess2_mean": float(np.mean(fold_accs["sess2"])),
        "val_sess1_mean": float(np.mean(fold_accs["sess1"])),
        "epoch_star": int(round(np.mean(fold_epochs["sess2"] + fold_epochs["sess1"]))),
        "per_seed": [round(x, 4) for x in per_seed],
    }
    print(f"[{name:22s}] mean={res['mean']:.4f} (±{res['std']:.4f}) "
          f"| sess2={res['val_sess2_mean']:.4f} sess1={res['val_sess1_mean']:.4f} "
          f"| E*={res['epoch_star']} | seeds={res['per_seed']}", flush=True)
    return res


def main():
    device = get_device()
    seeds = [42, 1, 7]
    print(f"Device={device} seeds={seeds}\n", flush=True)

    # 阶段1：先确认 zeropad 时移是否安全，再以它为新基准逐项扫描
    sweep = [
        ("roll_baseline", dict(shift_mode="roll")),
        ("zeropad_base", dict(shift_mode="zeropad")),
        ("zp_maxnorm1.0", dict(shift_mode="zeropad", max_norm_spatial=1.0)),
        ("zp_maxnorm0.5", dict(shift_mode="zeropad", max_norm_spatial=0.5)),
        ("zp_maxnorm2.0", dict(shift_mode="zeropad", max_norm_spatial=2.0)),
        ("zp_wd5e-4", dict(shift_mode="zeropad", weight_decay=5e-4)),
        ("zp_wd1e-3", dict(shift_mode="zeropad", weight_decay=1e-3)),
        ("zp_noise0.03", dict(shift_mode="zeropad", noise_std=0.03)),
        ("zp_noise0.05", dict(shift_mode="zeropad", noise_std=0.05)),
        ("zp_noise0.1", dict(shift_mode="zeropad", noise_std=0.1)),
        ("zp_noclassw", dict(shift_mode="zeropad", use_class_weight=False)),
        ("zp_dropout0.5", dict(shift_mode="zeropad", dropout=0.5)),
    ]

    results = []
    for name, ov in sweep:
        results.append(run_config(name, ov, seeds, device))

    results_sorted = sorted(results, key=lambda r: r["mean"], reverse=True)
    print("\n" + "=" * 70)
    print("阶段1 sweep 结果（按跨会话均值降序）")
    print("=" * 70)
    for r in results_sorted:
        print(f"  {r['name']:22s} mean={r['mean']:.4f} (±{r['std']:.4f})  E*={r['epoch_star']}")
    print("=" * 70)

    out = PROJECT_ROOT / "exp_stage1_results.json"
    out.write_text(json.dumps(results_sorted, ensure_ascii=False, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
