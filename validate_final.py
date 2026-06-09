"""端到端诚实验证：用最终提交配方（5-seed 集成 + TTA）在双向留一会话上评估，
得到真实的跨会话预期准确率，并在【集成概率】上重新挑选决策阈值（修正单模型 OOF
调阈值与集成推理之间的 mismatch）。

对每个留出会话：在另一会话上训练 cfg['seeds'] 个模型(E*)，用集成+TTA 预测留出会话。
汇总双向 OOF 概率 → acc@0.5 与最佳阈值 → 报告诚实跨会话准确率。
该估计略保守（每折只用单会话 ~850 样本，少于最终提交用的全部 1694）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments import ExpDataset, split_by_session, TRAIN_DIR, LABEL_CSV
from model import Net
from utils import get_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "best_config.json"


def _wif(wid):
    np.random.seed((torch.initial_seed() + wid) % (2 ** 32))


def shift_pad(x, s):
    if s == 0:
        return x
    out = torch.zeros_like(x)
    if s > 0:
        out[..., s:] = x[..., :-s]
    else:
        out[..., :s] = x[..., -s:]
    return out


def train_model(cfg, tr_idx, seed, device):
    set_seed(seed)
    ds = ExpDataset(data_dir=TRAIN_DIR, label_csv=LABEL_CSV, augment=True,
                    shift_mode=cfg["shift_mode"], noise_std=cfg["noise_std"],
                    max_time_shift=cfg["max_time_shift"], time_mask_ratio=cfg["time_mask_ratio"],
                    channel_mask_ratio=cfg["channel_mask_ratio"], selected_indices=tr_idx)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=6,
                        worker_init_fn=_wif, pin_memory=device.type == "cuda")
    sample, _ = ds[0]
    model = Net(input_shape=tuple(sample.shape), dropout=cfg["dropout"]).to(device)
    ls = ds.samples["label"]
    bg = (ls == "background").sum(); tg = (ls == "target").sum()
    w = torch.tensor([len(ls)/(2.0*bg), len(ls)/(2.0*tg)], dtype=torch.float32, device=device) \
        if cfg["use_class_weight"] else None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for _ in range(cfg["epochs_star"]):
        model.train()
        for eeg, labels in loader:
            eeg = eeg.to(device); labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(eeg), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
    model.eval()
    return model


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    device = get_device()
    shifts = cfg["tta_shifts"]
    print(f"端到端验证：seeds={cfg['seeds']} E*={cfg['epochs_star']} TTA={shifts}\n", flush=True)

    all_probs, all_labels = [], []
    fold_acc05 = {}
    for val_session in ("sess2", "sess1"):
        tr_idx, val_idx = split_by_session(val_session)
        models = [train_model(cfg, tr_idx, s, device) for s in cfg["seeds"]]
        print(f"[{val_session}] 训练好 {len(models)} 个模型，开始集成+TTA 预测", flush=True)

        val_ds = ExpDataset(data_dir=TRAIN_DIR, label_csv=LABEL_CSV, augment=False,
                            selected_indices=val_idx)
        loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4,
                            pin_memory=device.type == "cuda")
        probs, labs = [], []
        with torch.no_grad():
            for eeg, labels in loader:
                eeg = eeg.to(device)
                p = torch.zeros(eeg.size(0), 2, device=device)
                for m in models:
                    for s in shifts:
                        p += F.softmax(m(shift_pad(eeg, s)), dim=1)
                p /= (len(models) * len(shifts))
                probs.append(p[:, 1].cpu().numpy()); labs.append(labels.numpy())
        probs = np.concatenate(probs); labs = np.concatenate(labs)
        fold_acc05[val_session] = float(((probs >= 0.5).astype(int) == labs).mean())
        all_probs.append(probs); all_labels.append(labs)
        print(f"[{val_session}] 集成 acc@0.5 = {fold_acc05[val_session]:.4f}", flush=True)

    probs = np.concatenate(all_probs); labs = np.concatenate(all_labels)
    acc05 = float(((probs >= 0.5).astype(int) == labs).mean())
    best_t, best_acc = 0.5, acc05
    for t in np.arange(0.40, 0.611, 0.01):
        acc = float(((probs >= t).astype(int) == labs).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)

    print("\n" + "=" * 64)
    print("最终配方（5-seed 集成 + TTA）端到端跨会话诚实评估")
    print("=" * 64)
    print(f"  sess2 集成 acc@0.5 = {fold_acc05['sess2']:.4f}")
    print(f"  sess1 集成 acc@0.5 = {fold_acc05['sess1']:.4f}")
    print(f"  双向均值 acc@0.5    = {(fold_acc05['sess2']+fold_acc05['sess1'])/2:.4f}")
    print(f"  集成OOF整体 acc@0.5 = {acc05:.4f}")
    print(f"  集成最佳阈值 t={best_t:.2f} -> acc={best_acc:.4f}  (Δ={best_acc-acc05:+.4f})")
    # 仅当增益>=0.5pt且阈值靠近0.5才采用
    use_t = best_t if (best_acc - acc05) >= 0.005 and abs(best_t - 0.5) <= 0.10 else 0.5
    print(f"  >>> 采用阈值 = {use_t:.2f}  (诚实预期测试准确率 ≈ {best_acc if use_t==best_t else acc05:.4f})")
    print("=" * 64)

    # 把在【集成概率】上重新校准的阈值写回 meta，供 test.py 使用
    meta_path = MODEL_DIR / "ensemble_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"threshold": use_t, "tta_shifts": shifts,
                 "ensemble_oof_acc_at_0.5": acc05,
                 "ensemble_oof_best_t": best_t, "ensemble_oof_best_acc": best_acc,
                 "honest_cross_session_acc": (best_acc if use_t == best_t else acc05)})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"已把集成校准阈值写回 {meta_path}")


if __name__ == "__main__":
    main()
