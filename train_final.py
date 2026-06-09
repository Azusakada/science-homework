"""最终模型训练：用 best_config.json 的最优配置，在全量 sess1+sess2 数据上、
以固定 E* 个 epoch、多 seed 各训一个模型，保存为 models/model_seed{seed}.pth。

不使用验证集 / 早停（E* 已由 LSO 阶段确定）。这样既用满全部训练数据，又避免对
"同会话验证集"过拟合选点。

用法： python train_final.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments import ExpDataset          # 复用可配置增强
from model import Net
from utils import get_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
TRAIN_DIR = DATA_ROOT / "train"
LABEL_CSV = DATA_ROOT / "train_labels.csv"
CONFIG_PATH = PROJECT_ROOT / "best_config.json"


def _worker_init(worker_id: int) -> None:
    # 让每个 DataLoader worker 的 numpy 随机种子不同，保证增强多样性
    import numpy as _np
    _np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    device = get_device()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device={device}\nConfig={json.dumps(cfg, ensure_ascii=False)}\n", flush=True)

    n = len(pd.read_csv(LABEL_CSV))
    all_idx = list(range(n))

    saved = []
    for seed in cfg["seeds"]:
        set_seed(seed)
        ds = ExpDataset(
            data_dir=TRAIN_DIR, label_csv=LABEL_CSV, augment=True,
            shift_mode=cfg["shift_mode"], noise_std=cfg["noise_std"],
            max_time_shift=cfg["max_time_shift"], time_mask_ratio=cfg["time_mask_ratio"],
            channel_mask_ratio=cfg["channel_mask_ratio"], selected_indices=all_idx,
        )
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=6, worker_init_fn=_worker_init,
                            pin_memory=device.type == "cuda")

        sample_eeg, _ = ds[0]
        model = Net(input_shape=tuple(sample_eeg.shape), dropout=cfg["dropout"]).to(device)

        if cfg["use_class_weight"]:
            ls = ds.samples["label"]
            bg = (ls == "background").sum(); tg = (ls == "target").sum()
            w = torch.tensor([len(ls) / (2.0 * bg), len(ls) / (2.0 * tg)],
                             dtype=torch.float32, device=device)
        else:
            w = None
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["label_smoothing"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])

        for epoch in range(1, cfg["epochs_star"] + 1):
            model.train()
            running = 0.0
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
                running += loss.item() * labels.size(0)
            if epoch % 5 == 0 or epoch == cfg["epochs_star"]:
                print(f"  seed {seed} epoch {epoch:03d}/{cfg['epochs_star']} "
                      f"train_loss={running/len(ds):.4f}", flush=True)

        path = MODEL_DIR / f"model_seed{seed}.pth"
        torch.save(model.state_dict(), path)
        saved.append(path.name)
        print(f"[seed {seed}] saved -> {path.name}", flush=True)

    meta_path = MODEL_DIR / "ensemble_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"checkpoints": saved, "config": cfg})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(saved)} models. meta -> {meta_path}")


if __name__ == "__main__":
    main()
