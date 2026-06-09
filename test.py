from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from load_data import EEGDataset, INDEX_TO_LABEL
from model import Net
from utils import get_device


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RES_DIR = PROJECT_ROOT / "res"

CONFIG = {
    "test_dir": DATA_ROOT / "test",
    "output_csv": RES_DIR / "predictions.csv",
    "batch_size": 64,
    "num_workers": 0,
    "use_cpu": False,
    "fallback_model": MODEL_DIR / "best_model.pth",
}


def shift_pad(x: torch.Tensor, s: int) -> torch.Tensor:
    """零填充非循环时移，与训练增强 / 阈值调优保持一致。"""
    if s == 0:
        return x
    out = torch.zeros_like(x)
    if s > 0:
        out[..., s:] = x[..., :-s]
    else:
        out[..., :s] = x[..., -s:]
    return out


def load_ensemble(device, input_shape):
    """优先加载多 seed 集成 + 阈值；缺失时回退到单个 best_model.pth。"""
    meta_path = MODEL_DIR / "ensemble_meta.json"
    threshold = 0.5
    shifts = [0]
    ckpts = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        threshold = float(meta.get("threshold", 0.5))
        shifts = meta.get("tta_shifts", [0]) or [0]
        ckpts = [MODEL_DIR / c for c in meta.get("checkpoints", [])]
        ckpts = [c for c in ckpts if c.exists()]
    if not ckpts:
        ckpts = [CONFIG["fallback_model"]]
        print("[warn] 未找到集成 ckpt，回退到 best_model.pth（无 TTA / 阈值 0.5）")
        shifts = [0]

    models = []
    for c in ckpts:
        m = Net(input_shape=input_shape).to(device)
        m.load_state_dict(torch.load(c, map_location=device))
        m.eval()
        models.append(m)
    print(f"加载 {len(models)} 个模型 | TTA shifts={shifts} | 阈值={threshold}")
    return models, shifts, threshold


def main() -> None:
    device = get_device(CONFIG["use_cpu"])
    test_dataset = EEGDataset(data_dir=CONFIG["test_dir"], label_csv=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    sample_eeg, _ = test_dataset[0]
    models, shifts, threshold = load_ensemble(device, tuple(sample_eeg.shape))

    predictions = []
    with torch.no_grad():
        for eeg, eeg_files in test_loader:
            eeg = eeg.to(device)
            # 跨模型 + 跨时移 平均 softmax
            probs = torch.zeros(eeg.size(0), 2, device=device)
            for m in models:
                for s in shifts:
                    probs += F.softmax(m(shift_pad(eeg, s)), dim=1)
            probs /= (len(models) * len(shifts))

            target_prob = probs[:, 1].cpu()
            for eeg_file, p in zip(eeg_files, target_prob.tolist()):
                pred_index = 1 if p >= threshold else 0
                predictions.append({"eeg_file": eeg_file, "prediction": INDEX_TO_LABEL[pred_index]})

    RES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(CONFIG["output_csv"], index=False)
    print(f"Predictions saved to: {CONFIG['output_csv']} ({len(predictions)} rows)")


if __name__ == "__main__":
    main()
