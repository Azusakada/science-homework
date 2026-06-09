from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


LABEL_TO_INDEX: Dict[str, int] = {"background": 0, "target": 1}
INDEX_TO_LABEL: Dict[int, str] = {value: key for key, value in LABEL_TO_INDEX.items()}


class EEGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        label_csv: Optional[str | Path] = None,
        normalize: bool = True,
        augment: bool = False,
        noise_std: float = 0.01,
        max_time_shift: int = 10,
        time_mask_ratio: float = 0.06,
        channel_mask_ratio: float = 0.08,
        selected_indices: Optional[List[int]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.augment = augment
        self.noise_std = noise_std
        self.max_time_shift = max_time_shift
        self.time_mask_ratio = time_mask_ratio
        self.channel_mask_ratio = channel_mask_ratio

        if label_csv is None:
            eeg_files = sorted(path.name for path in self.data_dir.glob("*.npy"))
            self.samples = pd.DataFrame({"eeg_file": eeg_files})
            self.has_label = False
        else:
            self.samples = pd.read_csv(label_csv)
            self.has_label = True

        if selected_indices is not None:
            self.samples = self.samples.iloc[selected_indices].reset_index(drop=True)

        if "eeg_file" not in self.samples.columns:
            raise ValueError("label csv must contain column: eeg_file")

        if self.has_label and "label" not in self.samples.columns:
            raise ValueError("training label csv must contain column: label")

    def _augment_eeg(self, eeg: np.ndarray) -> np.ndarray:
        # 时间平移：用零填充的非循环平移（而非 np.roll 循环平移）。
        # 循环平移会把 epoch 末尾接到开头，制造测试集不存在的非物理断点。
        if self.max_time_shift > 0:
            shift = np.random.randint(-self.max_time_shift, self.max_time_shift + 1)
            if shift != 0:
                shifted = np.zeros_like(eeg)
                if shift > 0:
                    shifted[..., shift:] = eeg[..., :-shift]
                else:
                    shifted[..., :shift] = eeg[..., -shift:]
                eeg = shifted

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples.iloc[index]
        eeg_path = self.data_dir / row["eeg_file"]
        eeg = np.load(eeg_path).astype(np.float32)

        # 保留全局逐样本标准化：实测逐通道标准化会掉 3.5~4.8pt（通道间幅度是判别信号）。
        if self.normalize:
            mean = eeg.mean()
            std = eeg.std()
            eeg = (eeg - mean) / (std + 1e-6)

        if self.augment:
            eeg = self._augment_eeg(eeg)

        eeg_tensor = torch.from_numpy(eeg)
        if self.has_label:
            label = torch.tensor(LABEL_TO_INDEX[row["label"]], dtype=torch.long)
            return eeg_tensor, label
        return eeg_tensor, row["eeg_file"]


def build_split_indices(
    label_csv: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """按标签分层的随机划分（同会话）。注意：测试集是留出会话 sess3，
    这种随机划分会高估泛化（同会话乐观）。仅作兼容保留，调参请用按会话划分。"""
    df = pd.read_csv(label_csv)
    rng = np.random.default_rng(seed)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for label_name in sorted(df["label"].unique()):
        label_indices = np.where(df["label"].values == label_name)[0]
        shuffled = rng.permutation(label_indices)
        val_size = max(1, int(len(shuffled) * val_ratio))
        val_indices.extend(shuffled[:val_size].tolist())
        train_indices.extend(shuffled[val_size:].tolist())

    return sorted(train_indices), sorted(val_indices)


def build_split_indices_by_session(
    label_csv: str | Path,
    val_session: str = "sess2",
) -> Tuple[List[int], List[int]]:
    """按会话留一划分：val_session 做验证，其余会话做训练。

    这与真实测试分布一致（测试集 = 同被试的留出会话 sess3），是诚实的跨会话
    泛化估计。会话标识从 eeg_file 文件名解析（形如 sub01_sess1_epoch002.npy）。
    """
    df = pd.read_csv(label_csv)
    sessions = df["eeg_file"].astype(str).str.extract(r"(sess\d+)")[0]
    if sessions.isna().any():
        raise ValueError("有 eeg_file 无法解析出 sessN，请检查标签文件")

    val_indices = np.where(sessions.values == val_session)[0].tolist()
    train_indices = np.where(sessions.values != val_session)[0].tolist()
    if not val_indices or not train_indices:
        raise ValueError(f"按会话划分结果为空：val_session={val_session}")
    return sorted(train_indices), sorted(val_indices)
