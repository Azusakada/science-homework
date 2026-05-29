from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_data import EEGDataset, build_split_indices
from model import Net
from utils import get_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

CONFIG = {
    "train_dir": DATA_ROOT / "train",
    "train_label_csv": DATA_ROOT / "train_labels.csv",
    "batch_size": 64,
    "epochs": 80,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "val_ratio": 0.2,
    "seed": 42,
    "num_workers": 0,
    "use_cpu": False,
    "augment_train": True,
    "label_smoothing": 0.05,
    "grad_clip_norm": 1.0,
    "lr_patience": 5,
    "early_stop_patience": 15,
}


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for eeg, labels in dataloader:
            eeg = eeg.to(device)
            labels = labels.to(device)
            logits = model(eeg)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / total, correct / total


def main() -> None:
    set_seed(CONFIG["seed"])
    device = get_device(CONFIG["use_cpu"])

    train_indices, val_indices = build_split_indices(
        CONFIG["train_label_csv"],
        val_ratio=CONFIG["val_ratio"],
        seed=CONFIG["seed"],
    )

    train_dataset = EEGDataset(
        data_dir=CONFIG["train_dir"],
        label_csv=CONFIG["train_label_csv"],
        augment=CONFIG["augment_train"],
        selected_indices=train_indices,
    )
    val_dataset = EEGDataset(
        data_dir=CONFIG["train_dir"],
        label_csv=CONFIG["train_label_csv"],
        selected_indices=val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=device.type == "cuda",
    )

    sample_eeg, _ = train_dataset[0]
    model = Net(input_shape=tuple(sample_eeg.shape)).to(device)

    label_series = train_dataset.samples["label"]
    background_count = (label_series == "background").sum()
    target_count = (label_series == "target").sum()
    class_weights = torch.tensor(
        [
            len(label_series) / (2.0 * background_count),
            len(label_series) / (2.0 * target_count),
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=CONFIG["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=CONFIG["lr_patience"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        running_loss = 0.0

        for eeg, labels in train_loader:
            eeg = eeg.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(eeg)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_dataset)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, MODEL_DIR / "best_model.pth")
            print(f"Best model saved. val_acc={best_val_acc:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= CONFIG["early_stop_patience"]:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_DIR / "final_model.pth")
    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
