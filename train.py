#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from persian_digits.data import get_split_loader, get_train_val_loaders
from persian_digits.engine import (
    build_confusion_matrix,
    evaluate,
    save_confusion_matrix,
    train_one_epoch,
)
from persian_digits.model import PersianDigitCNN
from persian_digits.utils import labels_with_persian, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Persian digit recognizer.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Dataset root path.")
    parser.add_argument("--output-dir", type=str, default="runs/exp", help="Training output path.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split if /val is missing.")
    parser.add_argument("--image-size", type=int, default=32, help="Image size after resizing.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def make_checkpoint(
    model: nn.Module,
    class_names: list[str],
    image_size: int,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": image_size,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, class_names = get_train_val_loaders(
        dataset_dir=args.dataset_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    model = PersianDigitCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    history: list[dict] = []
    best_model_path = output_dir / "best_model.pt"
    last_model_path = output_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_acc": train_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_acc": val_metrics.accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}"
        )

        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            torch.save(make_checkpoint(model, class_names, args.image_size), best_model_path)

    torch.save(make_checkpoint(model, class_names, args.image_size), last_model_path)

    val_confusion = build_confusion_matrix(
        num_classes=len(class_names),
        targets=val_metrics.targets,
        predictions=val_metrics.predictions,
    )
    save_confusion_matrix(
        matrix=val_confusion,
        class_names=labels_with_persian(class_names),
        title="Validation Confusion Matrix",
        output_path=output_dir / "val_confusion_matrix.png",
    )

    test_loader, test_classes = get_split_loader(
        dataset_dir=args.dataset_dir,
        split="test",
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    final_report = {
        "best_val_acc": best_val_acc,
        "class_names": class_names,
        "history": history,
        "device": str(device),
    }

    if test_loader is not None and test_classes is not None:
        if test_classes != class_names:
            raise ValueError(f"Class mismatch between train and test splits: {class_names} vs {test_classes}")
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, criterion, device)
        final_report["test_loss"] = test_metrics.loss
        final_report["test_acc"] = test_metrics.accuracy

        test_confusion = build_confusion_matrix(
            num_classes=len(class_names),
            targets=test_metrics.targets,
            predictions=test_metrics.predictions,
        )
        save_confusion_matrix(
            matrix=test_confusion,
            class_names=labels_with_persian(class_names),
            title="Test Confusion Matrix",
            output_path=output_dir / "test_confusion_matrix.png",
        )
        print(f"Test: loss={test_metrics.loss:.4f} acc={test_metrics.accuracy:.4f}")

    save_json(final_report, output_dir / "metrics.json")
    save_json({"class_names": class_names}, output_dir / "class_names.json")
    print(f"Saved artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
