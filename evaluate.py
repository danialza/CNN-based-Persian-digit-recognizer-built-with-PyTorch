#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from persian_digits.data import get_split_loader
from persian_digits.engine import build_confusion_matrix, evaluate, save_confusion_matrix
from persian_digits.model import PersianDigitCNN
from persian_digits.utils import labels_with_persian, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Persian digit model.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Dataset root path.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to best_model.pt.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Evaluation split.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override image size; default uses image_size from checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save metrics and confusion matrix. Default: model directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = args.image_size if args.image_size is not None else checkpoint.get("image_size", 32)

    loader, split_classes = get_split_loader(
        dataset_dir=args.dataset_dir,
        split=args.split,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if loader is None or split_classes is None:
        raise FileNotFoundError(
            f"Split '{args.split}' not found in dataset. "
            f"Expected folder: {Path(args.dataset_dir) / args.split}"
        )
    if split_classes != class_names:
        raise ValueError(f"Class mismatch between model and dataset split: {class_names} vs {split_classes}")

    model = PersianDigitCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, device)
    print(f"{args.split}: loss={metrics.loss:.4f} acc={metrics.accuracy:.4f}")

    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    confusion = build_confusion_matrix(
        num_classes=len(class_names),
        targets=metrics.targets,
        predictions=metrics.predictions,
    )
    save_confusion_matrix(
        matrix=confusion,
        class_names=labels_with_persian(class_names),
        title=f"{args.split.title()} Confusion Matrix",
        output_path=output_dir / f"{args.split}_confusion_matrix.png",
    )
    save_json(
        {
            "split": args.split,
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
            "samples": len(metrics.targets),
        },
        output_dir / f"{args.split}_metrics.json",
    )
    print(f"Saved evaluation artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
