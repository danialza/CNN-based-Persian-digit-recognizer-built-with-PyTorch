#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from persian_digits.data import build_transform
from persian_digits.model import PersianDigitCNN
from persian_digits.utils import DIGIT_TO_PERSIAN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for one image.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override image size. Default uses image_size from checkpoint.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="How many predictions to show.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = args.image_size if args.image_size is not None else checkpoint.get("image_size", 32)

    model = PersianDigitCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transform(image_size=image_size, train=False)
    image = Image.open(image_path).convert("L")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_k = max(1, min(args.top_k, len(class_names)))
    confidence, indices = torch.topk(probabilities, k=top_k)

    print(f"Image: {image_path}")
    print("Predictions:")
    for rank, (score, idx) in enumerate(zip(confidence.tolist(), indices.tolist()), start=1):
        label = class_names[idx]
        persian = ""
        try:
            persian = DIGIT_TO_PERSIAN.get(int(label), "")
        except ValueError:
            pass
        label_display = f"{label} ({persian})" if persian else label
        print(f"{rank}. {label_display} -> {score:.4f}")


if __name__ == "__main__":
    main()
