from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    ops: list[transforms.Transform] = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        ops.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
            )
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transforms.Compose(ops)


def _validate_digit_classes(classes: list[str]) -> None:
    if len(classes) != 10:
        raise ValueError(
            "Expected 10 classes for digits, "
            f"but found {len(classes)} classes: {classes}"
        )


def get_train_val_loaders(
    dataset_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    root = Path(dataset_dir)
    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Missing training directory: {train_dir}. "
            "Expected dataset structure: <dataset>/train/<class>/images..."
        )

    train_aug_ds = datasets.ImageFolder(train_dir, transform=build_transform(image_size, train=True))
    _validate_digit_classes(train_aug_ds.classes)
    class_names = train_aug_ds.classes

    if val_dir.exists():
        val_ds = datasets.ImageFolder(val_dir, transform=build_transform(image_size, train=False))
        _validate_digit_classes(val_ds.classes)
        if val_ds.classes != class_names:
            raise ValueError(
                "Class mismatch between train and val splits. "
                f"train={class_names}, val={val_ds.classes}"
            )
        train_ds = train_aug_ds
    else:
        if not 0.0 < val_split < 1.0:
            raise ValueError(f"--val-split must be between 0 and 1, got {val_split}")
        train_eval_ds = datasets.ImageFolder(train_dir, transform=build_transform(image_size, train=False))
        total = len(train_aug_ds)
        val_count = max(1, int(total * val_split))
        if val_count >= total:
            raise ValueError(
                f"Not enough samples ({total}) to create a validation split with val_split={val_split}"
            )
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total, generator=generator).tolist()
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        train_ds = Subset(train_aug_ds, train_indices)
        val_ds = Subset(train_eval_ds, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, class_names


def get_split_loader(
    dataset_dir: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[Optional[DataLoader], Optional[list[str]]]:
    split_dir = Path(dataset_dir) / split
    if not split_dir.exists():
        return None, None
    ds = datasets.ImageFolder(split_dir, transform=build_transform(image_size, train=False))
    _validate_digit_classes(ds.classes)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, ds.classes
