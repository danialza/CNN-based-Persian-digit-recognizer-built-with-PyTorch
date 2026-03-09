#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HODA .cdb files into ImageFolder train/test structure."
    )
    parser.add_argument(
        "--hoda-reader-path",
        type=str,
        required=True,
        help="Path to HodaDatasetReader.py",
    )
    parser.add_argument(
        "--digitdb-dir",
        type=str,
        required=True,
        help="Directory containing HODA .cdb files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset_hoda",
        help="Output dataset directory (ImageFolder format).",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="Train 60000.cdb",
        help="Training .cdb filename inside --digitdb-dir.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="Test 20000.cdb",
        help="Test .cdb filename inside --digitdb-dir.",
    )
    parser.add_argument(
        "--remaining-file",
        type=str,
        default="RemainingSamples.cdb",
        help="Remaining samples .cdb filename inside --digitdb-dir.",
    )
    parser.add_argument(
        "--include-remaining",
        action="store_true",
        help="Append RemainingSamples to train split if present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it already exists.",
    )
    return parser.parse_args()


def load_hoda_module(reader_path: Path) -> ModuleType:
    if not reader_path.exists():
        raise FileNotFoundError(f"Hoda reader not found: {reader_path}")

    spec = importlib.util.spec_from_file_location("hoda_dataset_reader", reader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {reader_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "read_hoda_cdb"):
        raise AttributeError(
            "Loaded module does not expose read_hoda_cdb(path). "
            f"Got: {reader_path}"
        )
    return module


def normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            # Convert RGB-like arrays to grayscale.
            arr = arr.mean(axis=-1)

    arr = np.nan_to_num(arr).astype(np.float32)
    if arr.size == 0:
        raise ValueError("Encountered empty image while converting HODA dataset.")

    if arr.max() <= 1.0 and arr.min() >= 0.0:
        arr *= 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def init_split_dir(split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for label in range(10):
        (split_dir / str(label)).mkdir(parents=True, exist_ok=True)


def write_split(
    images,
    labels,
    split_dir: Path,
    prefix: str,
    counters: dict[int, int],
) -> int:
    total = 0
    for image, label in zip(images, labels):
        label_int = int(label)
        if label_int < 0 or label_int > 9:
            raise ValueError(f"Unexpected label {label_int}; expected digits 0..9.")

        image_arr = normalize_image(image)
        out_path = split_dir / str(label_int) / f"{prefix}_{counters[label_int]:06d}.png"
        Image.fromarray(image_arr, mode="L").save(out_path)
        counters[label_int] += 1
        total += 1
        if total % 5000 == 0:
            print(f"  wrote {total} images into {split_dir.name}...")
    return total


def read_cdb(module: ModuleType, cdb_path: Path):
    if not cdb_path.exists():
        raise FileNotFoundError(f"CDB file not found: {cdb_path}")
    print(f"Reading: {cdb_path}")
    images, labels = module.read_hoda_cdb(str(cdb_path))
    if len(images) != len(labels):
        raise ValueError(
            f"Image/label length mismatch in {cdb_path}: "
            f"{len(images)} images vs {len(labels)} labels."
        )
    print(f"  loaded {len(images)} samples")
    return images, labels


def maybe_prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        has_files = any(output_dir.iterdir())
        if has_files and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}\n"
                "Use --overwrite to replace it."
            )
        if has_files and overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    reader_path = Path(args.hoda_reader_path)
    digitdb_dir = Path(args.digitdb_dir)
    output_dir = Path(args.output_dir)
    train_path = digitdb_dir / args.train_file
    test_path = digitdb_dir / args.test_file
    remaining_path = digitdb_dir / args.remaining_file

    module = load_hoda_module(reader_path)
    maybe_prepare_output(output_dir, overwrite=args.overwrite)

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    init_split_dir(train_dir)
    init_split_dir(test_dir)

    train_counters = {i: 0 for i in range(10)}
    test_counters = {i: 0 for i in range(10)}

    train_images, train_labels = read_cdb(module, train_path)
    train_total = write_split(train_images, train_labels, train_dir, "train", train_counters)

    if args.include_remaining:
        if remaining_path.exists():
            remaining_images, remaining_labels = read_cdb(module, remaining_path)
            added = write_split(
                remaining_images,
                remaining_labels,
                train_dir,
                "remaining",
                train_counters,
            )
            train_total += added
        else:
            print(f"Remaining file not found, skipping: {remaining_path}")

    test_images, test_labels = read_cdb(module, test_path)
    test_total = write_split(test_images, test_labels, test_dir, "test", test_counters)

    print("\nDone.")
    print(f"Train samples: {train_total}")
    print(f"Test samples: {test_total}")
    print(f"Output dataset: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
