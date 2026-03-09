# Persian Digit Recognizer (PyTorch)

This module trains a neural network to classify Persian digits (`0..9`) from images.
It is isolated from the website code so you can publish it cleanly in the same repository.

## Project layout

```text
ml/persian-digit-recognizer/
├── prepare_hoda.py
├── persian_digits/
│   ├── data.py
│   ├── engine.py
│   ├── model.py
│   └── utils.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── train.py
```

## Dataset format for training

Use this folder structure:

```text
dataset/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 9/
├── val/              # optional
│   ├── 0/
│   └── ...
└── test/             # optional but recommended
    ├── 0/
    └── ...
```

Notes:
- If `val/` is missing, `train.py` creates validation split from `train/` using `--val-split`.
- `test/` is optional during training; if present, test metrics are computed automatically.

## Convert HODA dataset (`.cdb`) with `HodaDatasetReader`

This project supports the dataset from:
- [amir-saniyan/HodaDatasetReader](https://github.com/amir-saniyan/HodaDatasetReader)

Expected files in your HODA DigitDB folder:
- `Train 60000.cdb`
- `Test 20000.cdb`
- `RemainingSamples.cdb` (optional)

Convert to ImageFolder format:

```bash
python prepare_hoda.py \
  --hoda-reader-path /absolute/path/to/HodaDatasetReader.py \
  --digitdb-dir /absolute/path/to/DigitDB \
  --output-dir dataset_hoda \
  --include-remaining
```

After conversion, use:
- `dataset_hoda/train/...` for training
- `dataset_hoda/test/...` for testing

## Setup

```bash
cd ml/persian-digit-recognizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py \
  --dataset-dir /absolute/path/to/dataset \
  --epochs 20 \
  --batch-size 64 \
  --output-dir runs/exp1
```

Artifacts in `runs/exp1`:
- `best_model.pt`
- `last_model.pt`
- `metrics.json`
- `val_confusion_matrix.png`
- `test_confusion_matrix.png` (if test split exists)

## Evaluate

```bash
python evaluate.py \
  --dataset-dir /absolute/path/to/dataset \
  --model-path runs/exp1/best_model.pt \
  --split test
```

## Predict one image

```bash
python predict.py \
  --model-path runs/exp1/best_model.pt \
  --image-path /absolute/path/to/image.png \
  --top-k 3
```

## GitHub publish flow

From repository root:

```bash
git add ml/persian-digit-recognizer
git commit -m "Add Persian digit recognizer training pipeline"
git push
```
