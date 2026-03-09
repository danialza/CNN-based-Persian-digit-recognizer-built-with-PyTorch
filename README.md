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

## Experiment report (June 2025)

### Local paths used in this run

- HODA reader repository: `/Users/danial/llm/ml/persian-digit-recognizer/HodaDatasetReader`
- Raw HODA `.cdb` files: `/Users/danial/llm/ml/persian-digit-recognizer/HodaDatasetReader/DigitDB`
- Converted ImageFolder dataset: `/Users/danial/llm/ml/persian-digit-recognizer/dataset_hoda`
- Training outputs: `/Users/danial/llm/ml/persian-digit-recognizer/runs/hoda-exp1`

### Training log

- `Epoch 01/10`: end of the first epoch out of 10
- `train_loss=0.1337`: training loss
- `train_acc=0.9568`: training accuracy (`95.68%`)
- `val_loss=0.0312`, `val_acc=0.9902`: validation performance (`99.02%`)

### Evaluation

- After each epoch, the model is evaluated on validation data.
- The best model is selected and saved based on `val_acc`.
- At the end, the best checkpoint is evaluated on the test split.
- Final outputs are saved in `metrics.json` and confusion matrix images.

### Model

- Custom CNN: `PersianDigitCNN`
- Architecture:
- `Conv(1->32) + BatchNorm + ReLU + MaxPool`
- `Conv(32->64) + BatchNorm + ReLU + MaxPool`
- `Conv(64->128) + BatchNorm + ReLU + MaxPool`
- `AdaptiveAvgPool(3x3)`
- `Flatten -> Linear(1152->256) -> ReLU -> Dropout(0.3) -> Linear(256->10)`

### Training setup

- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW`

### Final results (`runs/hoda-exp1`)

- `best_val_acc=0.9976`
- `test_loss=0.0169`
- `test_acc=0.9956` (`99.56%`)

### Confusion matrix outputs

- Validation confusion matrix: `runs/hoda-exp1/val_confusion_matrix.png`
- Test confusion matrix: `runs/hoda-exp1/test_confusion_matrix.png`

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
