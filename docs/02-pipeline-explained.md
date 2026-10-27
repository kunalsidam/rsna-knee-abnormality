# 02. Pipeline Explained

This document walks through the notebook's stages in order and shows where each lives in `src/`.

## Stage 1. Environment check

The first notebook cell confirms PyTorch and the GPU (the run used two Tesla T4s) and locates the Kaggle input folder. Nothing to port; it is a sanity check.

## Stage 2. Data audit

Loads the five CSVs, verifies the target columns, counts gold labels, prints class balance and checks series/ID integrity. The results are summarised in [01-problem-and-data.md](01-problem-and-data.md). Code: `data.load_metadata`, `data.gold_mask`.

## Stage 3. Dataset and DataLoader

### 3a. Smoke test with dummy images

The first dataset returned all-zero tensors, then random noise, just to prove that shapes (`[B, 3, 224, 224]` images, `[B, 12]` targets), the loss and the optimizer all worked. A 3-epoch loop ran on noise. This validates plumbing only; the loss values mean nothing.

### 3b. Real DICOM loading

`RealKneeDataset` loads a real image: take the first series listed for the study, sort its `.dcm` files, read the **middle** file with `pydicom`, divide by the maximum pixel value, resize to 224x224 with bilinear interpolation, and copy the single channel to three.

### 3c. Multi-plane upgrade

`MultiPlaneKneeDataset` is the more meaningful input. It finds one series per anatomical plane and builds a 3-channel tensor:

| Channel | Source |
|---|---|
| 0 | Sagittal middle slice |
| 1 | Coronal middle slice |
| 2 | Axial middle slice |

A missing plane or unreadable file becomes zeros. This gives the network a rough 3-view summary of the knee at no extra architectural cost. Code: `data.KneeDataset`.

> Note: the channels are different views, not different colors. ImageNet-pretrained filters expect RGB, so this is a pragmatic trick rather than a principled fit.

## Stage 4. Baseline model

An ImageNet-pretrained **ResNet18** whose final layer is replaced by `Linear(512, 12)`. Trained with `BCEWithLogitsLoss` (one independent sigmoid per target) and `AdamW(lr=1e-4)`. Code: `model.build_model`.

## Stage 5. Validation

An 80/20 random split of the 58 gold studies (46 train, 12 validation), `random_state=42`. Score = **macro ROC-AUC**: ROC-AUC per target, averaged.

A target is skipped when the validation fold contains only one class, because AUC is undefined there. With 12 validation studies this happens often, so the printed line reports how many of the 12 classes were scorable. Code: `train.macro_auc`.

## Stage 6. Heuristic labels

Parses every report into 12 soft probabilities. Described in full in [03-heuristic-labeling.md](03-heuristic-labeling.md). Code: `labels.py`.

## Stage 7. Full-data training and inference

The notebook merges the heuristic labels into the training table and prepares an inference cell that loads a checkpoint, predicts the test studies and writes `submission.csv` in the sample's column order.

The training run itself is not in the notebook, and the checkpoint the inference cell looks for (`resnet18_full_epoch1.pth`) is never saved. In the repo, `train.py` closes that gap: it trains, saves `best.pth`, and `predict.py` consumes it.

## How the repo code differs from the notebook

| Notebook | Repo | Why |
|---|---|---|
| Filters the series table inside `__getitem__` for every sample | Indexes series by study once in `__init__` | Much faster data loading |
| Bare `except:` returns zeros silently | Logs a warning, then returns zeros | A missing import once silently produced all-zero images (see limitations) |
| Several near-duplicate copies of the dataset class | One `KneeDataset` for train and test | Removes copy/paste drift |
| Heuristic negation checked over the whole report | Checked within the keyword's sentence | Fewer false negatives |
| No checkpoint saved before inference | `train.py` saves best checkpoint and metrics | Makes the pipeline runnable end to end |
| Validation only on a split of the gold set | Option to train on heuristic labels and validate on all 58 gold studies | Gives a larger, cleaner validation set |
