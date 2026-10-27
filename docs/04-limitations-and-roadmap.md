# 04. Limitations and Roadmap

## What has and has not been demonstrated

| Claim | Status |
|---|---|
| Data audit and integrity checks | Done, output recorded in the notebook |
| Dataset loads real DICOM and builds 3-plane tensors | Shape-checked (`[3, 224, 224]`) |
| Training loop and macro-AUC code run without error | Run on dummy data and tiny splits; no scores recorded |
| Heuristic labels generated for all 4,407 studies | Code written, output file not inspected in the notebook |
| Training on heuristic labels | **Not run.** The cell titled "Full-Dataset Training Script" only merges labels; inference refers to a checkpoint that was never saved |
| Any validation or leaderboard score | **None recorded** |

## Issues found in the original notebook

1. **Silent all-zero images.** In the first train/validation cell, `pydicom` is never imported. The dataset wraps loading in a bare `except:` that returns zeros, so the `NameError` was swallowed and every image was blank. Any AUC from that cell reflects noise and class priors, not imaging. The later cell fixes the import, and the repo code logs failures instead of hiding them.
2. **Tiny validation set.** 12 studies. Several targets have one class only and are skipped, so "macro AUC over N of 12 classes" is high-variance and not comparable across runs.
3. **Dummy-image runs.** The 3-epoch and early smoke tests train on zeros or noise. Their losses are only a plumbing check.
4. **One slice per plane.** The middle file in a sorted folder is not guaranteed to be the anatomical center (sorting by filename, not `InstanceNumber` or slice position). It also discards most of the volume.
5. **First series per plane.** Studies can contain several sequences per plane (for example T1, PD, fat-suppressed). Picking the first is arbitrary.
6. **Preprocessing.** Dividing by each image's max is sensitive to bright outliers, and pretrained ResNet weights expect ImageNet mean/std normalization that is not applied.
7. **No augmentation, no regularization schedule, no class weighting.**
8. **Local test set has 3 rows.** Inference can only be verified for format locally.

## External labels (note)

The notebook loaded `raptor_sub.csv` from another Kaggle user's public notebook, apparently to use its predictions as soft labels. That is not included in this repo. If you pursue it, confirm that competition rules allow it, and treat those labels as another noisy teacher rather than ground truth.

## Roadmap (suggested order)

1. **Validate the heuristic labeler** on the 58 gold studies; report per-target AUC and fix the weakest rules.
2. **Run the real experiment:** `train.py --label-source heuristic`, and record `metrics.json` for gold-only versus heuristic training.
3. **Fix preprocessing:** sort slices by DICOM `InstanceNumber` / `ImagePositionPatient`, apply percentile clipping, and use ImageNet normalization.
4. **Use more of the volume:** average predictions over several slices per series, or encode a stack of slices (2.5D) or a 3D CNN.
5. **Use all series:** one branch per plane or per sequence with late fusion, instead of picking the first.
6. **Better validation:** k-fold cross-validation on the gold set, with out-of-fold predictions saved, to reduce variance.
7. **Loss design:** weight gold labels above heuristic ones; add class weights for rare targets (MCL, Lateral OA).
8. **Augmentation:** flips, small rotations, intensity jitter.
9. **Stronger backbone** once the data pipeline is trusted (ResNet50, EfficientNet, a medical-pretrained model).
10. **Ensembling** across folds and seeds.

## Safety note

This is a research and competition exercise. Models trained on a small, noisily labeled dataset must not be used for clinical decisions.
