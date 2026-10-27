# 01. Problem and Data

## The task

For each knee MRI **study** (one patient scan session), predict the probability that each of 12 findings is present. Submissions are a CSV with `StudyInstanceUID` plus 12 probability columns, in the same column order as `sample_submission.csv`.

| Group | Targets |
|---|---|
| Ligaments | ACL, MCL |
| Menisci | Medial Meniscus, Lateral Meniscus |
| Osteoarthritis | Medial OA, Lateral OA, PF OA (patellofemoral) |
| Joint fluid / lining | Effusion, Synovitis, Baker's (cyst) |
| Bone | Contusion, Fracture |

## Files

| File | Rows x cols | What it holds |
|---|---|---|
| `train.csv` | 4,407 x 14 | `StudyInstanceUID`, `Report`, then 12 label columns (mostly NaN) |
| `train_series.csv` | 24,371 x 5 | One row per MRI series, including `SeriesInstanceUID` and `Anatomical_Plane` |
| `test.csv` | 3 x 1 | Study IDs (placeholder locally; the hidden set is swapped in on Kaggle) |
| `test_series.csv` | 15 x 5 | Series metadata for test studies |
| `sample_submission.csv` | 3 x 13 | Required output format |

DICOM images live under `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm` and `test_series/...`.

## The central problem: almost no labels

| Label status | Studies |
|---|---|
| All 12 expert labels present ("gold") | **58** |
| Partial labels | 0 |
| No labels (all NaN) | 4,349 |
| Has a text report | 4,407 |

Labels are all-or-nothing per study. 58 studies is far too few to train a deep network reliably, which is why the pipeline looks at the reports as a second source of supervision.

## Gold-set class balance

| Target | Positive | Negative | Positive % |
|---|---:|---:|---:|
| ACL | 24 | 34 | 41.4 |
| MCL | 9 | 49 | 15.5 |
| Medial Meniscus | 26 | 32 | 44.8 |
| Lateral Meniscus | 23 | 35 | 39.7 |
| Medial OA | 15 | 43 | 25.9 |
| Lateral OA | 11 | 47 | 19.0 |
| PF OA | 21 | 37 | 36.2 |
| Effusion | 35 | 23 | 60.3 |
| Synovitis | 27 | 31 | 46.6 |
| Baker's | 12 | 46 | 20.7 |
| Contusion | 19 | 39 | 32.8 |
| Fracture | 18 | 40 | 31.0 |

These rates are measured on 58 studies, so each carries a wide margin of error. They may not reflect the hidden test set.

## MRI series structure

- 3 to 14 series per study (median 5).
- Series are tagged by plane: **Sagittal** (9,864), **Coronal** (8,609), **Axial** (5,898).
- Different series within a plane usually use different MRI sequences (for example fat-suppressed versus not), which highlight different tissues. The baseline ignores this and takes the first series per plane.

## Integrity checks already done

- `train.csv` and `sample_submission.csv` have the same 12 target columns in the same order (asserted in code).
- Every training study appears in `train_series.csv`.
- No `StudyInstanceUID` appears in both train and test.
- Reports are not unique (4,276 distinct texts across 4,407 studies), so some are templated or duplicated. This matters for any text-based labeling.
