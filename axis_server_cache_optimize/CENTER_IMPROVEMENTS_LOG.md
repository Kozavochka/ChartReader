# Line Center Improvements Log

## Purpose

This file tracks the changes made to improve `KPDetection` center quality for
line-only training, while preserving keypoint quality.

## Baseline Before Changes

- Evaluation dataset: `line_data/val`
- Checkpoint: `axis_server_cache_optimize/nnet/KPDetection/KPDetection_best.pkl`
- Evaluation script: `evaluation/evaluate_kd.py`

Metrics before code changes:

- `mAP keypoints = 0.36276023054429`
- `mAP centers = 0.010738907136639544`

## Implemented Changes

### 1. Generator changes

Repository:
`/mnt/c/Users/Пользователь/Desktop/IT/PetProjects/Python/datasetgenerator`

- Enforced odd point counts for generated line polylines.
- Added validation in `mixed_dataset.py` so line annotations with even point
  counts are reported as invalid.

Goal:

- Make the semantic center align with a real middle line point.

### 2. Training target changes

Repository:
`/home/kozavochka/projects/ChartReader`

- Added `line_center_mode = "nearest_real"` to line target building.
- Added `center_gaussian_radius = 6` for softer center supervision.
- Kept keypoint target generation unchanged.

Goal:

- Make center supervision more observable and less brittle.

### 3. Decode changes

Repository:
`/home/kozavochka/projects/ChartReader`

- Split keypoint and center decode parameters.
- Added separate center decode settings:
  - `center_top_k = 50`
  - `key_nms_kernel = 1`
  - `center_nms_kernel = 3`

Goal:

- Reduce false positive centers without affecting keypoint recall.

## Sanity Checks Completed

- Python compilation succeeded for modified generator files.
- Python compilation succeeded for modified ChartReader files.
- Generator sanity test produced odd line point counts.
- `sample_data(...)` for `KPDetection` completed successfully on `line_data`.
- `val_extraction.py` ran successfully on the existing `KPDetection_best.pkl`.

## Intermediate Result On Existing Checkpoint

Using the same checkpoint as the baseline, but with the updated decode path:

- `mAP keypoints = 0.36276023054429`
- `mAP centers = 0.03682764639479401`

Interpretation:

- Keypoint quality stayed unchanged.
- Center quality improved at inference time, but the main expected gain should
  come after dataset regeneration and retraining.

## Required Next Step

After regenerating the synthetic dataset and retraining `KPDetection`:

1. Re-run `val_extraction.py`.
2. Re-run `evaluation/evaluate_kd.py` on the same `line_data/val`.
3. Compare the new metrics against the baseline recorded above.

## Commit References

Record the final commit hashes and optional tags here after both commits are
created.

- ChartReader commit:
- ChartReader tag:
- Dataset generator commit:
- Dataset generator tag:
