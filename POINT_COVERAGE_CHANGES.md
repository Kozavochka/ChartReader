# KPDetection Point-Coverage Changes (2026-02-06)

## Purpose
Primary objective: improve keypoint detection coverage along all line series (fewer empty segments), while reducing negative impact from center supervision during KPDetection training.

## What Was Changed (ChartReader)

### 1) Decoding capacity increased
- File: `models/py_utils/kp_utils.py`
- Change:
  - `_decode_detection(..., K=300, ...)`
  - `_decode_group(..., K=300, ...)`
- Why:
  - Fixed low `K` cuts off many valid line points in dense scenes.
  - Higher `K` preserves more candidates before post-filtering.

### 2) Inference cap increased
- File: `config/KPDetection.json`
- Change:
  - `top_k: 300`
  - `max_per_image: 300`
- Why:
  - Prevent aggressive truncation of point candidates per image.
  - Align config with higher decode capacity.

### 3) Detection loss rebalanced toward keypoints
- Files:
  - `models/py_utils/kp.py` (`DetectionLoss`)
  - `models/KPDetection.py` (loss init)
- Change:
  - Added branch weights in `DetectionLoss`:
    - `key_heat_weight`, `center_heat_weight`
    - `key_regr_weight`, `center_regr_weight`
  - For KPDetection:
    - `key_heat_weight=1.0`
    - `center_heat_weight=0.35`
    - `key_regr_weight=1.0`
    - `center_regr_weight=0.35`
- Why:
  - Center task is noisier/harder and can pull shared features away from keypoint quality.
  - Rebalancing keeps center supervision but prioritizes the main goal: point coverage/quality.

## Expected Effect
- More predicted point candidates on long/multi-line charts.
- Better line coverage by keypoints after thresholding/selection.
- Lower risk of keypoint degradation from center branch.

## To Verify Next (Required)
1. Regenerate train/val on the updated generator settings (see datasetgenerator doc).
2. Train KPDetection from scratch with current config.
3. Evaluate on a fixed validation set and compare:
   - `mAP keypoints`
   - line-point coverage (recall of GT line points within distance threshold)
   - qualitative overlays (no large empty line segments).
4. If FP grows too much, tune score threshold and/or reduce `K`/`max_per_image` moderately.
