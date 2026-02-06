#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from db.datasets import datasets
from config import system_configs


def load_db(config_path, data_dir, cache_dir, split, snapshot_name="PretrainKP"):
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    configs["system"]["data_dir"] = data_dir
    configs["system"]["cache_dir"] = cache_dir
    configs["system"]["dataset"] = "Chart"
    if snapshot_name:
        configs["system"]["snapshot_name"] = snapshot_name

    system_configs.update_config(configs["system"])
    return datasets["Chart"](configs["db"], split)


def get_pie_center(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ca = c - a
    cb = c - b
    cosine_angle = np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb))
    angle = np.arccos(cosine_angle)
    r_square = (ca**2).sum()

    if ca[0] * cb[1] - ca[1] * cb[0] >= 0:
        return (a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, 0.5 * angle * r_square
    return 2 * c[0] - (a[0] + b[0] + c[0]) / 3.0, 2 * c[1] - (a[1] + b[1] + c[1]) / 3.0, np.pi * r_square - 0.5 * angle * r_square


def get_points(gts, preds, chart_type, pred_key):
    gt_keys, gt_cens = [], []

    if chart_type == "vbar_categorical":
        for bbox in gts.tolist():
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            gt_keys.append((bbox[0], bbox[1], area))
            gt_keys.append((bbox[2], bbox[3], area))
            gt_cens.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, area))
    elif chart_type == "pie":
        for bbox in gts.tolist():
            a, b, c = (bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5])
            xce, yce, area = get_pie_center(a, b, c)
            gt_keys.append((bbox[0], bbox[1], area))
            gt_keys.append((bbox[2], bbox[3], area))
            gt_keys.append((bbox[4], bbox[5], area))
            gt_cens.append((xce, yce, area))
    elif chart_type == "line":
        for bbox in gts[0]:
            detection = np.array(bbox)
            if len(detection) <= 1:
                continue
            # Strip trailing category if present (odd length).
            if len(detection) % 2 == 1:
                detection = detection[:-1]
            if len(detection) <= 1:
                continue
            if (len(detection) // 2) % 2 == 0:
                mid = len(detection) // 2
                xce, yce = (detection[mid - 2] + detection[mid]) / 2, (detection[mid - 1] + detection[mid + 1]) / 2
            else:
                mid = len(detection) // 2
                xce, yce = detection[mid - 1].copy(), detection[mid].copy()
            xs = detection[0:len(detection):2]
            ys = detection[1:len(detection):2]
            area = (max(max(xs) - min(xs), max(ys) - min(ys)) / len(detection) * 2) ** 2

            for x, y in zip(xs, ys):
                gt_keys.append((x, y, area))
            gt_cens.append((xce, yce, area))

    pred_keys, pred_cens = [], []
    pred_groups = []
    is_kp_format = isinstance(preds, list) and preds and isinstance(preds[0], dict) and pred_key in preds[0]

    if not is_kp_format:
        # Baseline predictions (legacy formats).
        if chart_type == "pie":
            pred_keys.append((preds[0][0][0], preds[0][0][1], preds[0][-1]))
            for pred in preds:
                pred_keys.append((pred[1][0], pred[1][1], pred[-1]))
        elif chart_type == "line":
            for pred in preds:
                pred_groups.append(np.array(pred))
        else:
            for pred in preds:
                pred_keys.append((pred[0], pred[1], 1.0))
                pred_keys.append((pred[2], pred[3], 1.0))
    else:
        for point in preds[0][pred_key]:
            pred_keys.append((point[2], point[3], point[0]))
        for point in preds[1][pred_key]:
            pred_cens.append((point[2], point[3], point[0]))

    return gt_keys, gt_cens, pred_keys, pred_cens


def OKS(gt_p, pred_p):
    d2 = (gt_p[0] - pred_p[0]) ** 2 + (gt_p[1] - pred_p[1]) ** 2
    k2 = 0.1
    s2 = gt_p[2]
    return np.exp(d2 / (s2 * k2) * (-1))


def computeTargetLabel(gt_ps, pred_ps, thres=0.75):
    y_true = []
    for pred_p in pred_ps:
        found = False
        for gt_p in gt_ps:
            if OKS(gt_p, pred_p) > thres:
                y_true.append(1)
                found = True
                break
        if not found:
            y_true.append(0)
    return y_true


def computeDetectedGT(gt_ps, pred_ps, thres=0.75):
    y_true = []
    for gt_p in gt_ps:
        found = False
        for pred_p in pred_ps:
            if OKS(gt_p, pred_p) > thres:
                y_true.append(1)
                found = True
                break
        if not found:
            y_true.append(0)
    return y_true


def main():
    parser = argparse.ArgumentParser(description="Evaluate KPDetection results (ported from evaluation_kd.ipynb)")
    parser.add_argument("--config", default="config/KPDetection.json", help="Path to KPDetection config JSON")
    parser.add_argument("--data_dir", required=True, help="Dataset root (COCO format)")
    parser.add_argument("--cache_dir", default="data/cache/", help="Cache directory")
    parser.add_argument("--split", default="valchart", help="Dataset split (trainchart/valchart/testchart)")
    parser.add_argument("--prediction_json", required=True, help="Path to KPDetection prediction JSON")
    parser.add_argument("--chart_type", default="line", choices=["line", "vbar_categorical", "pie"])
    parser.add_argument("--pred_key", default="0", help="Category key in prediction JSON (default: 0)")
    parser.add_argument("--max_iter", type=int, default=-1, help="Limit number of images (default: all)")
    parser.add_argument("--snapshot_name", default="PretrainKP", help="Snapshot name for system_configs")
    args = parser.parse_args()

    db = load_db(args.config, args.data_dir, args.cache_dir, args.split, args.snapshot_name)

    with open(args.prediction_json, "r", encoding="utf-8") as f:
        prediction = json.load(f)

    mAP_keys = []
    mAP_cens = []

    max_iter = db.db_inds.size if args.max_iter < 0 else min(args.max_iter, db.db_inds.size)
    for i in tqdm(range(max_iter)):
        db_ind = db.db_inds[i]
        image_file = db.image_file(db_ind)
        gts = db.detections(db_ind)
        if gts is None or len(gts) == 0:
            continue

        image_name = os.path.basename(image_file)
        preds = prediction.get(image_name)
        if preds is None or len(preds) == 0:
            continue
        if isinstance(preds, list) and len(preds) == 3 and len(preds[2]) == 0:
            continue

        gt_keys, gt_cens, pred_keys, pred_cens = get_points(gts, preds, args.chart_type, args.pred_key)

        if pred_keys:
            y_true_keys = computeTargetLabel(gt_keys, pred_keys)
            y_score_keys = [key[2] for key in pred_keys]
            detected_gt_keys = computeDetectedGT(gt_keys, pred_keys)
            miss_count = len(detected_gt_keys) - sum(detected_gt_keys)
            y_true_keys = y_true_keys + [1] * miss_count
            y_score_keys = y_score_keys + [0] * miss_count
            mAP_keys = np.append(mAP_keys, average_precision_score(y_true_keys, y_score_keys))

        if pred_cens:
            y_true_cens = computeTargetLabel(gt_cens, pred_cens)
            y_score_cens = [key[2] for key in pred_cens]
            detected_gt_cens = computeDetectedGT(gt_cens, pred_cens)
            miss_count = len(detected_gt_cens) - sum(detected_gt_cens)
            y_true_cens = y_true_cens + [1] * miss_count
            y_score_cens = y_score_cens + [0] * miss_count
            mAP_cens = np.append(mAP_cens, average_precision_score(y_true_cens, y_score_cens))

    mAP_keys = np.array(mAP_keys)
    mAP_cens = np.array(mAP_cens)
    key_mean = mAP_keys[~np.isnan(mAP_keys)].mean() if mAP_keys.size else float("nan")
    cen_mean = mAP_cens[~np.isnan(mAP_cens)].mean() if mAP_cens.size else float("nan")
    print("mAP for keypoints:", key_mean, " mAP for center points:", cen_mean)


if __name__ == "__main__":
    main()
