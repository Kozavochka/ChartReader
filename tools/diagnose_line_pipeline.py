#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sampling_function import _clip_detections, _full_image_crop, _resize_image


def _load_coco(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p)) if arr.size else float("nan")


def _mean(arr: np.ndarray) -> float:
    return float(arr.mean()) if arr.size else float("nan")


def _std(arr: np.ndarray) -> float:
    return float(arr.std()) if arr.size else float("nan")


def _pad_bboxes(bboxes: List[List[float]], pad_value: float = -1.0) -> np.ndarray:
    max_len = max((len(b) for b in bboxes), default=0)
    if max_len == 0:
        return np.zeros((0, 0), dtype=np.float32)
    out = []
    for b in bboxes:
        arr = list(b)
        if len(arr) < max_len:
            arr.extend([pad_value] * (max_len - len(arr)))
        out.append(arr)
    return np.array(out, dtype=np.float32)


def _valid_points_from_row(row: List[float]) -> List[Tuple[float, float]]:
    pts = []
    max_idx = len(row) - (len(row) % 2)
    for i in range(0, max_idx, 2):
        x = float(row[i])
        y = float(row[i + 1])
        if x <= 0 or y <= 0:
            continue
        pts.append((x, y))
    return pts


def _image_keypoints_and_lines(coco: dict) -> Tuple[Dict[int, List[List[float]]], Dict[int, dict]]:
    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    lines_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("category_id") != 2:
            continue
        lines_by_image[ann["image_id"]].append(ann.get("bbox", []))
    return lines_by_image, images_by_id


def dataset_stats(coco: dict) -> dict:
    lines_by_image, images_by_id = _image_keypoints_and_lines(coco)
    lines_per_image = []
    points_per_image = []
    points_per_line = []

    for image_id in images_by_id:
        lines = lines_by_image.get(image_id, [])
        lines_per_image.append(len(lines))
        total_points = 0
        for bbox in lines:
            n = len(bbox) // 2
            points_per_line.append(n)
            total_points += n
        points_per_image.append(total_points)

    lpi = np.array(lines_per_image, dtype=np.float32)
    ppi = np.array(points_per_image, dtype=np.float32)
    ppl = np.array(points_per_line, dtype=np.float32)

    return {
        "num_images": int(len(images_by_id)),
        "num_line_annotations": int(len(points_per_line)),
        "lines_per_image_mean": _mean(lpi),
        "lines_per_image_p90": _percentile(lpi, 90),
        "lines_per_image_max": int(lpi.max()) if lpi.size else 0,
        "points_per_line_mean": _mean(ppl),
        "points_per_line_p50": _percentile(ppl, 50),
        "points_per_line_p90": _percentile(ppl, 90),
        "points_per_line_max": int(ppl.max()) if ppl.size else 0,
        "points_per_image_mean": _mean(ppi),
        "points_per_image_p50": _percentile(ppi, 50),
        "points_per_image_p90": _percentile(ppi, 90),
        "points_per_image_max": int(ppi.max()) if ppi.size else 0,
        "images_points_gt_100": int((ppi > 100).sum()),
        "images_points_gt_200": int((ppi > 200).sum()),
    }


def preprocess_spacing_stats(
    coco: dict,
    data_dir: str,
    split: str,
    input_size: Tuple[int, int],
    output_size: Tuple[int, int],
) -> dict:
    lines_by_image, images_by_id = _image_keypoints_and_lines(coco)
    wr = output_size[1] / input_size[1]
    hr = output_size[0] / input_size[0]

    gap_mean_per_line = []
    gap_max_per_line = []
    unique_ratio_per_image = []
    points_per_image = []
    unique_cells_per_image = []
    used_images = 0

    for image_id, image_info in images_by_id.items():
        lines = lines_by_image.get(image_id, [])
        if not lines:
            continue
        image_path = os.path.join(data_dir, "images", split, image_info["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue

        det = _pad_bboxes(lines)
        if det.size == 0:
            continue

        image_used = False
        cropped, det = _full_image_crop(image, det)
        resized, det = _resize_image(cropped, det, list(input_size))
        det = _clip_detections(resized, det)

        uniq_cells = set()
        total_points_img = 0
        for row in det.tolist():
            pts = _valid_points_from_row(row)
            out_pts = []
            for x, y in pts:
                fx = x * wr
                fy = y * hr
                xi = int(fx)
                yi = int(fy)
                if xi <= 0 or yi <= 0 or xi >= output_size[1] - 1e-2 or yi >= output_size[0] - 1e-2:
                    continue
                out_pts.append((fx, fy, xi, yi))
                uniq_cells.add((xi, yi))

            if len(out_pts) >= 2:
                image_used = True
                seg = []
                for i in range(len(out_pts) - 1):
                    dx = out_pts[i + 1][0] - out_pts[i][0]
                    dy = out_pts[i + 1][1] - out_pts[i][1]
                    seg.append((dx * dx + dy * dy) ** 0.5)
                seg_arr = np.array(seg, dtype=np.float32)
                gap_mean_per_line.append(float(seg_arr.mean()))
                gap_max_per_line.append(float(seg_arr.max()))
                total_points_img += len(out_pts)

        if image_used and total_points_img > 0:
            used_images += 1
            points_per_image.append(total_points_img)
            unique_cells_per_image.append(len(uniq_cells))
            unique_ratio_per_image.append(len(uniq_cells) / total_points_img)

    gmean = np.array(gap_mean_per_line, dtype=np.float32)
    gmax = np.array(gap_max_per_line, dtype=np.float32)
    ppi = np.array(points_per_image, dtype=np.float32)
    uci = np.array(unique_cells_per_image, dtype=np.float32)
    uri = np.array(unique_ratio_per_image, dtype=np.float32)

    return {
        "used_images": int(used_images),
        "output_points_per_image_mean": _mean(ppi),
        "output_points_per_image_p90": _percentile(ppi, 90),
        "output_points_per_image_max": float(ppi.max()) if ppi.size else float("nan"),
        "unique_output_cells_per_image_mean": _mean(uci),
        "unique_output_cells_per_image_p90": _percentile(uci, 90),
        "unique_over_total_ratio_mean": _mean(uri),
        "unique_over_total_ratio_p10": _percentile(uri, 10),
        "line_seg_gap_output_px_mean": _mean(gmean),
        "line_seg_gap_output_px_p90": _percentile(gmean, 90),
        "line_max_gap_output_px_mean": _mean(gmax),
        "line_max_gap_output_px_p90": _percentile(gmax, 90),
        "line_max_gap_output_px_max": float(gmax.max()) if gmax.size else float("nan"),
    }


def prediction_stats(
    coco: dict,
    prediction_json: str,
    score_threshold: float,
    distance_threshold: float,
) -> dict:
    with open(prediction_json, "r", encoding="utf-8") as f:
        pred = json.load(f)

    lines_by_image, images_by_id = _image_keypoints_and_lines(coco)
    key_scores = []
    center_scores = []
    line_recalls = []
    line_hit_count = 0
    line_total = 0
    missing_images = 0

    for image_id, image_info in images_by_id.items():
        file_name = image_info["file_name"]
        gt_lines = lines_by_image.get(image_id, [])
        if not gt_lines:
            continue
        p = pred.get(file_name)
        if p is None:
            missing_images += 1
            continue

        keys_dict = p[0]
        centers_dict = p[1]
        keys = keys_dict.get("0", keys_dict.get(0, []))
        centers = centers_dict.get("0", centers_dict.get(0, []))
        keys = [k for k in keys if k[0] > score_threshold]
        centers = [c for c in centers if c[0] > score_threshold]
        key_scores.extend([k[0] for k in keys])
        center_scores.extend([c[0] for c in centers])
        key_xy = np.array([(k[2], k[3]) for k in keys], dtype=np.float32) if keys else np.zeros((0, 2), dtype=np.float32)

        for bbox in gt_lines:
            gt_pts = np.array([(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)], dtype=np.float32)
            if gt_pts.size == 0:
                continue
            if key_xy.size == 0:
                recall = 0.0
            else:
                dist = ((gt_pts[:, None, :] - key_xy[None, :, :]) ** 2).sum(axis=2) ** 0.5
                recall = float((dist.min(axis=1) <= distance_threshold).mean())
            line_recalls.append(recall)
            line_total += 1
            if recall >= 0.5:
                line_hit_count += 1

    karr = np.array(key_scores, dtype=np.float32)
    carr = np.array(center_scores, dtype=np.float32)
    larr = np.array(line_recalls, dtype=np.float32)

    return {
        "missing_pred_images": int(missing_images),
        "key_score_mean": _mean(karr),
        "key_score_p50": _percentile(karr, 50),
        "key_score_p90": _percentile(karr, 90),
        "center_score_mean": _mean(carr),
        "center_score_p50": _percentile(carr, 50),
        "center_score_p90": _percentile(carr, 90),
        "line_point_recall_mean": _mean(larr),
        "line_point_recall_p50": _percentile(larr, 50),
        "line_point_recall_p90": _percentile(larr, 90),
        "lines_recall_lt_0_2_ratio": float((larr < 0.2).mean()) if larr.size else float("nan"),
        "lines_recall_ge_0_5_ratio": float((larr >= 0.5).mean()) if larr.size else float("nan"),
        "line_hit_ratio_ge_0_5": (line_hit_count / line_total) if line_total > 0 else float("nan"),
        "num_lines_eval": int(line_total),
    }


def _print_block(title: str, stats: dict) -> None:
    print(f"\n[{title}]")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose line-point pipeline and generator outputs.")
    parser.add_argument("--data_dir", default="line_data", help="Dataset root in COCO format")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="COCO split")
    parser.add_argument("--input_size", default="511,511", help="Model input size, e.g. 511,511")
    parser.add_argument("--output_size", default="128,128", help="Model output map size, e.g. 128,128")
    parser.add_argument("--prediction_json", default=None, help="Optional prediction JSON from val_extraction.py")
    parser.add_argument("--score_threshold", type=float, default=0.0, help="Score threshold for prediction analysis")
    parser.add_argument("--distance_threshold", type=float, default=4.0, help="Distance threshold (px) for line recall")
    args = parser.parse_args()

    ann_path = os.path.join(args.data_dir, "annotations", f"{args.split}.json")
    coco = _load_coco(ann_path)
    in_h, in_w = [int(x) for x in args.input_size.split(",")]
    out_h, out_w = [int(x) for x in args.output_size.split(",")]

    _print_block("dataset", dataset_stats(coco))
    _print_block(
        "preprocess_spacing",
        preprocess_spacing_stats(
            coco=coco,
            data_dir=args.data_dir,
            split=args.split,
            input_size=(in_h, in_w),
            output_size=(out_h, out_w),
        ),
    )

    if args.prediction_json:
        _print_block(
            "prediction",
            prediction_stats(
                coco=coco,
                prediction_json=args.prediction_json,
                score_threshold=args.score_threshold,
                distance_threshold=args.distance_threshold,
            ),
        )


if __name__ == "__main__":
    main()
