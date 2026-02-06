import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sampling_function import _clip_detections, _full_image_crop, _resize_image


def _load_coco(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    return images, anns_by_image


def _bbox_to_points(bbox):
    pts = []
    for i in range(0, len(bbox), 2):
        x = bbox[i]
        y = bbox[i + 1] if i + 1 < len(bbox) else None
        if y is None:
            break
        pts.append((float(x), float(y)))
    return pts


def _pad_bboxes(bboxes, pad_value=-1.0):
    max_len = max((len(b) for b in bboxes), default=0)
    padded = []
    for b in bboxes:
        arr = list(b)
        if len(arr) < max_len:
            arr.extend([pad_value] * (max_len - len(arr)))
        padded.append(arr)
    if not padded:
        return np.zeros((0, 0), dtype=np.float32)
    return np.array(padded, dtype=np.float32)


def _draw_points(image, bboxes, color):
    out = image.copy()
    h, w = out.shape[:2]
    for bbox in bboxes:
        for x, y in _bbox_to_points(bbox):
            if x <= 0 or y <= 0:
                continue
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                cv2.circle(out, (xi, yi), 2, color, -1, lineType=cv2.LINE_AA)
    return out


def _side_by_side(left, right):
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[: left.shape[0], : left.shape[1]] = left
    out[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize KPDetection preprocessing for line charts.")
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--seed", default=317, type=int)
    parser.add_argument("--input_size", default="511,511", type=str)
    parser.add_argument(
        "--save_near_images",
        action="store_true",
        help="If set (or out_dir is omitted), save into <data_dir>/images/<split>_preproc_vis",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    input_size = [int(x) for x in args.input_size.split(",")]
    ann_path = os.path.join(args.data_dir, "annotations", f"{args.split}.json")
    img_dir = os.path.join(args.data_dir, "images", args.split)

    images, anns_by_image = _load_coco(ann_path)
    image_ids = list(images.keys())
    random.shuffle(image_ids)

    if args.out_dir:
        out_dir = args.out_dir
    else:
        # By default, save next to source split images for quick visual comparison.
        out_dir = os.path.join(args.data_dir, "images", f"{args.split}_preproc_vis")
    if args.save_near_images:
        out_dir = os.path.join(args.data_dir, "images", f"{args.split}_preproc_vis")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for image_id in image_ids:
        if saved >= args.num:
            break
        info = images[image_id]
        image_path = os.path.join(img_dir, info["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue

        anns = anns_by_image.get(image_id, [])
        line_bboxes = [ann["bbox"] for ann in anns if ann.get("category_id") == 2]
        if not line_bboxes:
            continue

        original_vis = _draw_points(image, line_bboxes, (0, 255, 0))

        detections = _pad_bboxes(line_bboxes)
        if detections.size == 0:
            continue

        cropped, detections = _full_image_crop(image, detections)
        resized, detections = _resize_image(cropped, detections, input_size)
        detections = _clip_detections(resized, detections)

        preproc_bboxes = detections.tolist()
        preproc_vis = _draw_points(resized, preproc_bboxes, (0, 0, 255))

        combined = _side_by_side(original_vis, preproc_vis)
        file_stem = os.path.splitext(info["file_name"])[0]
        cv2.imwrite(os.path.join(out_dir, f"{file_stem}__orig_points.jpg"), original_vis)
        cv2.imwrite(os.path.join(out_dir, f"{file_stem}__preproc_points.jpg"), preproc_vis)
        cv2.imwrite(os.path.join(out_dir, f"{file_stem}__side_by_side.jpg"), combined)
        saved += 1

    print(f"Saved {saved} samples to: {out_dir}")


if __name__ == "__main__":
    main()
