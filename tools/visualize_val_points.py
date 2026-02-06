#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize polyline points from COCO val annotations and save overlays."
    )
    parser.add_argument(
        "--data_dir",
        default="line_data",
        help="Dataset root containing annotations/ and images/val/",
    )
    parser.add_argument(
        "--ann_file",
        default=None,
        help="Path to COCO annotation JSON (default: <data_dir>/annotations/val.json)",
    )
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Path to val images directory (default: <data_dir>/images/val)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory to save visualized images (default: <data_dir>/vis_val_points)",
    )
    parser.add_argument(
        "--point_radius",
        type=int,
        default=3,
        help="Circle radius for points",
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        default=2,
        help="Polyline thickness",
    )
    parser.add_argument(
        "--hide_center",
        action="store_true",
        help="Disable center marker drawing",
    )
    return parser.parse_args()


def load_coco(ann_file: str) -> Tuple[Dict[int, dict], Dict[int, List[dict]]]:
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    return images_by_id, anns_by_image


def bbox_to_points(bbox: List[float]) -> List[Point]:
    points: List[Point] = []
    max_idx = len(bbox) - (len(bbox) % 2)
    for i in range(0, max_idx, 2):
        x = float(bbox[i])
        y = float(bbox[i + 1])
        points.append((x, y))
    return points


def line_center(points: List[Point]) -> Point:
    n = len(points)
    if n == 0:
        return (0.0, 0.0)
    if n % 2 == 0:
        p1 = points[n // 2 - 1]
        p2 = points[n // 2]
        return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
    return points[n // 2]


def color_for_idx(idx: int) -> Tuple[int, int, int]:
    palette = [
        (0, 255, 255),
        (0, 200, 0),
        (255, 140, 0),
        (255, 0, 180),
        (80, 80, 255),
        (255, 255, 0),
    ]
    return palette[idx % len(palette)]


def main() -> None:
    args = parse_args()

    ann_file = args.ann_file or os.path.join(args.data_dir, "annotations", "val.json")
    images_dir = args.images_dir or os.path.join(args.data_dir, "images", "val")
    out_dir = args.out_dir or os.path.join(args.data_dir, "vis_val_points")
    os.makedirs(out_dir, exist_ok=True)

    images_by_id, anns_by_image = load_coco(ann_file)

    saved = 0
    missing = 0
    for image_id, image_info in images_by_id.items():
        file_name = image_info["file_name"]
        image_path = os.path.join(images_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            missing += 1
            print(f"[WARN] image not found/readable: {image_path}")
            continue

        anns = anns_by_image.get(image_id, [])
        for ann_idx, ann in enumerate(anns):
            points = bbox_to_points(ann.get("bbox", []))
            if len(points) == 0:
                continue

            color = color_for_idx(ann_idx)
            # Draw polyline to preserve point order visibility.
            if len(points) > 1:
                poly = np.array(
                    [(int(round(x)), int(round(y))) for x, y in points],
                    dtype=np.int32,
                )
                cv2.polylines(
                    image,
                    [poly],
                    isClosed=False,
                    color=color,
                    thickness=args.line_thickness,
                    lineType=cv2.LINE_AA,
                )

            for x, y in points:
                cv2.circle(
                    image,
                    (int(round(x)), int(round(y))),
                    args.point_radius,
                    color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

            if not args.hide_center:
                cx, cy = line_center(points)
                cx_i = int(round(cx))
                cy_i = int(round(cy))
                # Draw center as a separate visual element (red point + white ring + label).
                cv2.circle(
                    image,
                    (cx_i, cy_i),
                    args.point_radius + 3,
                    (255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                cv2.circle(
                    image,
                    (cx_i, cy_i),
                    args.point_radius + 1,
                    (0, 0, 255),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
                cv2.drawMarker(
                    image,
                    (cx_i, cy_i),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=10,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    "C",
                    (cx_i + 6, cy_i - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        out_path = os.path.join(out_dir, file_name)
        cv2.imwrite(out_path, image)
        saved += 1

    print(f"Saved overlays: {saved} images -> {out_dir}")
    if missing:
        print(f"Missing/unreadable images: {missing}")


if __name__ == "__main__":
    main()
