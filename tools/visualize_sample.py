import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import system_configs  # noqa: E402
from db.datasets import datasets  # noqa: E402
from sampling_function import sample_data  # noqa: E402


def _denormalize(img, mean, std):
    # img: (3, H, W), float32
    img = img * std[:, None, None] + mean[:, None, None]
    return np.clip(img, 0.0, 1.0)


def _save_overlay(image, heatmap, out_path, title):
    plt.figure(figsize=(6, 4))
    plt.imshow(image.transpose(1, 2, 0))
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="KPDetection")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--cache_path", default="./data/cache")
    parser.add_argument("--split", default="trainchart", choices=["trainchart", "valchart", "testchart"])
    parser.add_argument("--out_dir", default="./debug_samples")
    parser.add_argument("--num", type=int, default=2)
    args = parser.parse_args()

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["data_dir"] = args.data_dir
    configs["system"]["cache_dir"] = args.cache_path
    configs["system"]["dataset"] = "Chart"
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    dataset = datasets[system_configs.dataset](configs["db"], args.split)

    os.makedirs(args.out_dir, exist_ok=True)

    k_ind = 0
    data, _ = sample_data(dataset, k_ind)
    images = data["xs"][0].numpy()
    key_heatmaps = data["ys"][0].numpy()
    center_heatmaps = data["ys"][1].numpy()

    mean = dataset.mean
    std = dataset.std

    batch = min(args.num, images.shape[0])
    for i in range(batch):
        img = _denormalize(images[i], mean, std)
        key_hm = key_heatmaps[i].max(axis=0)
        cen_hm = center_heatmaps[i].max(axis=0)

        base_path = os.path.join(args.out_dir, f"sample_{i}")
        plt.imsave(base_path + "_image.png", img.transpose(1, 2, 0))
        _save_overlay(img, key_hm, base_path + "_key_overlay.png", "Key Heatmap Overlay")
        _save_overlay(img, cen_hm, base_path + "_center_overlay.png", "Center Heatmap Overlay")

        plt.imsave(base_path + "_key_heatmap.png", key_hm, cmap="jet")
        plt.imsave(base_path + "_center_heatmap.png", cen_hm, cmap="jet")

    print(f"Saved {batch} samples to {args.out_dir}")


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    main()
