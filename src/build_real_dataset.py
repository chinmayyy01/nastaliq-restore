"""Build a *real-text* dataset from UTRSet-Real (printed Urdu lines).

We take real scanned/printed line images (UTRNet, ICDAR 2023; CC BY-NC-SA,
research use) as the clean targets, then apply our synthetic damage simulator
so the (damaged, mask, clean) pipeline can be evaluated on genuinely real
Nastaliq renderings rather than our own font rendering.

    python src/build_real_dataset.py \
        --src /path/to/UTRSet-Real --out data_real --img-height 64 --severity 1.5

Layout produced (paths stored relative to the repo root):
    <out>/clean/*.png, <out>/damaged/*.png, <out>/damage_masks/*.png
    <out>/manifest_train.json, <out>/manifest_val.json
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from damage_simulator import simulate_damage


def _read_gt(gt_path):
    rows = []
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            rel, text = line.split("\t", 1)
            rows.append((rel, text))
    return rows


def _to_gray_64(img_path, img_height):
    img = Image.open(img_path).convert("L")
    w, h = img.size
    new_w = max(1, int(round(w * img_height / h)))
    return img.resize((new_w, img_height), Image.BILINEAR)


def build_split(split_dir, split_name, out_dir, img_height, severity, start_index):
    gt = _read_gt(os.path.join(split_dir, "gt.txt"))
    # gt paths look like "test/<name>.jpg"; the images live under <split>/test/
    manifest = []
    clean_dir = os.path.join(out_dir, "clean")
    damaged_dir = os.path.join(out_dir, "damaged")
    mask_dir = os.path.join(out_dir, "damage_masks")
    for i, (rel, text) in enumerate(gt):
        img_path = os.path.join(split_dir, rel)
        if not os.path.exists(img_path):
            img_path = os.path.join(split_dir, os.path.basename(rel))
        if not os.path.exists(img_path):
            continue
        clean = _to_gray_64(img_path, img_height)
        damaged, mask = simulate_damage(clean, seed=start_index + i, severity=severity)
        name = f"{split_name}_{i:05d}.png"
        clean.save(os.path.join(clean_dir, name))
        damaged.save(os.path.join(damaged_dir, name))
        mask.save(os.path.join(mask_dir, name))
        manifest.append({
            "id": f"{split_name}_{i}",
            "text": text,
            "clean_path": os.path.relpath(os.path.join(clean_dir, name), _REPO_ROOT).replace(os.sep, "/"),
            "damaged_path": os.path.relpath(os.path.join(damaged_dir, name), _REPO_ROOT).replace(os.sep, "/"),
            "mask_path": os.path.relpath(os.path.join(mask_dir, name), _REPO_ROOT).replace(os.sep, "/"),
        })
        if (i + 1) % 1000 == 0:
            print(f"  {split_name}: {i + 1}/{len(gt)}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="UTRSet-Real root (has train/ and test/)")
    parser.add_argument("--out", default=os.path.join(_REPO_ROOT, "data_real"))
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--severity", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    for d in ("clean", "damaged", "damage_masks"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    train = build_split(os.path.join(args.src, "train"), "train", args.out,
                        args.img_height, args.severity, 0)
    val = build_split(os.path.join(args.src, "test"), "val", args.out,
                      args.img_height, args.severity, 100000)
    if args.limit:
        train, val = train[:args.limit], val[:max(1, args.limit // 4)]

    for path, data in [("manifest_train.json", train), ("manifest_val.json", val)]:
        with open(os.path.join(args.out, path), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"real dataset: {len(train)} train / {len(val)} val -> {args.out}")


if __name__ == "__main__":
    main()
