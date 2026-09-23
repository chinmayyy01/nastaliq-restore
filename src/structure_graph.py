import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

MANIFEST = os.path.join(_REPO_ROOT, "data", "manifest.json")
OUT_DIR = os.path.join(_REPO_ROOT, "results", "structure_graph")
NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)


def ink_mask(gray, threshold=128):
    return gray < threshold


def ink_skeleton(ink):
    return skeletonize(ink)


def neighbour_counts(skel):
    counts = convolve(skel.astype(np.int32), NEIGHBOURS, mode="constant")
    return counts * skel.astype(np.int32)


def component_table(ink, skel):
    labels, n = label(ink, connectivity=2, return_num=True)
    rows = []
    for region in regionprops(labels):
        comp_skel = skel & (labels == region.label)
        min_y, min_x, max_y, max_x = region.bbox
        rows.append({
            "id": int(region.label),
            "area": int(region.area),
            "skeleton_length": int(comp_skel.sum()),
            "centroid": [float(region.centroid[1]), float(region.centroid[0])],
            "bbox": [int(min_x), int(min_y), int(max_x), int(max_y)],
        })
    return labels, rows


def split_strokes_dots(rows, dot_max_area):
    strokes = [r for r in rows if r["area"] > dot_max_area]
    dots = [r for r in rows if r["area"] <= dot_max_area]
    return strokes, dots


def associate_dots(dots, strokes, skel, labels):
    point_blocks = []
    id_blocks = []
    for stroke in strokes:
        ys, xs = np.where((labels == stroke["id"]) & skel)
        if len(xs) == 0:
            continue
        point_blocks.append(np.column_stack([xs, ys]).astype(np.float64))
        id_blocks.append(np.full(len(xs), stroke["id"], dtype=np.int64))
    if not point_blocks:
        return []
    points = np.vstack(point_blocks)
    point_ids = np.concatenate(id_blocks)
    tree = cKDTree(points)
    associations = []
    for dot in dots:
        distance, index = tree.query(np.asarray(dot["centroid"], dtype=np.float64))
        associations.append({
            "dot": dot["id"],
            "stroke": int(point_ids[index]),
            "distance": float(distance),
            "attach_point": [float(points[index][0]), float(points[index][1])],
        })
    return associations


def order_strokes(strokes):
    ordered = sorted(strokes, key=lambda r: -r["centroid"][0])
    for position, stroke in enumerate(ordered, start=1):
        stroke["order"] = position
    return ordered


def build_structure_graph(gray, dot_max_area):
    ink = ink_mask(gray)
    skel = ink_skeleton(ink)
    counts = neighbour_counts(skel)
    endpoints = int(((counts == 1) & skel).sum())
    junctions = int(((counts >= 3) & skel).sum())
    labels, rows = component_table(ink, skel)
    strokes, dots = split_strokes_dots(rows, dot_max_area)
    strokes = order_strokes(strokes)
    associations = associate_dots(dots, strokes, skel, labels)
    graph = {
        "n_components": len(rows),
        "n_strokes": len(strokes),
        "n_dots": len(dots),
        "endpoints": endpoints,
        "junctions": junctions,
        "strokes": strokes,
        "dots": dots,
        "dot_associations": associations,
    }
    return graph, labels, skel


def render_overlay(gray, graph, labels, skel):
    base = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    base = (base * 0.35 + 255 * 0.65).astype(np.uint8)
    cmap = plt.get_cmap("tab20")
    for stroke in graph["strokes"]:
        color = cmap((stroke["order"] - 1) % 20)[:3]
        rgb = tuple(int(255 * c) for c in color)
        base[labels == stroke["id"]] = rgb
    for dot in graph["dots"]:
        cx, cy = int(dot["centroid"][0]), int(dot["centroid"][1])
        cv2.circle(base, (cx, cy), 3, (0, 0, 220), -1, cv2.LINE_AA)
    for assoc in graph["dot_associations"]:
        dot = next(d for d in graph["dots"] if d["id"] == assoc["dot"])
        start = (int(dot["centroid"][0]), int(dot["centroid"][1]))
        end = (int(assoc["attach_point"][0]), int(assoc["attach_point"][1]))
        cv2.line(base, start, end, (200, 0, 200), 1, cv2.LINE_AA)
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--dot-max-area", type=int, default=90)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = manifest[:args.num]

    overlays = []
    all_stats = []
    first_graph = None
    for index, item in enumerate(samples):
        path = os.path.join(_REPO_ROOT, item["clean_path"])
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        graph, labels, skel = build_structure_graph(gray, args.dot_max_area)
        overlay = render_overlay(gray, graph, labels, skel)
        overlays.append((item["id"], overlay))
        strokes_with_dots = {a["stroke"] for a in graph["dot_associations"]}
        all_stats.append({
            "id": item["id"],
            "components": graph["n_components"],
            "strokes": graph["n_strokes"],
            "dots": graph["n_dots"],
            "junctions": graph["junctions"],
            "endpoints": graph["endpoints"],
            "strokes_with_dots": len(strokes_with_dots),
            "mean_assoc_distance": float(np.mean([a["distance"] for a in graph["dot_associations"]])) if graph["dot_associations"] else 0.0,
        })
        if index == 0:
            first_graph = graph
        cv2.imwrite(os.path.join(args.out_dir, f"overlay_{item['id']:04d}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    with open(os.path.join(args.out_dir, "structure_graph_0000.json"), "w", encoding="utf-8") as handle:
        json.dump(first_graph, handle, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as handle:
        json.dump(all_stats, handle, indent=2)

    columns = 2
    rows = int(np.ceil(len(overlays) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(18, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, (sample_id, overlay) in zip(axes, overlays):
        axis.imshow(overlay)
        axis.set_title(f"sample {sample_id}", fontsize=9)
        axis.axis("off")
    for axis in axes[len(overlays):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "montage.png"), dpi=120)
    plt.close(fig)

    aggregate = {
        key: float(np.mean([s[key] for s in all_stats]))
        for key in ("components", "strokes", "dots", "junctions", "endpoints", "strokes_with_dots", "mean_assoc_distance")
    }
    print(f"analysed {len(all_stats)} images")
    print("mean per image: " + ", ".join(f"{k}={v:.1f}" for k, v in aggregate.items()))
    print(f"outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
