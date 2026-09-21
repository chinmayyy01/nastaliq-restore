"""Feasibility spike: can we extract a sane stroke-topology graph from clean
Nastaliq renderings?

Pipeline per image:
    grayscale -> ink mask -> 1px skeleton -> keypoints (junctions/endpoints)
    -> segments between keypoints -> graph (nodes = keypoints, edges = segments)

Outputs an overlay per sample (skeleton, coloured segments as edges, red
junctions, blue endpoints) plus a montage, a 4-panel detail figure, and a
stats JSON. This is analysis only -- no model is trained.

Usage:
    python experiments/graph_feasibility.py --num 20
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.measure import label
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

MANIFEST = os.path.join(_REPO_ROOT, "data", "manifest.json")
OUT_DIR = os.path.join(_REPO_ROOT, "results", "graph_feasibility")

_NEIGHBOURS = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=np.uint8)


def extract_graph(gray: np.ndarray, ink_threshold: int = 128):
    """Return skeleton, keypoint/segment label maps and a node/edge graph."""
    ink = (gray < ink_threshold).astype(np.uint8)
    skel = skeletonize(ink > 0)

    # NB: scipy.ndimage.convolve, not cv2.filter2D -- the latter returned
    # wrong neighbour counts on this skeleton during the spike.
    neighbour_count = convolve(skel.astype(np.int32), _NEIGHBOURS, mode="constant")
    neighbour_count = neighbour_count * skel.astype(np.int32)

    endpoints = skel & (neighbour_count == 1)
    junctions = skel & (neighbour_count >= 3)
    keypoints = endpoints | junctions

    node_labels, n_nodes = label(keypoints, connectivity=2, return_num=True)
    segments_mask = skel & ~keypoints
    seg_labels, n_segments = label(segments_mask, connectivity=2, return_num=True)

    skel_labels, n_components = label(skel, connectivity=2, return_num=True)
    comp_sizes = np.bincount(skel_labels.ravel())[1:]

    nodes = {}
    node_type = {}
    for nid in range(1, n_nodes + 1):
        ys, xs = np.where(node_labels == nid)
        nodes[nid] = (float(xs.mean()), float(ys.mean()))
        node_type[nid] = "junction" if junctions[ys, xs].any() else "endpoint"

    edges = []
    h, w = skel.shape
    for sid in range(1, n_segments + 1):
        ys, xs = np.where(seg_labels == sid)
        touched = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = ys + dy, xs + dx
                valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
                vals = node_labels[yy[valid], xx[valid]]
                touched.update(int(v) for v in vals[vals > 0])
        touched = sorted(touched)
        for i in range(len(touched)):
            for j in range(i + 1, len(touched)):
                edges.append((touched[i], touched[j], sid))

    return {
        "ink": ink.astype(bool),
        "skeleton": skel,
        "endpoints": endpoints,
        "junctions": junctions,
        "node_labels": node_labels,
        "seg_labels": seg_labels,
        "nodes": nodes,
        "node_type": node_type,
        "edges": edges,
        "n_segments": n_segments,
        "n_components": int(n_components),
        "n_dots": int((comp_sizes <= 12).sum()),
    }


def render_overlay(gray: np.ndarray, g: dict) -> np.ndarray:
    base = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    base = (base * 0.45 + 255 * 0.55).astype(np.uint8)

    base[g["skeleton"]] = (170, 170, 170)

    rng = np.random.default_rng(0)
    palette = (rng.integers(40, 230, size=(g["n_segments"] + 1, 3))).astype(np.uint8)
    for sid in range(1, g["n_segments"] + 1):
        base[g["seg_labels"] == sid] = palette[sid]

    for nid, (x, y) in g["nodes"].items():
        colour = (220, 0, 0) if g["node_type"][nid] == "junction" else (0, 0, 220)
        cv2.circle(base, (int(x), int(y)), 3, colour, -1, cv2.LINE_AA)

    return base


def stats_for(g: dict) -> dict:
    degree = {}
    for a, b, _ in g["edges"]:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    degrees = list(degree.values())
    n_junction = sum(1 for t in g["node_type"].values() if t == "junction")
    return {
        "nodes": len(g["nodes"]),
        "junctions": n_junction,
        "endpoints": len(g["nodes"]) - n_junction,
        "segments": g["n_segments"],
        "edges": len(g["edges"]),
        "components": g["n_components"],
        "dots": g["n_dots"],
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(max(degrees)) if degrees else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest[:args.num]

    overlays = []
    all_stats = []
    for i, item in enumerate(samples):
        path = os.path.join(_REPO_ROOT, item["clean_path"])
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        g = extract_graph(gray)
        overlay = render_overlay(gray, g)
        overlays.append((item["id"], overlay))
        st = stats_for(g)
        st["id"] = item["id"]
        all_stats.append(st)
        cv2.imwrite(os.path.join(args.out_dir, f"overlay_{item['id']:04d}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if i == 0:
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            panels = [
                (gray, "1. clean grayscale"),
                (g["ink"].astype(np.uint8) * 255, "2. binarised ink mask"),
                (g["skeleton"].astype(np.uint8) * 255, "3. 1px skeleton"),
                (overlay, "4. graph overlay (red=junction, blue=endpoint, "
                          "coloured stroke=segment/edge)"),
            ]
            for ax, (img, title) in zip(axes, panels):
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                ax.set_title(title, fontsize=12)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "detail_sample_0000.png"), dpi=130)
            plt.close(fig)

    # montage
    cols = 2
    rows = int(np.ceil(len(overlays) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (sid, overlay) in zip(axes, overlays):
        ax.imshow(overlay)
        ax.set_title(f"sample {sid}", fontsize=9)
        ax.axis("off")
    for ax in axes[len(overlays):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "montage.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    agg = {k: float(np.mean([s[k] for s in all_stats]))
           for k in ("nodes", "junctions", "endpoints", "segments", "edges",
                     "components", "dots", "mean_degree")}
    print(f"analysed {len(all_stats)} images")
    print("mean per image: " + ", ".join(f"{k}={v:.1f}" for k, v in agg.items()))
    print(f"outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
