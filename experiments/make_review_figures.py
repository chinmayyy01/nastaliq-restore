"""Generate the review-1 figure pack from the current repo state.

Outputs into results/:
    01_pipeline.png        clean | damaged | damage-mask grid
    03_training_curve.png  train/val loss + val CER vs epoch
    04_architecture.png    CNN + BiLSTM + CTC block diagram

(The graph figures come from experiments/graph_feasibility.py, which writes
results/graph_feasibility/.)

Usage:
    python experiments/make_review_figures.py
"""

import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_REPO_ROOT, "results")
MANIFEST = os.path.join(_REPO_ROOT, "data", "manifest.json")
HISTORY = os.path.join(RESULTS, "train_history.json")


def _load(rel_path):
    return cv2.imread(os.path.join(_REPO_ROOT, rel_path), cv2.IMREAD_GRAYSCALE)


def figure_pipeline(out_path, n=3):
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)[:n]

    fig, axes = plt.subplots(n, 3, figsize=(16, 3.0 * n))
    axes = np.atleast_2d(axes)
    columns = [("clean (ground truth)", "clean_path"),
               ("damaged (input)", "damaged_path"),
               ("damage mask (supervision)", "mask_path")]
    for c, (title, key) in enumerate(columns):
        axes[0, c].set_title(title, fontsize=13)
    for r, item in enumerate(manifest):
        for c, (_, key) in enumerate(columns):
            axes[r, c].imshow(_load(item[key]), cmap="gray")
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def figure_architecture(out_path):
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    stages = [
        ("Input  [N, 1, 64, W]", "clean grayscale line", "#dbeafe"),
        ("CNN: 4 conv blocks", "32->64->128->256 ch, BN+ReLU+MaxPool\n"
                               "pool (2,2),(2,2),(2,1),(2,1)", "#c7f9cc"),
        ("Feature map  [N, 256, 4, W/4]", "height kept, width = sequence axis", "#e9d8fd"),
        ("Collapse height -> channels", "[T = W/4, N, 1024]", "#e9d8fd"),
        ("2-layer BiLSTM (hidden 128)", "reads both directions -> [T, N, 256]", "#fde68a"),
        ("Linear -> vocab, log_softmax", "log_probs [T, N, vocab_size]", "#fca5a5"),
        ("CTC loss (train) / greedy decode (infer)", "no character alignment needed", "#f3d1d1"),
    ]

    y = 11.2
    for i, (title, sub, colour) in enumerate(stages):
        box = FancyBboxPatch((1.2, y - 0.75), 7.6, 1.0,
                             boxstyle="round,pad=0.08", linewidth=1.2,
                             edgecolor="#333333", facecolor=colour)
        ax.add_patch(box)
        ax.text(5.0, y - 0.18, title, ha="center", va="center",
                fontsize=12, fontweight="bold")
        ax.text(5.0, y - 0.56, sub, ha="center", va="center", fontsize=9.5)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrow(5.0, y - 0.82, 0, -0.28, width=0.03,
                                    head_width=0.18, head_length=0.12,
                                    length_includes_head=True, color="#333333"))
        y -= 1.45

    ax.text(5.0, 11.75, "Baseline recogniser: CNN + BiLSTM + CTC",
            ha="center", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def figure_training_curve(history_path, out_path):
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_cer = [h["val_cer"] for h in history]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(epochs, train_loss, "o-", label="train loss", color="#2563eb")
    ax1.plot(epochs, val_loss, "s-", label="val loss", color="#16a34a")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("CTC loss")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_cer, "^--", label="val CER", color="#dc2626")
    ax2.set_ylabel("val CER (lower is better)")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    best = min(history, key=lambda h: h["val_cer"])
    ax1.set_title(f"Baseline training on synthetic data "
                  f"(best val CER {best['val_cer']:.3f} @ epoch {best['epoch']})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    figure_pipeline(os.path.join(RESULTS, "01_pipeline.png"))
    figure_architecture(os.path.join(RESULTS, "04_architecture.png"))
    print("wrote 01_pipeline.png, 04_architecture.png")
    if os.path.exists(HISTORY):
        figure_training_curve(HISTORY, os.path.join(RESULTS, "03_training_curve.png"))
        print("wrote 03_training_curve.png")
    else:
        print(f"skipped training curve (no {HISTORY})")


if __name__ == "__main__":
    main()
