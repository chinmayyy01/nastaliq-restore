"""Generate the restoration figure pack.

Outputs into ``results/figures/``:
    montage_synthetic.png  damaged | mask | baseline | RITL | clean  (synthetic)
    montage_real.png       same, on UTRSet-Real lines
    curves_training.png    training curves for all synthetic restoration variants
    curves_real.png        training curves for the real-data runs
    comparison.png         CER bars (synthetic + real) with bootstrap 95% CIs

Usage:
    python experiments/make_restoration_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from models.restoration_unet import RestorationUNet  # noqa: E402
from restoration_data import _load_gray, load_manifest  # noqa: E402

OUT = os.path.join(_REPO_ROOT, "results", "figures")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def commafy(x, _=None):
    return f"{x:.3f}"


def build_restorer(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    base = ckpt.get("args", {}).get("base", 32)
    model = RestorationUNet(in_channels=2, base=base).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def restore_array(model, damaged, mask):
    h, w = damaged.shape
    tw = ((w + 7) // 8) * 8
    dp = np.pad(damaged, ((0, 0), (0, tw - w)), constant_values=1.0)
    mp = np.pad(mask, ((0, 0), (0, tw - w)), constant_values=0.0)
    d_t = torch.from_numpy(dp)[None, None].to(DEVICE)
    m_t = torch.from_numpy(mp)[None, None].to(DEVICE)
    with torch.no_grad():
        out = model(torch.cat([d_t, m_t], dim=1))[0, 0, :, :w].cpu().numpy()
    return out


def montage(manifest_path, baseline_ckpt, ritl_ckpt, out_path, n=4, title=""):
    manifest = load_manifest(manifest_path)[:n]
    base = build_restorer(baseline_ckpt)
    ritl = build_restorer(ritl_ckpt)
    cols = ["damaged (input)", "damage mask", "baseline U-Net", "ours (RITL)", "clean (target)"]
    fig, axes = plt.subplots(n, 5, figsize=(17, 2.5 * n))
    axes = np.atleast_2d(axes)
    for c, name in enumerate(cols):
        axes[0, c].set_title(name, fontsize=12)
    for r, item in enumerate(manifest):
        damaged = _load_gray(item["damaged_path"], 64)
        mask = _load_gray(item["mask_path"], 64)
        clean = _load_gray(item["clean_path"], 64)
        w = min(damaged.shape[1], mask.shape[1], clean.shape[1])
        damaged, mask, clean = damaged[:, :w], mask[:, :w], clean[:, :w]
        panels = [damaged, mask, restore_array(base, damaged, mask),
                  restore_array(ritl, damaged, mask), clean]
        for c, img in enumerate(panels):
            axes[r, c].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _history(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def curves(histories, out_path, title):
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()
    for label, path, color in histories:
        h = _history(path)
        if not h:
            continue
        ep = [e["epoch"] for e in h]
        if "psnr" in h[0]:
            ax1.plot(ep, [e["psnr"] for e in h], "-", color=color, label=f"{label} (PSNR)")
        if h[0].get("cer") is not None:
            ax2.plot(ep, [e["cer"] for e in h], "--", marker=".", color=color,
                     label=f"{label} (CER)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("PSNR (higher better)")
    ax2.set_ylabel("val CER (lower better)")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    if lines:
        ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def _cer_per_sample(path, key):
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)["per_sample"]
    from train_baseline import levenshtein
    d = np.array([levenshtein(r[key], r["ref"]) for r in recs], float)
    L = np.array([max(1, len(r["ref"])) for r in recs], float)
    rng = np.random.default_rng(0)
    N = len(d)
    boots = [d[rng.integers(0, N, N)].sum() / L[rng.integers(0, N, N)].sum() for _ in range(1000)]
    return d.sum() / L.sum(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def comparison_bars(out_path):
    # (label, damaged, baseline, ritl, clean) computed from eval JSONs
    sets = {
        "synthetic": ("results/stats/baseline.json", "results/stats/ritl.json"),
        "real": ("results/stats/real/baseline.json", "results/stats/real/ritl.json"),
    }
    methods = ["damaged", "baseline", "RITL", "clean"]
    keys = {"damaged": "damaged", "baseline": "restored", "RITL": "restored", "clean": "clean"}
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = np.arange(len(methods))
    for i, (setname, (bpath, rpath)) in enumerate(sets.items()):
        vals, errs = [], []
        for m in methods:
            path = bpath if m in ("damaged", "baseline", "clean") else rpath
            cer, lo, hi = _cer_per_sample(os.path.join(_REPO_ROOT, path), keys[m])
            vals.append(cer)
            errs.append([cer - lo, hi - cer])
        errs = np.array(errs).T
        ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs, capsize=3,
               label=setname, color=["#93c5fd", "#2563eb"][i])
        for xi, v in zip(x + (i - 0.5) * width, vals):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("val CER (lower better)")
    ax.set_title("Restoration comparison (95% CI)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    data = os.path.join(_REPO_ROOT, "data")
    data_real = os.path.join(_REPO_ROOT, "data_real")
    ck = os.path.join(_REPO_ROOT, "checkpoints")

    montage(os.path.join(data, "manifest_val.json"),
            os.path.join(ck, "restoration_hard.pt"),
            os.path.join(ck, "restoration_hard_ocr.pt"),
            os.path.join(OUT, "montage_synthetic.png"),
            title="Synthetic Nastaliq — restoration")
    montage(os.path.join(data_real, "manifest_val.json"),
            os.path.join(ck, "real_restoration.pt"),
            os.path.join(ck, "real_ritl.pt"),
            os.path.join(OUT, "montage_real.png"),
            title="UTRSet-Real (real printed Urdu) — restoration")

    curves([
        ("baseline", os.path.join(_REPO_ROOT, "results/restoration_hard_history.json"), "#2563eb"),
        ("RITL", os.path.join(_REPO_ROOT, "results/restoration_hard_ocr_history.json"), "#dc2626"),
        ("flow", os.path.join(_REPO_ROOT, "results/restoration_flow_history.json"), "#16a34a"),
        ("flow+RITL", os.path.join(_REPO_ROOT, "results/restoration_flow_ocr_history.json"), "#9333ea"),
        ("iterative RITL", os.path.join(_REPO_ROOT, "results/restoration_iterative_history.json"), "#ea580c"),
    ], os.path.join(OUT, "curves_training.png"), "Synthetic restoration — training")

    curves([
        ("restoration", os.path.join(_REPO_ROOT, "results/real_restoration_history.json"), "#2563eb"),
        ("RITL", os.path.join(_REPO_ROOT, "results/real_ritl_history.json"), "#dc2626"),
    ], os.path.join(OUT, "curves_real.png"), "Real (UTRSet-Real) restoration — training")

    comparison_bars(os.path.join(OUT, "comparison.png"))


if __name__ == "__main__":
    main()
