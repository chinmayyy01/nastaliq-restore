"""Train the U-Net restoration baseline on damaged Nastaliq lines.

Supervision is the synthetic ground truth: the model sees the damaged line
plus the damage mask and is asked to reconstruct the clean line. The loss is
an L1 reconstruction term plus a mask-weighted L1 term that pushes the model
to fix exactly the corrupted regions.

Validation reports L1, PSNR and SSIM against the clean line, saving the best
checkpoint to ``checkpoints/restoration_best.pt``.

    python src/train_restoration.py --epochs 20
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Subset

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from restoration_data import RestorationDataset, restoration_collate
from models.restoration_unet import RestorationUNet


def masked_l1(pred, target, mask, eps: float = 1e-6):
    """L1 restricted to the damaged region (mask-weighted mean)."""
    diff = (pred - target).abs() * mask
    return diff.sum() / (mask.sum() + eps)


def build_input(damaged, mask, structure, use_structure: bool):
    """Assemble the network input; add the structure prior only when enabled."""
    if use_structure:
        return torch.cat([damaged, mask, structure], dim=1)
    return torch.cat([damaged, mask], dim=1)


def train_one_epoch(model, loader, optimizer, device, lambda_mask=1.0,
                    grad_clip=5.0, use_structure=False):
    model.train()
    total, count = 0.0, 0
    l1 = nn.L1Loss()
    for damaged, mask, clean, structure, _, _ in loader:
        damaged, mask, clean = damaged.to(device), mask.to(device), clean.to(device)
        structure = structure.to(device)
        pred = model(build_input(damaged, mask, structure, use_structure))
        loss = l1(pred, clean) + lambda_mask * masked_l1(pred, clean, mask)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * damaged.shape[0]
        count += damaged.shape[0]
    return total / count if count else 0.0


@torch.no_grad()
def evaluate(model, loader, device, lambda_mask=1.0, use_structure=False):
    model.eval()
    l1 = nn.L1Loss()
    total, count = 0.0, 0
    psnr_vals, ssim_vals = [], []
    for damaged, mask, clean, structure, _, _ in loader:
        damaged, mask, clean = damaged.to(device), mask.to(device), clean.to(device)
        structure = structure.to(device)
        pred = model(build_input(damaged, mask, structure, use_structure))
        loss = l1(pred, clean) + lambda_mask * masked_l1(pred, clean, mask)
        total += loss.item() * damaged.shape[0]
        count += damaged.shape[0]

        pred_np = pred.cpu().numpy()[:, 0]
        clean_np = clean.cpu().numpy()[:, 0]
        for p, c in zip(pred_np, clean_np):
            psnr_vals.append(peak_signal_noise_ratio(c, p, data_range=1.0))
            ssim_vals.append(structural_similarity(c, p, data_range=1.0))
    return (total / count if count else 0.0,
            float(np.mean(psnr_vals)), float(np.mean(ssim_vals)))


def main():
    parser = argparse.ArgumentParser(description="Train the restoration U-Net")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=32, help="U-Net base channels")
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--use-structure", action="store_true",
                        help="Feed the stroke-structure prior as extra input channels")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--data-dir", default=os.path.join(_REPO_ROOT, "data"))
    parser.add_argument("--checkpoint", default=os.path.join(
        _REPO_ROOT, "checkpoints", "restoration_best.pt"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--history-out", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    train_ds = RestorationDataset(
        os.path.join(args.data_dir, "manifest_train.json"), args.img_height,
        use_structure=args.use_structure)
    val_ds = RestorationDataset(
        os.path.join(args.data_dir, "manifest_val.json"), args.img_height,
        use_structure=args.use_structure)
    if args.limit_train:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))
    print(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=restoration_collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=restoration_collate, num_workers=args.num_workers)

    in_channels = 4 if args.use_structure else 2
    model = RestorationUNet(in_channels=in_channels, base=args.base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_loader)),
        pct_start=0.1)

    best_psnr = float("-inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     args.lambda_mask, args.grad_clip,
                                     args.use_structure)
        scheduler.step()
        val_loss, psnr, ssim = evaluate(model, val_loader, device,
                                        args.lambda_mask, args.use_structure)
        elapsed = time.time() - start
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "psnr": psnr, "ssim": ssim})
        print(f"epoch {epoch:>3}/{args.epochs} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | PSNR {psnr:.2f} | SSIM {ssim:.4f} | {elapsed:.1f}s")
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "psnr": psnr, "ssim": ssim, "args": vars(args)},
                       args.checkpoint)
            print(f"    saved -> {args.checkpoint} (best PSNR {best_psnr:.2f})")

    if args.history_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.history_out)), exist_ok=True)
        with open(args.history_out, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"history -> {args.history_out}")
    print(f"done. best val PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    main()
