"""Train the flow-matching restoration backbone (novelty #3).

Conditional flow matching: the model learns the velocity field that transports
noise to a clean line conditioned on the damaged line + mask. Validation
samples the ODE (Euler) and reports PSNR/SSIM against the clean target.

    python src/train_flow.py --epochs 20
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
from models.flow_restoration import FlowRestoration, cfm_loss, sample


def train_one_epoch(model, loader, optimizer, device, grad_clip=5.0):
    model.train()
    total, count = 0.0, 0
    for damaged, mask, clean, _, _, _ in loader:
        damaged, mask, clean = damaged.to(device), mask.to(device), clean.to(device)
        loss = cfm_loss(model, clean, damaged, mask)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * damaged.shape[0]
        count += damaged.shape[0]
    return total / count if count else 0.0


@torch.no_grad()
def evaluate(model, loader, device, steps=12, limit=None):
    model.eval()
    psnr_vals, ssim_vals = [], []
    for i, (damaged, mask, clean, _, _, _) in enumerate(loader):
        if limit is not None and i * loader.batch_size >= limit:
            break
        damaged, mask, clean = damaged.to(device), mask.to(device), clean.to(device)
        out = sample(model, damaged, mask, steps=steps)
        out_np = out.cpu().numpy()[:, 0]
        clean_np = clean.cpu().numpy()[:, 0]
        for p, c in zip(out_np, clean_np):
            psnr_vals.append(peak_signal_noise_ratio(c, p, data_range=1.0))
            ssim_vals.append(structural_similarity(c, p, data_range=1.0))
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))


def main():
    parser = argparse.ArgumentParser(description="Train the flow-matching restoration model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=12)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--data-dir", default=os.path.join(_REPO_ROOT, "data"))
    parser.add_argument("--checkpoint", default=os.path.join(
        _REPO_ROOT, "checkpoints", "restoration_flow.pt"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None,
                        help="Cap val samples used each epoch (speed)")
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

    train_ds = RestorationDataset(os.path.join(args.data_dir, "manifest_train.json"),
                                  args.img_height)
    val_ds = RestorationDataset(os.path.join(args.data_dir, "manifest_val.json"),
                                args.img_height)
    if args.limit_train:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))
    print(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=restoration_collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=restoration_collate, num_workers=args.num_workers)

    model = FlowRestoration(base=args.base, emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_loader)),
        pct_start=0.1)

    best_psnr = float("-inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip)
        scheduler.step()
        psnr, ssim = evaluate(model, val_loader, device, args.sample_steps, args.val_limit)
        history.append({"epoch": epoch, "train_loss": train_loss, "psnr": psnr, "ssim": ssim})
        print(f"epoch {epoch:>3}/{args.epochs} | cfm_loss {train_loss:.4f} | "
              f"PSNR {psnr:.2f} | SSIM {ssim:.4f} | {time.time()-start:.1f}s")
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save({"model": model.state_dict(), "epoch": epoch, "psnr": psnr,
                        "ssim": ssim, "flow": True, "args": vars(args)}, args.checkpoint)
            print(f"    saved -> {args.checkpoint} (best PSNR {best_psnr:.2f})")

    if args.history_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.history_out)), exist_ok=True)
        with open(args.history_out, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    print(f"done. best val PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    main()
