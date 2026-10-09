"""Train the structure-completion network (novelty #1).

Target is the clean line's structure prior (skeleton + stroke width) computed
by ``structure_prior.structure_channels``. The network sees only the damaged
line and the damage mask, so it must infer the missing topology from context.

    python src/train_structure.py --epochs 20
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from restoration_data import RestorationDataset, restoration_collate
from models.structure_completion import StructureCompletion


def skeleton_iou(pred, target, threshold=0.5):
    """IoU of the predicted vs target skeleton channel."""
    p = pred[:, 0] > threshold
    t = target[:, 0] > threshold
    inter = (p & t).sum().item()
    union = (p | t).sum().item()
    return inter / union if union else 1.0


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total, count, iou_total = 0.0, 0, 0.0
    l1 = nn.L1Loss()
    torch.set_grad_enabled(train)
    for damaged, mask, _, structure, _, _ in loader:
        damaged, mask, structure = damaged.to(device), mask.to(device), structure.to(device)
        pred = model(torch.cat([damaged, mask], dim=1))
        loss = l1(pred, structure)
        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += loss.item() * damaged.shape[0]
        count += damaged.shape[0]
        iou_total += skeleton_iou(pred.detach(), structure) * damaged.shape[0]
    torch.set_grad_enabled(True)
    return total / count if count else 0.0, iou_total / count if count else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train the structure-completion net")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--data-dir", default=os.path.join(_REPO_ROOT, "data"))
    parser.add_argument("--checkpoint", default=os.path.join(
        _REPO_ROOT, "checkpoints", "structure_completion.pt"))
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

    train_ds = RestorationDataset(os.path.join(args.data_dir, "manifest_train.json"),
                                  args.img_height, structure_source="clean")
    val_ds = RestorationDataset(os.path.join(args.data_dir, "manifest_val.json"),
                                args.img_height, structure_source="clean")
    if args.limit_train:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))
    print(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=restoration_collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=restoration_collate, num_workers=args.num_workers)

    model = StructureCompletion(in_channels=2, base=args.base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_loader)),
        pct_start=0.1)

    best_iou = float("-inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_iou = run_epoch(model, train_loader, optimizer, device, True)
        scheduler.step()
        val_loss, val_iou = run_epoch(model, val_loader, optimizer, device, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_iou": train_iou, "val_iou": val_iou})
        print(f"epoch {epoch:>3}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"skelIoU {val_iou:.4f} | {time.time()-start:.1f}s")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "epoch": epoch, "l1": val_loss,
                        "iou": val_iou, "args": vars(args)}, args.checkpoint)
            print(f"    saved -> {args.checkpoint} (best skelIoU {best_iou:.4f})")

    if args.history_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.history_out)), exist_ok=True)
        with open(args.history_out, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    print(f"done. best val skeleton IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
