"""Iterative restoration with recognition-in-the-loop (novelty #4).

A single refinement U-Net is applied ``--stages`` times. Each pass takes the
current estimate together with the damaged input and mask:

    est_0 = damaged
    est_{k+1} = U([damaged, mask, est_k])

Every stage is supervised against the clean target (deep supervision) and the
final estimate also carries a CTC loss from the frozen recogniser, so
recognition guides the refinement rather than only scoring it. This is the
iterative form of the recognition-in-the-loop idea (cf. IMAGE, 2024).

    python src/train_iterative.py --epochs 20 --stages 2
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

from dataset import build_vocab, load_manifest
from restoration_data import RestorationDataset, restoration_collate
from models.restoration_unet import RestorationUNet
from models.baseline_recognizer import BaselineRecognizer, greedy_decode
from train_baseline import character_error_rate
from train_restoration import (masked_l1, ctc_loss_on_restored,
                               _infer_recognizer_config, prepare_frozen_recognizer)


def run_iterative(model, damaged, mask, stages):
    est = damaged
    outs = []
    for _ in range(stages):
        est = model(torch.cat([damaged, mask, est], dim=1))
        outs.append(est)
    return outs


def train_one_epoch(model, loader, optimizer, device, stages, lambda_mask=1.0,
                    grad_clip=5.0, recognizer=None, vocab=None,
                    ocr_criterion=None, lambda_ocr=0.0):
    model.train()
    total, count = 0.0, 0
    l1 = nn.L1Loss()
    for damaged, mask, clean, _, widths, texts in loader:
        damaged, mask, clean = damaged.to(device), mask.to(device), clean.to(device)
        outs = run_iterative(model, damaged, mask, stages)
        loss = sum(l1(e, clean) + lambda_mask * masked_l1(e, clean, mask) for e in outs)
        if recognizer is not None and lambda_ocr > 0:
            loss = loss + lambda_ocr * ctc_loss_on_restored(
                outs[-1], texts, widths, recognizer, vocab, ocr_criterion, device)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * damaged.shape[0]
        count += damaged.shape[0]
    return total / count if count else 0.0


@torch.no_grad()
def evaluate(model, loader, device, stages, recognizer=None, idx_to_char=None):
    model.eval()
    psnr_vals, ssim_vals = [], []
    preds, refs = [], []
    for damaged, mask, clean, _, _, texts in loader:
        damaged, mask, clean = damaged.to(device), mask.to(device), clean.to(device)
        final = run_iterative(model, damaged, mask, stages)[-1]
        if recognizer is not None and idx_to_char is not None:
            preds.extend(greedy_decode(recognizer(final).cpu(), idx_to_char))
            refs.extend(texts)
        f_np = final.cpu().numpy()[:, 0]
        c_np = clean.cpu().numpy()[:, 0]
        for p, c in zip(f_np, c_np):
            psnr_vals.append(peak_signal_noise_ratio(c, p, data_range=1.0))
            ssim_vals.append(structural_similarity(c, p, data_range=1.0))
    cer = character_error_rate(preds, refs) if refs else None
    return float(np.mean(psnr_vals)), float(np.mean(ssim_vals)), cer


def main():
    parser = argparse.ArgumentParser(description="Iterative RITL restoration")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--lambda-ocr", type=float, default=0.1)
    parser.add_argument("--ocr-checkpoint", default=os.path.join(
        _REPO_ROOT, "checkpoints", "baseline_best.pt"))
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--data-dir", default=os.path.join(_REPO_ROOT, "data"))
    parser.add_argument("--checkpoint", default=os.path.join(
        _REPO_ROOT, "checkpoints", "restoration_iterative.pt"))
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
                                  args.img_height)
    val_ds = RestorationDataset(os.path.join(args.data_dir, "manifest_val.json"),
                                args.img_height)
    if args.limit_train:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))
    print(f"train={len(train_ds)} val={len(val_ds)} stages={args.stages}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=restoration_collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=restoration_collate, num_workers=args.num_workers)

    model = RestorationUNet(in_channels=3, base=args.base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_loader)),
        pct_start=0.1)

    recognizer, vocab, idx_to_char, ocr_criterion = None, None, None, None
    if args.lambda_ocr > 0:
        tm = load_manifest(os.path.join(args.data_dir, "manifest_train.json"))
        vm = load_manifest(os.path.join(args.data_dir, "manifest_val.json"))
        vocab = build_vocab(tm + vm)
        idx_to_char = {i: c for c, i in vocab.items() if c != "<blank>"}
        rckpt = torch.load(args.ocr_checkpoint, map_location="cpu")
        recognizer = BaselineRecognizer(vocab_size=len(vocab),
                                        **_infer_recognizer_config(rckpt["model_state_dict"])).to(device)
        recognizer.load_state_dict(rckpt["model_state_dict"])
        prepare_frozen_recognizer(recognizer)
        ocr_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        print(f"recognition-in-the-loop: lambda_ocr={args.lambda_ocr}")

    best_score = float("-inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.stages,
                                     args.lambda_mask, args.grad_clip, recognizer, vocab,
                                     ocr_criterion, args.lambda_ocr)
        scheduler.step()
        psnr, ssim, cer = evaluate(model, val_loader, device, args.stages,
                                   recognizer, idx_to_char)
        history.append({"epoch": epoch, "train_loss": train_loss, "psnr": psnr,
                        "ssim": ssim, "cer": cer})
        cer_str = f" | CER {cer:.4f}" if cer is not None else ""
        print(f"epoch {epoch:>3}/{args.epochs} | loss {train_loss:.4f} | "
              f"PSNR {psnr:.2f} | SSIM {ssim:.4f}{cer_str} | {time.time()-start:.1f}s")
        score = -cer if cer is not None else psnr
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "psnr": psnr,
                        "ssim": ssim, "cer": cer, "iterative": True,
                        "stages": args.stages, "args": vars(args)}, args.checkpoint)
            print(f"    saved -> {args.checkpoint} (best {'CER' if cer is not None else 'PSNR'})")

    if args.history_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.history_out)), exist_ok=True)
        with open(args.history_out, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    print(f"done. best score: {best_score:.4f}")


if __name__ == "__main__":
    main()
