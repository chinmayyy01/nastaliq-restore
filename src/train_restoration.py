"""Train the U-Net restoration model on damaged Nastaliq lines.

Supervision is the synthetic ground truth: the model sees the damaged line
plus the damage mask and is asked to reconstruct the clean line. The loss is
an L1 reconstruction term plus a mask-weighted L1 term that pushes the model
to fix exactly the corrupted regions.

With ``--lambda-ocr > 0`` a frozen recogniser is put in the loop: a CTC loss
on the *restored* image is added so restoration is optimised for readability
(recognition-in-the-loop), not pixel fidelity alone.

Validation reports L1, PSNR and SSIM against the clean line, saving the best
checkpoint.

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

from dataset import build_vocab, load_manifest, WIDTH_DOWNSAMPLE
from restoration_data import RestorationDataset, restoration_collate
from models.restoration_unet import RestorationUNet
from models.baseline_recognizer import BaselineRecognizer, greedy_decode
from train_baseline import character_error_rate


def masked_l1(pred, target, mask, eps: float = 1e-6):
    """L1 restricted to the damaged region (mask-weighted mean)."""
    diff = (pred - target).abs() * mask
    return diff.sum() / (mask.sum() + eps)


def _infer_recognizer_config(state: dict) -> dict:
    """Recover BaselineRecognizer constructor args from a saved state_dict."""
    import re
    layer_ids = [int(m.group(1)) for k in state
                 for m in [re.search(r"lstm\.weight_ih_l(\d+)$", k)] if m]
    return {
        "hidden_size": state["lstm.weight_hh_l0"].shape[0] // 4,
        "lstm_layers": max(layer_ids) + 1 if layer_ids else 1,
        "proj_size": state["proj.weight"].shape[0],
        "img_height": (state["proj.weight"].shape[1] // 256) * 16,
    }


def prepare_frozen_recognizer(recognizer):
    """Freeze a recogniser for recognition-in-the-loop.

    cuDNN refuses to back-propagate through an RNN in eval mode, so the LSTM
    must stay in train mode; but train-mode BatchNorm would corrupt the frozen
    running stats, and dropout would add noise. So: LSTM in train (dropout off),
    everything else in eval, all parameters frozen.
    """
    recognizer.train()
    recognizer.lstm.dropout = 0.0
    recognizer.dropout.eval()
    for m in recognizer.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
    for p in recognizer.parameters():
        p.requires_grad_(False)
    return recognizer


def build_input(damaged, mask, structure, use_structure: bool):
    """Assemble the network input; add the structure prior only when enabled."""
    if use_structure:
        return torch.cat([damaged, mask, structure], dim=1)
    return torch.cat([damaged, mask], dim=1)


def resolve_structure(structure, damaged, mask, predictor):
    """Use the frozen predictor when given, else the dataset-provided tensor."""
    if predictor is not None:
        with torch.no_grad():
            return predictor(torch.cat([damaged, mask], dim=1))
    return structure


def ctc_loss_on_restored(pred, texts, widths, recognizer, vocab, criterion, device):
    """CTC loss of the frozen recogniser reading the restored image."""
    labels = [torch.tensor([vocab[c] for c in t], dtype=torch.long) for t in texts]
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long, device=device)
    flat = torch.cat(labels).to(device)
    input_lengths = torch.tensor([max(1, w // WIDTH_DOWNSAMPLE) for w in widths],
                                 dtype=torch.long, device=device)
    log_probs = recognizer(pred)
    return criterion(log_probs, flat, input_lengths, label_lengths)


def train_one_epoch(model, loader, optimizer, device, lambda_mask=1.0,
                    grad_clip=5.0, use_structure=False, predictor=None,
                    recognizer=None, vocab=None, ocr_criterion=None, lambda_ocr=0.0):
    model.train()
    total, count = 0.0, 0
    l1 = nn.L1Loss()
    for damaged, mask, clean, structure, widths, texts in loader:
        damaged, mask, clean = damaged.to(device), mask.to(device), clean.to(device)
        structure = resolve_structure(structure.to(device), damaged, mask, predictor)
        pred = model(build_input(damaged, mask, structure, use_structure))
        loss = l1(pred, clean) + lambda_mask * masked_l1(pred, clean, mask)
        if recognizer is not None and lambda_ocr > 0:
            loss = loss + lambda_ocr * ctc_loss_on_restored(
                pred, texts, widths, recognizer, vocab, ocr_criterion, device)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * damaged.shape[0]
        count += damaged.shape[0]
    return total / count if count else 0.0


@torch.no_grad()
def evaluate(model, loader, device, lambda_mask=1.0, use_structure=False,
             predictor=None, recognizer=None, idx_to_char=None):
    model.eval()
    l1 = nn.L1Loss()
    total, count = 0.0, 0
    psnr_vals, ssim_vals = [], []
    preds, refs = [], []
    for damaged, mask, clean, structure, widths, texts in loader:
        damaged, mask, clean = damaged.to(device), mask.to(device), clean.to(device)
        structure = resolve_structure(structure.to(device), damaged, mask, predictor)
        pred = model(build_input(damaged, mask, structure, use_structure))
        loss = l1(pred, clean) + lambda_mask * masked_l1(pred, clean, mask)
        total += loss.item() * damaged.shape[0]
        count += damaged.shape[0]

        if recognizer is not None and idx_to_char is not None:
            log_probs = recognizer(pred)
            preds.extend(greedy_decode(log_probs.cpu(), idx_to_char))
            refs.extend(texts)

        pred_np = pred.cpu().numpy()[:, 0]
        clean_np = clean.cpu().numpy()[:, 0]
        for p, c in zip(pred_np, clean_np):
            psnr_vals.append(peak_signal_noise_ratio(c, p, data_range=1.0))
            ssim_vals.append(structural_similarity(c, p, data_range=1.0))
    cer = character_error_rate(preds, refs) if refs else None
    return (total / count if count else 0.0,
            float(np.mean(psnr_vals)), float(np.mean(ssim_vals)), cer)


def main():
    parser = argparse.ArgumentParser(description="Train the restoration U-Net")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base", type=int, default=32, help="U-Net base channels")
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--structure-source", default="none",
                        choices=["none", "damaged", "clean", "predicted"],
                        help="Stroke-structure conditioning: observed ('damaged'), "
                             "supervision ('clean'), or predicted by a frozen "
                             "structure-completion net ('predicted')")
    parser.add_argument("--structure-checkpoint", default=os.path.join(
        _REPO_ROOT, "checkpoints", "structure_completion.pt"),
        help="Frozen structure-completion net used when --structure-source=predicted")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ocr-checkpoint", default=os.path.join(
        _REPO_ROOT, "checkpoints", "baseline_best.pt"),
        help="Frozen recogniser used for the recognition-in-the-loop loss")
    parser.add_argument("--lambda-ocr", type=float, default=0.0,
                        help="Weight of the CTC loss on the restored image "
                             "(0 disables recognition-in-the-loop)")
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

    # The dataset only needs to provide structure for the observed/supervision
    # modes; for "predicted" it is generated on the fly by the frozen predictor.
    ds_source = args.structure_source if args.structure_source in ("damaged", "clean") else "none"
    train_ds = RestorationDataset(
        os.path.join(args.data_dir, "manifest_train.json"), args.img_height,
        structure_source=ds_source)
    val_ds = RestorationDataset(
        os.path.join(args.data_dir, "manifest_val.json"), args.img_height,
        structure_source=ds_source)
    if args.limit_train:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))
    print(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=restoration_collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=restoration_collate, num_workers=args.num_workers)

    use_structure = args.structure_source != "none"
    in_channels = 4 if use_structure else 2
    model = RestorationUNet(in_channels=in_channels, base=args.base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, args.epochs * len(train_loader)),
        pct_start=0.1)

    predictor = None
    if args.structure_source == "predicted":
        from models.structure_completion import StructureCompletion
        pckpt = torch.load(args.structure_checkpoint, map_location="cpu")
        predictor = StructureCompletion(base=pckpt.get("args", {}).get("base", 32)).to(device)
        predictor.load_state_dict(pckpt["model"])
        predictor.eval()
        for p in predictor.parameters():
            p.requires_grad_(False)
        print(f"structure predictor: {args.structure_checkpoint} "
              f"(epoch {pckpt.get('epoch')}, L1 {pckpt.get('l1', float('nan')):.4f})")

    recognizer, vocab, idx_to_char, ocr_criterion = None, None, None, None
    if args.lambda_ocr > 0:
        train_manifest = load_manifest(os.path.join(args.data_dir, "manifest_train.json"))
        val_manifest = load_manifest(os.path.join(args.data_dir, "manifest_val.json"))
        vocab = build_vocab(train_manifest + val_manifest)
        idx_to_char = {i: c for c, i in vocab.items() if c != "<blank>"}
        rckpt = torch.load(args.ocr_checkpoint, map_location="cpu")
        rec_cfg = _infer_recognizer_config(rckpt["model_state_dict"])
        recognizer = BaselineRecognizer(vocab_size=len(vocab), **rec_cfg).to(device)
        recognizer.load_state_dict(rckpt["model_state_dict"])
        prepare_frozen_recognizer(recognizer)
        ocr_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        print(f"recogniser-in-the-loop: {args.ocr_checkpoint} (CER {rckpt.get('cer', float('nan')):.4f}), "
              f"lambda_ocr={args.lambda_ocr}")

    best_score = float("-inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     args.lambda_mask, args.grad_clip,
                                     use_structure, predictor,
                                     recognizer, vocab, ocr_criterion, args.lambda_ocr)
        scheduler.step()
        val_loss, psnr, ssim, cer = evaluate(
            model, val_loader, device, args.lambda_mask, use_structure, predictor,
            recognizer, idx_to_char)
        elapsed = time.time() - start
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "psnr": psnr, "ssim": ssim, "cer": cer})
        cer_str = f" | CER {cer:.4f}" if cer is not None else ""
        print(f"epoch {epoch:>3}/{args.epochs} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | PSNR {psnr:.2f} | SSIM {ssim:.4f}{cer_str} | {elapsed:.1f}s")
        # Select on downstream CER when recognition is in the loop, else PSNR.
        score = -cer if cer is not None else psnr
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "psnr": psnr, "ssim": ssim, "cer": cer, "args": vars(args)},
                       args.checkpoint)
            print(f"    saved -> {args.checkpoint} (best {'CER' if cer is not None else 'PSNR'})")

    if args.history_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.history_out)), exist_ok=True)
        with open(args.history_out, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"history -> {args.history_out}")
    print(f"done. best score: {best_score:.4f}")


if __name__ == "__main__":
    main()
