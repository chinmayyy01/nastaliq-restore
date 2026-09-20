"""Train the CNN + BiLSTM + CTC baseline recogniser.

Runs on CPU (small scale, for smoke-testing) or GPU (Colab) automatically.
After every epoch it validates with greedy CTC decoding and Character Error
Rate (Levenshtein distance / total reference characters), saving the best
checkpoint (lowest val CER) to ``checkpoints/baseline_best.pt``.

Example smoke test:
    python src/train_baseline.py --epochs 3 --smoke
Full run (GPU):
    python src/train_baseline.py --epochs 30
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dataset import build_vocab, load_manifest, OCRDataset, collate_fn
from models.baseline_recognizer import BaselineRecognizer, greedy_decode
from utils.checkpoint import save_checkpoint


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (insert/delete/substitute)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


def character_error_rate(preds: list, refs: list) -> float:
    """CER = sum(edit distance) / sum(reference length)."""
    total_distance = sum(levenshtein(p, r) for p, r in zip(preds, refs))
    total_length = sum(len(r) for r in refs)
    return total_distance / total_length if total_length else 0.0


def _idx_to_char(vocab: dict) -> dict:
    return {idx: ch for ch, idx in vocab.items() if ch != "<blank>"}


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels, label_lengths, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        log_probs = model(images)                 # [T, N, C]
        t, n, _ = log_probs.shape
        input_lengths = torch.full((n,), t, dtype=torch.long, device=device)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * n
        total_items += n
    return total_loss / total_items if total_items else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device, idx_to_char):
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_preds = []
    all_refs = []
    for images, labels, label_lengths, texts in loader:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        log_probs = model(images)
        t, n, _ = log_probs.shape
        input_lengths = torch.full((n,), t, dtype=torch.long, device=device)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        total_loss += loss.item() * n
        total_items += n

        all_preds.extend(greedy_decode(log_probs.cpu(), idx_to_char))
        all_refs.extend(texts)

    avg_loss = total_loss / total_items if total_items else 0.0
    cer = character_error_rate(all_preds, all_refs)
    return avg_loss, cer, all_preds, all_refs


def main():
    parser = argparse.ArgumentParser(description="Train the baseline OCR model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--data-dir", default=os.path.join(_REPO_ROOT, "data"))
    parser.add_argument("--checkpoint", default=os.path.join(
        _REPO_ROOT, "checkpoints", "baseline_best.pt"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="cpu / cuda / mps (default: auto-detect)")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="Use only the first N training samples")
    parser.add_argument("--limit-val", type=int, default=None,
                        help="Use only the first N validation samples")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny CPU run (32 train / 16 val samples)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    limit_train = 32 if args.smoke else args.limit_train
    limit_val = 16 if args.smoke else args.limit_val

    train_manifest = load_manifest(os.path.join(args.data_dir, "manifest_train.json"))
    val_manifest = load_manifest(os.path.join(args.data_dir, "manifest_val.json"))
    vocab = build_vocab(train_manifest + val_manifest)
    idx_to_char = _idx_to_char(vocab)
    print(f"vocab size (incl. blank): {len(vocab)}  "
          f"train={len(train_manifest)} val={len(val_manifest)}")

    train_ds = OCRDataset(os.path.join(args.data_dir, "manifest_train.json"), vocab)
    val_ds = OCRDataset(os.path.join(args.data_dir, "manifest_val.json"), vocab)
    if limit_train:
        train_ds = Subset(train_ds, range(min(limit_train, len(train_ds))))
    if limit_val:
        val_ds = Subset(val_ds, range(min(limit_val, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))

    model = BaselineRecognizer(vocab_size=len(vocab),
                               hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_cer = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_cer, preds, refs = evaluate(
            model, val_loader, criterion, device, idx_to_char)
        elapsed = time.time() - start

        print(f"epoch {epoch:>3}/{args.epochs} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_CER {val_cer:.4f} | {elapsed:.1f}s")
        print(f"    ref : {refs[0]}")
        print(f"    pred: {preds[0]}")

        if val_cer < best_cer:
            best_cer = val_cer
            save_checkpoint(model, optimizer, epoch, val_cer, args.checkpoint)
            print(f"    saved checkpoint -> {args.checkpoint} (best CER {best_cer:.4f})")

    print(f"done. best val CER: {best_cer:.4f}")


if __name__ == "__main__":
    main()
