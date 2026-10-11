"""End-to-end evaluation: does restoration help the recogniser read damaged lines?

For every validation sample we decode the text from three versions of the line:

    damaged  : the raw degraded input (the recogniser's "before")
    restored : the U-Net output conditioned on the damage mask
    clean    : the ground truth (the recogniser's upper bound)

and report Character Error Rate for each, plus PSNR/SSIM of the restoration.
The gap between "damaged" and "restored" is the headline number for the
restoration branch.

    python experiments/eval_restore_ocr.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from dataset import build_vocab, load_manifest, WIDTH_DOWNSAMPLE  # noqa: E402
from models.baseline_recognizer import BaselineRecognizer, greedy_decode  # noqa: E402
from models.restoration_unet import RestorationUNet  # noqa: E402
from models.flow_restoration import FlowRestoration, sample as flow_sample  # noqa: E402
from restoration_data import _load_gray  # noqa: E402
from structure_prior import structure_channels  # noqa: E402
from train_baseline import character_error_rate  # noqa: E402


def infer_recognizer_config(state: dict) -> dict:
    """Recover constructor args from a saved state_dict."""
    import re
    layer_ids = [int(m.group(1)) for k in state
                 for m in [re.search(r"lstm\.weight_ih_l(\d+)$", k)] if m]
    lstm_layers = max(layer_ids) + 1 if layer_ids else 1
    hidden = state["lstm.weight_hh_l0"].shape[0] // 4
    proj_size = state["proj.weight"].shape[0]
    feature_height = state["proj.weight"].shape[1] // 256
    return {"hidden_size": hidden, "lstm_layers": lstm_layers,
            "proj_size": proj_size, "img_height": feature_height * 16}


def decode_batch(model, images, widths, idx_to_char, device):
    """Greedy-decode a list of [1, H, W] tensors (variable widths)."""
    max_w = max(w for w in widths)
    padded = []
    for img in images:
        pad = max_w - img.shape[2]
        padded.append(torch.nn.functional.pad(img, (0, pad), value=1.0) if pad else img)
    batch = torch.stack(padded).to(device)
    input_lengths = torch.tensor([max(1, w // WIDTH_DOWNSAMPLE) for w in widths])
    with torch.no_grad():
        log_probs = model(batch)
    return greedy_decode(log_probs.cpu(), idx_to_char)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restoration-checkpoint", default=os.path.join(
        _REPO_ROOT, "checkpoints", "restoration_best.pt"))
    parser.add_argument("--recognizer-checkpoint", default=os.path.join(
        _REPO_ROOT, "checkpoints", "baseline_best.pt"))
    parser.add_argument("--data-dir", default=os.path.join(_REPO_ROOT, "data"))
    parser.add_argument("--img-height", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=20,
                        help="Euler steps for flow-matching checkpoints")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=os.path.join(_REPO_ROOT, "results", "restore_ocr_eval.json"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_manifest = load_manifest(os.path.join(args.data_dir, "manifest_train.json"))
    val_manifest = load_manifest(os.path.join(args.data_dir, "manifest_val.json"))
    # Build the vocab from the *full* manifests -- truncating first would drop
    # rare characters and desync the class indices from the checkpoint.
    vocab = build_vocab(train_manifest + val_manifest)
    idx_to_char = {i: c for c, i in vocab.items() if c != "<blank>"}
    if args.limit:
        val_manifest = val_manifest[:args.limit]

    rec_ckpt = torch.load(args.recognizer_checkpoint, map_location="cpu")
    rec_cfg = infer_recognizer_config(rec_ckpt["model_state_dict"])
    recognizer = BaselineRecognizer(vocab_size=len(vocab), **rec_cfg).to(device)
    recognizer.load_state_dict(rec_ckpt["model_state_dict"])
    recognizer.eval()
    print(f"recognizer config: {rec_cfg}  (checkpoint CER {rec_ckpt.get('cer', float('nan')):.4f})")

    res_ckpt = torch.load(args.restoration_checkpoint, map_location="cpu")
    res_args = res_ckpt.get("args", {})
    is_flow = bool(res_ckpt.get("flow", False) or res_args.get("flow", False))
    if is_flow:
        base = res_args.get("base", 32)
        restoration = FlowRestoration(base=base, emb_dim=res_args.get("emb_dim", 128)).to(device)
        restoration.load_state_dict(res_ckpt["model"])
        restoration.eval()
        structure_source = "none"
        use_structure = False
        print(f"restoration: FLOW  base={base}  epoch={res_ckpt.get('epoch')}  "
              f"PSNR={res_ckpt.get('psnr', float('nan')):.2f}")
    else:
        base = res_args.get("base", 32)
        # New checkpoints store structure_source; old ones stored a bool flag.
        structure_source = res_args.get("structure_source",
                                        "damaged" if res_args.get("use_structure") else "none")
        use_structure = structure_source != "none"
        restoration = RestorationUNet(in_channels=4 if use_structure else 2, base=base).to(device)
        restoration.load_state_dict(res_ckpt["model"])
        restoration.eval()
        print(f"restoration: base={base}  structure={structure_source}  "
              f"epoch={res_ckpt.get('epoch')}  PSNR={res_ckpt.get('psnr', float('nan')):.2f}")

    predictor = None
    if structure_source == "predicted":
        from models.structure_completion import StructureCompletion
        pckpt = torch.load(res_args.get("structure_checkpoint",
                                       os.path.join(_REPO_ROOT, "checkpoints",
                                                    "structure_completion.pt")),
                           map_location="cpu")
        predictor = StructureCompletion(base=pckpt.get("args", {}).get("base", 32)).to(device)
        predictor.load_state_dict(pckpt["model"])
        predictor.eval()
        print(f"structure predictor: skelIoU {pckpt.get('iou', float('nan')):.4f}")

    refs, damaged_preds, restored_preds, clean_preds = [], [], [], []
    damage_fracs = []
    psnr_vals, ssim_vals = [], []
    for item in val_manifest:
        damaged = _load_gray(item["damaged_path"], args.img_height)
        mask = _load_gray(item["mask_path"], args.img_height)
        clean = _load_gray(item["clean_path"], args.img_height)
        w = min(damaged.shape[1], mask.shape[1], clean.shape[1])
        damaged, mask, clean = damaged[:, :w], mask[:, :w], clean[:, :w]
        damage_fracs.append(float((mask > 0.5).mean()))

        # Pad to a multiple of 8 (white for images, zero for the mask) as the
        # training collate did, then crop the restoration back to the real width.
        target_w = ((w + 7) // 8) * 8
        pad = lambda a, v: np.pad(a, ((0, 0), (0, target_w - a.shape[1])),
                                  constant_values=v)
        d_t = torch.from_numpy(pad(damaged, 1.0)).unsqueeze(0).unsqueeze(0).to(device)
        m_t = torch.from_numpy(pad(mask, 0.0)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            if is_flow:
                restored = flow_sample(restoration, d_t, m_t, steps=args.flow_steps,
                                       seed=0)[0, :, :, :w].cpu()
            else:
                channels = [d_t, m_t]
                if use_structure:
                    if structure_source == "predicted":
                        st = predictor(torch.cat([d_t, m_t], dim=1))
                    elif structure_source == "clean":
                        st = torch.from_numpy(structure_channels(pad(clean, 1.0))).unsqueeze(0).to(device)
                    else:  # "damaged"
                        st = torch.from_numpy(structure_channels(pad(damaged, 1.0))).unsqueeze(0).to(device)
                    channels.append(st)
                restored = restoration(torch.cat(channels, dim=1))[0, :, :, :w].cpu()
        restored_np = restored[0].numpy()

        widths = [w]
        damaged_preds.extend(decode_batch(recognizer, [torch.from_numpy(damaged).unsqueeze(0)],
                                          widths, idx_to_char, device))
        restored_preds.extend(decode_batch(recognizer, [restored], widths, idx_to_char, device))
        clean_preds.extend(decode_batch(recognizer, [torch.from_numpy(clean).unsqueeze(0)],
                                        widths, idx_to_char, device))
        refs.append(item["text"])
        psnr_vals.append(peak_signal_noise_ratio(clean, restored_np, data_range=1.0))
        ssim_vals.append(structural_similarity(clean, restored_np, data_range=1.0))

    # CER broken down by damage severity: does structure help where it should?
    med = float(np.median(damage_fracs))
    buckets = {}
    for name, lo, hi in [("low_damage", 0.0, med), ("high_damage", med, 1.0)]:
        idx = [i for i, f in enumerate(damage_fracs) if lo <= f < hi or (hi == 1.0 and f >= hi)]
        if not idx:
            continue
        r = [refs[i] for i in idx]
        buckets[name] = {
            "n": len(idx),
            "cer_damaged": character_error_rate([damaged_preds[i] for i in idx], r),
            "cer_restored": character_error_rate([restored_preds[i] for i in idx], r),
            "cer_clean": character_error_rate([clean_preds[i] for i in idx], r),
        }

    result = {
        "n": len(refs),
        "cer_damaged": character_error_rate(damaged_preds, refs),
        "cer_restored": character_error_rate(restored_preds, refs),
        "cer_clean": character_error_rate(clean_preds, refs),
        "restoration_psnr": float(np.mean(psnr_vals)),
        "restoration_ssim": float(np.mean(ssim_vals)),
        "damage_buckets": buckets,
        "samples": [
            {"ref": r, "damaged": d, "restored": s, "clean": c}
            for r, d, s, c in zip(refs[:20], damaged_preds[:20], restored_preds[:20], clean_preds[:20])
        ],
    }
    print(f"\nval CER  damaged : {result['cer_damaged']:.4f}")
    print(f"val CER  restored: {result['cer_restored']:.4f}")
    print(f"val CER  clean   : {result['cer_clean']:.4f}  (recogniser upper bound)")
    print(f"restoration PSNR {result['restoration_psnr']:.2f} | SSIM {result['restoration_ssim']:.4f}")
    for name, b in buckets.items():
        print(f"  [{name:>11} n={b['n']:>3}] damaged {b['cer_damaged']:.4f} -> "
              f"restored {b['cer_restored']:.4f}  (clean {b['cer_clean']:.4f})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
