"""Build the synthetic OCR dataset for Nastaliq restoration.

For every line in ``resources/urdu_sentences.txt`` we:

1. render a clean image with :func:`text_renderer.render_line`,
2. synthesise a damaged copy + damage mask with
   :func:`damage_simulator.simulate_damage`,
3. write the three PNGs under ``data/{clean,damaged,damage_masks}/`` and
   record the sample in ``data/manifest.json``.

The manifest is then split 80/20 (shuffled with a fixed seed) into
``data/manifest_train.json`` and ``data/manifest_val.json`` so that every
run is reproducible.

Paths stored in the manifests are relative to the repository root, which
keeps them portable between a local checkout and Colab.
"""

import argparse
import json
import os
import random
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from text_renderer import render_line
from damage_simulator import simulate_damage

_REPO_ROOT = os.path.dirname(_SRC_DIR)

SENTENCES_PATH = os.path.join(_REPO_ROOT, "resources", "urdu_sentences.txt")
DATA_DIR = os.path.join(_REPO_ROOT, "data")
SPLIT_SEED = 42
VAL_FRACTION = 0.2


def read_sentences(path: str = SENTENCES_PATH) -> list:
    """Read one non-empty sentence per line, preserving file order."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _rel(path: str) -> str:
    """Return ``path`` relative to the repo root (POSIX separators)."""
    return os.path.relpath(path, _REPO_ROOT).replace(os.sep, "/")


def build(sentences_path: str = SENTENCES_PATH, data_dir: str = DATA_DIR,
          font_size: int = 64, limit: int = None,
          damage_severity: float = 1.0) -> list:
    """Render + damage every sentence and write images + manifests.

    Returns the full list of manifest entries.
    """
    sentences = read_sentences(sentences_path)
    if limit is not None:
        sentences = sentences[:limit]

    clean_dir = os.path.join(data_dir, "clean")
    damaged_dir = os.path.join(data_dir, "damaged")
    mask_dir = os.path.join(data_dir, "damage_masks")
    for d in (clean_dir, damaged_dir, mask_dir):
        os.makedirs(d, exist_ok=True)

    manifest = []
    for i, text in enumerate(sentences):
        clean = render_line(text, font_size=font_size)
        damaged, mask = simulate_damage(clean, seed=i, severity=damage_severity)

        name = f"sample_{i:04d}.png"
        clean_path = os.path.join(clean_dir, name)
        damaged_path = os.path.join(damaged_dir, name)
        mask_path = os.path.join(mask_dir, name)

        clean.save(clean_path)
        damaged.save(damaged_path)
        mask.save(mask_path)

        manifest.append({
            "id": i,
            "text": text,
            "clean_path": _rel(clean_path),
            "damaged_path": _rel(damaged_path),
            "mask_path": _rel(mask_path),
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(sentences):
            print(f"  built {i + 1}/{len(sentences)} samples")

    _write_json(os.path.join(data_dir, "manifest.json"), manifest)
    train, val = split(manifest)
    _write_json(os.path.join(data_dir, "manifest_train.json"), train)
    _write_json(os.path.join(data_dir, "manifest_val.json"), val)
    print(f"manifest: {len(manifest)} samples "
          f"({len(train)} train / {len(val)} val)")
    return manifest


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def split(manifest: list, val_fraction: float = VAL_FRACTION,
          seed: int = SPLIT_SEED):
    """Shuffle with a fixed seed and split into (train, val) lists."""
    shuffled = list(manifest)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if shuffled else 0
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Build the Nastaliq OCR dataset")
    parser.add_argument("--sentences", default=SENTENCES_PATH,
                        help="Path to the one-sentence-per-line corpus")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="Output data directory")
    parser.add_argument("--font-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only build the first N samples (debugging)")
    parser.add_argument("--damage-severity", type=float, default=1.0,
                        help="Damage multiplier: 1.0 mild, ~2.0 hard")
    args = parser.parse_args()
    build(args.sentences, args.data_dir, args.font_size, args.limit,
          args.damage_severity)


if __name__ == "__main__":
    main()
