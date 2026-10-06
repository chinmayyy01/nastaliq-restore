"""Dataset + vocabulary + collation utilities for CTC-based OCR.

The recogniser is trained on the *clean* renderings: the damaged images are
reserved for the restoration branch that comes later, so this dataset simply
teaches the model "given a clean Nastaliq line, what does it say?".
"""

import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

BLANK_INDEX = 0
IMG_HEIGHT = 64


def build_vocab(manifest: list) -> dict:
    """Build a character -> integer mapping from all ``text`` fields.

    Index 0 is reserved for the CTC blank token; real characters start at 1.
    The mapping is sorted by codepoint so it is deterministic across runs.
    """
    chars = sorted({ch for item in manifest for ch in item["text"]})
    vocab = {"<blank>": BLANK_INDEX}
    for i, ch in enumerate(chars, start=1):
        vocab[ch] = i
    return vocab


def load_manifest(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(path: str) -> str:
    """Resolve a manifest path relative to the repo root when needed."""
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def augment_line(img: Image.Image) -> Image.Image:
    """Mild photometric + geometric jitter to fight overfitting.

    The synthetic renderings are all the same font at the same size, so the
    recogniser memorises the exact pixel patterns. Random rotation, width
    scaling, contrast and noise force it to key on the actual strokes.
    """
    angle = random.uniform(-1.5, 1.5)
    if abs(angle) > 0.1:
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False,
                         fillcolor=255)

    scale = random.uniform(0.92, 1.08)
    new_w = max(4, int(round(img.width * scale)))
    img = img.resize((new_w, img.height), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32)
    arr = arr * random.uniform(0.85, 1.15) + random.uniform(-12, 12)
    if random.random() < 0.5:
        arr = arr + np.random.normal(0, random.uniform(2, 8), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class OCRDataset(Dataset):
    """Loads (clean image, text) pairs from a manifest file.

    Images are loaded in grayscale, resized to a fixed height of
    ``img_height`` while preserving aspect ratio, and scaled to ``[0, 1]``
    (white background = 1.0). Labels are mapped through ``vocab``.
    """

    def __init__(self, manifest_path: str, vocab: dict,
                 img_height: int = IMG_HEIGHT, augment: bool = False):
        self.manifest = load_manifest(manifest_path)
        self.vocab = vocab
        self.img_height = img_height
        self.augment = augment

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        item = self.manifest[idx]

        img = Image.open(_resolve(item["clean_path"])).convert("L")
        w, h = img.size
        new_w = max(1, int(round(w * self.img_height / h)))
        img = img.resize((new_w, self.img_height), Image.BILINEAR)
        if self.augment:
            img = augment_line(img)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        image = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

        text = item["text"]
        label = torch.tensor([self.vocab[c] for c in text], dtype=torch.long)
        return image, label, text


# Each ConvBlock pools the width by 2; the first three blocks pool the width,
# the last keeps it, so the sequence axis is downsampled by 8 overall.
WIDTH_DOWNSAMPLE = 8


def collate_fn(batch):
    """Pad variable-width images to the widest in the batch.

    Returns the tuple expected by ``torch.nn.CTCLoss``:
        padded_images      : [N, 1, H, W_max]
        flat_labels        : 1-D concatenation of all label tensors
        label_lengths      : [N] target lengths
        input_lengths      : [N] valid timesteps per sample (excludes padding)
        raw_texts          : list[str] ground-truth strings
    """
    images, labels, texts = zip(*batch)

    widths = [img.shape[2] for img in images]
    max_w = max(widths)
    padded = []
    for img in images:
        pad = max_w - img.shape[2]
        if pad > 0:
            # Pad with white (1.0) so background matches the rendered lines.
            img = torch.nn.functional.pad(img, (0, pad), value=1.0)
        padded.append(img)
    images = torch.stack(padded, dim=0)

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    input_lengths = torch.tensor(
        [max(1, w // WIDTH_DOWNSAMPLE) for w in widths], dtype=torch.long)
    flat_labels = (torch.cat(labels) if labels
                   else torch.empty(0, dtype=torch.long))
    return images, flat_labels, label_lengths, input_lengths, list(texts)


if __name__ == "__main__":
    manifest_path = os.path.join(_REPO_ROOT, "data", "manifest_train.json")
    manifest = load_manifest(manifest_path)
    vocab = build_vocab(manifest)
    print(f"vocab size (incl. blank): {len(vocab)}")

    ds = OCRDataset(manifest_path, vocab)
    print(f"dataset length: {len(ds)}")

    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    images, labels, label_lengths, input_lengths, texts = next(iter(loader))
    print(f"images:        {tuple(images.shape)}  (N, C, H, W)")
    print(f"flat_labels:   {tuple(labels.shape)}  sum(len)={int(label_lengths.sum())}")
    print(f"label_lengths: {label_lengths.tolist()}")
    print(f"input_lengths: {input_lengths.tolist()}")
    print(f"raw_texts:     {texts}")
    assert images.dim() == 4 and images.shape[1] == 1
    assert images.shape[2] == IMG_HEIGHT
    assert labels.numel() == int(label_lengths.sum())
    assert len(texts) == images.shape[0]
    assert len(input_lengths) == images.shape[0]
    print("dataset sanity check OK")
