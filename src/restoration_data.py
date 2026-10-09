"""Dataset + collation for the restoration branch.

Each manifest entry pairs three images:

    damaged_path : the degraded input the model sees
    mask_path    : where the damage was applied (supervision / conditioning)
    clean_path   : the ground-truth restoration target

All three are resized to a common height (preserving aspect ratio) and
returned as float tensors in ``[0, 1]`` with white = 1.0. Widths are padded to
a multiple of 8 in the collate step so the U-Net's three downsamples stay
exact.
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from structure_prior import structure_channels, N_CHANNELS

IMG_HEIGHT = 64
SIZE_MULTIPLE = 8


def load_manifest(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def _load_gray(path: str, img_height: int) -> np.ndarray:
    img = Image.open(_resolve(path)).convert("L")
    w, h = img.size
    new_w = max(1, int(round(w * img_height / h)))
    img = img.resize((new_w, img_height), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


class RestorationDataset(Dataset):
    """Loads (damaged, mask, clean) triples from a manifest file.

    ``structure_source`` controls the stroke-structure tensor (see
    ``structure_prior``):

        "none"    -> zeros (structure channel unused)
        "damaged" -> structure observed in the damaged image (the naive prior)
        "clean"   -> structure of the clean target (supervision for the
                     structure-completion predictor)

    The tensor is always returned (zeros when unused) so the tuple layout is
    stable.
    """

    def __init__(self, manifest_path: str, img_height: int = IMG_HEIGHT,
                 structure_source: str = "none"):
        assert structure_source in ("none", "damaged", "clean")
        self.manifest = load_manifest(manifest_path)
        self.img_height = img_height
        self.structure_source = structure_source

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        item = self.manifest[idx]
        damaged = _load_gray(item["damaged_path"], self.img_height)
        mask = _load_gray(item["mask_path"], self.img_height)
        clean = _load_gray(item["clean_path"], self.img_height)

        # All three were resized from the same source size, so widths agree.
        w = min(damaged.shape[1], mask.shape[1], clean.shape[1])
        damaged, mask, clean = damaged[:, :w], mask[:, :w], clean[:, :w]

        if self.structure_source == "damaged":
            structure = structure_channels(damaged)
        elif self.structure_source == "clean":
            structure = structure_channels(clean)
        else:
            structure = np.zeros((N_CHANNELS,) + damaged.shape, dtype=np.float32)

        to_tensor = lambda a: torch.from_numpy(a)  # [C, H, W]
        return (to_tensor(damaged).unsqueeze(0), to_tensor(mask).unsqueeze(0),
                to_tensor(clean).unsqueeze(0), to_tensor(structure), item["text"])


def _pad_to_multiple(t: torch.Tensor, target_w: int, value: float) -> torch.Tensor:
    pad = target_w - t.shape[2]
    if pad > 0:
        t = torch.nn.functional.pad(t, (0, pad), value=value)
    return t


def restoration_collate(batch):
    """Pad a batch to a common width (multiple of 8).

    Returns:
        damaged   : [N, 1, H, W]
        mask      : [N, 1, H, W]
        clean     : [N, 1, H, W]
        structure : [N, 2, H, W]  (zeros unless the dataset computed it)
        widths    : list[int] true (unpadded) widths
        texts     : list[str] ground-truth strings
    """
    damaged, masks, cleans, structures, texts = zip(*batch)
    max_w = max(t.shape[2] for t in damaged)
    target_w = ((max_w + SIZE_MULTIPLE - 1) // SIZE_MULTIPLE) * SIZE_MULTIPLE

    # Pad damaged/clean with white (1.0), mask and structure with zero so the
    # network sees a consistent background and no spurious damage signal.
    damaged = torch.stack([_pad_to_multiple(t, target_w, 1.0) for t in damaged])
    masks = torch.stack([_pad_to_multiple(t, target_w, 0.0) for t in masks])
    cleans = torch.stack([_pad_to_multiple(t, target_w, 1.0) for t in cleans])
    structures = torch.stack([_pad_to_multiple(t, target_w, 0.0) for t in structures])
    widths = [t.shape[2] for t in damaged]
    return damaged, masks, cleans, structures, widths, list(texts)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    manifest_path = os.path.join(_REPO_ROOT, "data", "manifest_train.json")
    ds = RestorationDataset(manifest_path, structure_source="damaged")
    print(f"dataset length: {len(ds)}")

    loader = DataLoader(ds, batch_size=4, collate_fn=restoration_collate)
    damaged, mask, clean, structure, widths, texts = next(iter(loader))
    print(f"damaged {tuple(damaged.shape)}  mask {tuple(mask.shape)}  "
          f"clean {tuple(clean.shape)}  structure {tuple(structure.shape)}")
    print(f"widths:  {widths}")
    print(f"texts:   {texts}")
    assert damaged.shape == mask.shape == clean.shape
    assert structure.shape[1] == N_CHANNELS
    assert damaged.shape[3] % SIZE_MULTIPLE == 0
    print("restoration dataset sanity check OK")
