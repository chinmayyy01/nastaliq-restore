"""Stroke-structure prior for the restoration branch.

The recogniser and the plain U-Net both see only pixels. Nastaliq ligatures
are defined by their stroke topology -- which strokes join, where the dots
attach -- and a tear can destroy exactly that topology while leaving the
surrounding pixels intact. So we hand the restoration network an explicit
structural view of the *damaged* line:

    channel 0 : skeleton        -- 1px medial axis of the surviving ink
    channel 1 : stroke distance -- distance-to-background, i.e. stroke width

These are cheap, deterministic and computed from the damaged image alone
(available at inference). They are the rasterised stand-in for the stroke
graph in ``structure_graph.py``; a GNN over that graph can replace them later.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

N_CHANNELS = 2


def structure_channels(gray: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return ``[2, H, W]`` structural channels for a grayscale image in [0,1]."""
    ink = gray < threshold
    skeleton = skeletonize(ink).astype(np.float32)
    distance = distance_transform_edt(ink).astype(np.float32)
    distance = distance / (distance.max() + 1e-6)
    return np.stack([skeleton, distance], axis=0)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from restoration_data import _load_gray, _REPO_ROOT

    item_path = os.path.join(_REPO_ROOT, "data", "manifest_train.json")
    import json
    with open(item_path, encoding="utf-8") as f:
        item = json.load(f)[0]

    gray = _load_gray(item["damaged_path"], 64)
    ch = structure_channels(gray)
    print(f"damaged {gray.shape} -> structure {ch.shape}")
    print(f"  skeleton ink fraction: {float(ch[0].mean()):.4f}")
    print(f"  distance max (norm)  : {float(ch[1].max()):.3f}")
    assert ch.shape == (N_CHANNELS,) + gray.shape
    assert 0.0 <= ch.min() and ch.max() <= 1.0
    print("structure prior sanity check OK")
