import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

def _to_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.uint8)

def add_ink_loss(arr: np.ndarray, mask: np.ndarray, n_blobs=(3, 8), radius=(3, 9)):
    """Small circular patches erased to background (255 = white/blank)."""
    h, w = arr.shape
    n = random.randint(*n_blobs)
    for _ in range(n):
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        r = random.randint(*radius)
        yy, xx = np.ogrid[:h, :w]
        blob = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        arr[blob] = 255
        mask[blob] = 255
    return arr, mask


def add_tear(arr: np.ndarray, mask: np.ndarray, n_tears=(1, 2), size_frac=(0.08, 0.22)):
    """Larger irregular missing regions simulating physical tears."""
    h, w = arr.shape
    n = random.randint(*n_tears)
    for _ in range(n):
        tw = int(w * random.uniform(*size_frac))
        th = int(h * random.uniform(0.5, 1.0))
        x0 = random.randint(0, max(1, w - tw))
        y0 = random.randint(0, max(1, h - th))
        pts = []
        steps = 6
        for i in range(steps + 1):
            jitter = random.randint(-tw // 6, tw // 6)
            pts.append((x0 + jitter, y0 + int(i * th / steps)))
        for i in range(steps + 1):
            jitter = random.randint(-tw // 6, tw // 6)
            pts.append((x0 + tw + jitter, y0 + int((steps - i) * th / steps)))
        poly_mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(poly_mask).polygon(pts, fill=255)
        poly_np = np.array(poly_mask) > 0
        arr[poly_np] = 255
        mask[poly_np] = 255
    return arr, mask


def add_stain(arr: np.ndarray, mask: np.ndarray, n_stains=(1, 3), radius=(15, 35), strength=(0.3, 0.6)):
    """Blotches that partially darken/obscure ink rather than erasing fully
    (simulates water stains, ink bleed) -- lower contrast, not full erasure."""
    h, w = arr.shape
    n = random.randint(*n_stains)
    farr = arr.astype(np.float32)
    for _ in range(n):
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        r = random.randint(*radius)
        strength_val = random.uniform(*strength)
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        blob = dist <= r
        falloff = np.clip(1 - dist / r, 0, 1) * strength_val
        farr[blob] = farr[blob] * (1 - falloff[blob]) + 190 * falloff[blob]
        mask[blob] = np.maximum(mask[blob], (falloff[blob] * 255).astype(np.uint8))
    return farr.clip(0, 255).astype(np.uint8), mask


def add_fading(arr: np.ndarray, mask: np.ndarray, band_frac=(0.15, 0.35), fade_strength=(0.4, 0.7)):
    """A gradual low-contrast fading band, simulating ink that has faded
    over time rather than being physically removed."""
    h, w = arr.shape
    bw = int(w * random.uniform(*band_frac))
    x0 = random.randint(0, max(1, w - bw))
    strength = random.uniform(*fade_strength)
    farr = arr.astype(np.float32)
    farr[:, x0:x0 + bw] = farr[:, x0:x0 + bw] * (1 - strength) + 255 * strength
    band_mask = np.zeros((h, w), dtype=np.uint8)
    band_mask[:, x0:x0 + bw] = int(strength * 255)
    mask = np.maximum(mask, band_mask)
    return farr.clip(0, 255).astype(np.uint8), mask


def simulate_damage(img: Image.Image, seed: int = None) -> tuple[Image.Image, Image.Image]:
    """Apply a random combination of damage types to one clean text-line image.

    Returns:
        damaged_img: the damaged grayscale PIL image
        damage_mask: a PIL image (mode L) where brighter = more damaged;
                     this is our training supervision signal for "where to focus"
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    arr = _to_array(img).copy()
    mask = np.zeros_like(arr)

    damage_fns = [add_ink_loss, add_tear, add_stain, add_fading]
    chosen = random.sample(damage_fns, k=random.randint(1, 3))
    for fn in chosen:
        arr, mask = fn(arr, mask)

    damaged_img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.4))
    damage_mask = Image.fromarray(mask)
    return damaged_img, damage_mask


if __name__ == "__main__":
    import os
    os.makedirs("data/damaged", exist_ok=True)
    os.makedirs("data/damage_masks", exist_ok=True)

    clean_dir = "data/clean"
    for fname in sorted(os.listdir(clean_dir)):
        if not fname.endswith(".png"):
            continue
        img = Image.open(os.path.join(clean_dir, fname))
        damaged, mask = simulate_damage(img, seed=hash(fname) % 10000)
        damaged.save(os.path.join("data/damaged", fname))
        mask.save(os.path.join("data/damage_masks", fname))
        print(f"Damaged: {fname}")