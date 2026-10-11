"""Flow-matching restoration backbone (novelty #3).

Instead of a deterministic U-Net that regresses the clean image directly, we
learn a **conditional flow** (Lipman et al., flow matching) that transports a
noise sample to a clean line, conditioned on the damaged line and damage mask.

Conditional flow matching (CFM) training:
    x0 ~ N(0, I)           (noise)
    x1 = clean             (data)
    t  ~ U[0, 1]
    x_t = (1 - t) x0 + t x1                  (linear interpolant)
    target velocity u = x1 - x0
    loss = || v_theta(x_t, t | damaged, mask) - u ||^2

Sampling integrates the learned velocity from t=0 to t=1 (Euler). We then blend
the generated image with the input outside the damage mask so undamaged pixels
are preserved exactly.

Time conditioning is injected with FiLM (per-channel scale/shift) so the same
U-Net can model the whole trajectory.
"""

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):  # t: [N] in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) *
                          torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return self.mlp(emb)


class FiLMConv(nn.Module):
    """Double conv with per-channel FiLM conditioning on the time embedding."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.film1 = nn.Linear(emb_dim, out_ch * 2)
        self.film2 = nn.Linear(emb_dim, out_ch * 2)

    @staticmethod
    def _film(h, scale, shift):
        return h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, x, emb):
        h = torch.relu(self.norm1(self.conv1(x)))
        h = self._film(h, *self.film1(emb).chunk(2, dim=1))
        h = torch.relu(self.norm2(self.conv2(h)))
        h = self._film(h, *self.film2(emb).chunk(2, dim=1))
        return h


def _match(x, ref):
    for dim in (-1, -2):
        if x.shape[dim] > ref.shape[dim]:
            x = x.narrow(dim, 0, ref.shape[dim])
        elif x.shape[dim] < ref.shape[dim]:
            pad = [0, 0, 0, 0]
            pad[1 if dim == -1 else 3] = ref.shape[dim] - x.shape[dim]
            x = torch.nn.functional.pad(x, pad)
    return x


class FlowDown(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = FiLMConv(in_ch, out_ch, emb_dim)

    def forward(self, x, emb):
        return self.conv(self.pool(x), emb)


class FlowUp(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = FiLMConv(out_ch + skip_ch, out_ch, emb_dim)

    def forward(self, x, skip, emb):
        x = _match(self.up(x), skip)
        return self.conv(torch.cat([skip, x], dim=1), emb)


class FlowRestoration(nn.Module):
    """Predicts the velocity field v(x_t, t | damaged, mask)."""

    def __init__(self, base: int = 32, emb_dim: int = 128, in_channels: int = 3):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.time = TimeEmbedding(emb_dim)
        self.inc = FiLMConv(in_channels, c1, emb_dim)
        self.down1 = FlowDown(c1, c2, emb_dim)
        self.down2 = FlowDown(c2, c3, emb_dim)
        self.down3 = FlowDown(c3, c4, emb_dim)
        self.up1 = FlowUp(c4, c3, c3, emb_dim)
        self.up2 = FlowUp(c3, c2, c2, emb_dim)
        self.up3 = FlowUp(c2, c1, c1, emb_dim)
        self.outc = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x, t):
        emb = self.time(t)
        x1 = self.inc(x, emb)
        x2 = self.down1(x1, emb)
        x3 = self.down2(x2, emb)
        x4 = self.down3(x3, emb)
        x = self.up1(x4, x3, emb)
        x = self.up2(x, x2, emb)
        x = self.up3(x, x1, emb)
        return self.outc(x)  # velocity (unconstrained sign)


def cfm_loss(model, clean, damaged, mask, t=None):
    """Conditional flow-matching loss for a batch."""
    n = clean.shape[0]
    if t is None:
        t = torch.rand(n, device=clean.device)
    x0 = torch.randn_like(clean)
    t_ = t[:, None, None, None]
    x_t = (1 - t_) * x0 + t_ * clean
    target = clean - x0
    cond = torch.cat([x_t, damaged, mask], dim=1)
    pred = model(cond, t)
    return nn.functional.mse_loss(pred, target)


@torch.no_grad()
def sample(model, damaged, mask, steps: int = 20, seed: int = None):
    """Integrate the flow from noise to a restored line, then mask-blend."""
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn_like(damaged)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((damaged.shape[0],), i * dt, device=damaged.device)
        v = model(torch.cat([x, damaged, mask], dim=1), t)
        x = x + v * dt
    x = x.clamp(0, 1)
    # Preserve undamaged pixels exactly.
    return damaged * (1 - mask) + x * mask


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torch.utils.data import DataLoader
    from restoration_data import RestorationDataset, restoration_collate

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ds = RestorationDataset(os.path.join(repo, "data", "manifest_train.json"))
    damaged, mask, clean, structure, widths, _ = next(
        iter(DataLoader(ds, batch_size=4, collate_fn=restoration_collate)))

    model = FlowRestoration()
    loss = cfm_loss(model, clean, damaged, mask)
    print(f"CFM loss {loss.item():.4f}")
    out = sample(model, damaged, mask, steps=5)
    print(f"sample {tuple(out.shape)}  range [{out.min():.3f}, {out.max():.3f}]")
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("flow restoration sanity check OK")
