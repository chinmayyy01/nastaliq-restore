"""U-Net restoration baseline for damaged Nastaliq lines.

The network maps ``[damaged, damage_mask]`` to a clean line. It is the
deterministic backbone the novel conditioning (stroke graph) and generative
(flow matching) stages will build on, so it is deliberately plain:

    encoder: 3 down blocks (double conv + maxpool) -> /8 resolution
    bottleneck: double conv at the coarsest scale
    decoder: 3 up blocks (upsample + skip concat + double conv)
    head: 1x1 conv -> 1 channel -> sigmoid

Skips carry the fine stroke detail from the input so the output can sharpen
the regions the mask flags without hallucinating the rest.
"""

import torch
import torch.nn as nn


def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = double_conv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Upsampling can produce a size off by one vs the skip when the input
        # width is not a multiple of 8; crop or pad to match before concat.
        x = _match(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


def _match(x, ref):
    for dim in (-1, -2):
        if x.shape[dim] > ref.shape[dim]:
            x = x.narrow(dim, 0, ref.shape[dim])
        elif x.shape[dim] < ref.shape[dim]:
            pad = [0, 0, 0, 0]
            pad[1 if dim == -1 else 3] = ref.shape[dim] - x.shape[dim]
            x = torch.nn.functional.pad(x, pad)
    return x


class RestorationUNet(nn.Module):
    def __init__(self, in_channels: int = 2, base: int = 32, out_channels: int = 1):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.inc = double_conv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.up1 = Up(c4, c3, c3)
        self.up2 = Up(c3, c2, c2)
        self.up3 = Up(c2, c1, c1)
        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return torch.sigmoid(self.outc(x))


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from restoration_data import RestorationDataset, restoration_collate

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ds = RestorationDataset(os.path.join(repo, "data", "manifest_train.json"))
    damaged, mask, clean, structure, widths, _ = next(
        iter(DataLoader(ds, batch_size=4, collate_fn=restoration_collate)))

    model = RestorationUNet(in_channels=2)
    pred = model(torch.cat([damaged, mask], dim=1))
    print(f"input  {tuple(damaged.shape)} + mask -> pred {tuple(pred.shape)}")
    assert pred.shape == clean.shape
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.2f}M")
    print("restoration U-Net sanity check OK")
