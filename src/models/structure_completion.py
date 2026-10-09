"""Structure-completion network.

This is novelty #1: instead of conditioning restoration on the structure that
*survived* the damage (redundant where the tears are), we predict the structure
of the *clean* line from the damaged line plus the damage mask. The prediction
hallucinates the missing stroke topology, so conditioning restoration on it
adds information the damaged pixels do not contain.

Output has two channels matching ``structure_prior``: skeleton and stroke-width
(distance), both in ``[0, 1]``.
"""

import os
import sys

import torch
import torch.nn as nn

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.restoration_unet import double_conv, Down, Up


class StructureCompletion(nn.Module):
    def __init__(self, in_channels: int = 2, base: int = 32, out_channels: int = 2):
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
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torch.utils.data import DataLoader
    from restoration_data import RestorationDataset, restoration_collate

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ds = RestorationDataset(os.path.join(repo, "data", "manifest_train.json"),
                            structure_source="clean")
    damaged, mask, clean, structure, widths, _ = next(
        iter(DataLoader(ds, batch_size=4, collate_fn=restoration_collate)))

    model = StructureCompletion(in_channels=2)
    pred = model(torch.cat([damaged, mask], dim=1))
    print(f"input {tuple(damaged.shape)} -> structure pred {tuple(pred.shape)} "
          f"target {tuple(structure.shape)}")
    assert pred.shape == structure.shape
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("structure-completion sanity check OK")
