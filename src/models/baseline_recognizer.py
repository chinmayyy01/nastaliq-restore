"""CNN + BiLSTM + CTC baseline recogniser for Nastaliq OCR.

Architecture (matches the standard Urdu OCR CTC baselines this project
compares against):

    input  [N, 1, 64, W]
      -> 4 conv blocks: 32 -> 64 -> 128 -> 256 channels
         (BatchNorm + ReLU + MaxPool each)
         blocks 1-2 pool (2, 2); blocks 3-4 pool (2, 1) so horizontal
         resolution is preserved -- width becomes the sequence axis
      -> collapse the remaining height into the channel dimension
         [N, 256, 4, W'] -> [T=W', N, 1024]
      -> 2-layer bidirectional LSTM (hidden 128)
      -> Linear -> vocab_size
      -> log_softmax
    output [T, N, vocab_size]   (the layout CTC expects)
"""

import os
import sys

import torch
import torch.nn as nn

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, pool):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool),
        )

    def forward(self, x):
        return self.block(x)


class BaselineRecognizer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128,
                 lstm_layers: int = 2, img_height: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.img_height = img_height

        # 4x height downsampling (all blocks) and 2x width downsampling
        # (only blocks 1-2), so the collapsed feature height is 4 at 64px.
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, pool=(2, 2)),
            ConvBlock(32, 64, pool=(2, 2)),
            ConvBlock(64, 128, pool=(2, 1)),
            ConvBlock(128, 256, pool=(2, 1)),
        )
        self.feature_height = img_height // 16
        self.lstm = nn.LSTM(
            input_size=256 * self.feature_height,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=False,
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        """x: [N, 1, H, W] -> log_probs: [T, N, vocab_size]."""
        f = self.cnn(x)                       # [N, 256, H', W']
        n, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).contiguous().view(n, w, c * h)
        f = f.permute(1, 0, 2)                # [T=w, N, c*h]
        out, _ = self.lstm(f)                 # [T, N, 2*hidden]
        logits = self.fc(out)                 # [T, N, vocab]
        return logits.log_softmax(dim=2)


def greedy_decode(log_probs, idx_to_char: dict, blank: int = 0) -> list:
    """CTC greedy decode: argmax per timestep, collapse repeats, drop blanks.

    Args:
        log_probs: [T, N, C] log-probabilities from the model.
        idx_to_char: mapping from vocab index back to character.
    Returns:
        list of N decoded strings.
    """
    best = log_probs.argmax(dim=2)            # [T, N]
    results = []
    for n in range(best.shape[1]):
        out = []
        prev = None
        for idx in best[:, n].tolist():
            if idx != blank and idx != prev:
                out.append(idx_to_char.get(idx, ""))
            prev = idx
        results.append("".join(out))
    return results


if __name__ == "__main__":
    import torch.utils.data

    from dataset import build_vocab, load_manifest, OCRDataset, collate_fn

    repo_root = os.path.dirname(_SRC_DIR)
    manifest_path = os.path.join(repo_root, "data", "manifest_train.json")
    manifest = load_manifest(manifest_path)
    vocab = build_vocab(manifest)
    print(f"vocab size (incl. blank): {len(vocab)}")

    ds = OCRDataset(manifest_path, vocab)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    images, labels, label_lengths, texts = next(iter(loader))

    model = BaselineRecognizer(vocab_size=len(vocab))
    log_probs = model(images)
    t, n, c = log_probs.shape
    print(f"images:    {tuple(images.shape)}")
    print(f"log_probs: {tuple(log_probs.shape)}  (T, N, vocab)")
    assert c == len(vocab)
    assert n == images.shape[0]

    input_lengths = torch.full((n,), t, dtype=torch.long)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(log_probs, labels, input_lengths, label_lengths)
    print(f"CTC loss:  {loss.item():.4f}")
    print("model sanity check OK")
