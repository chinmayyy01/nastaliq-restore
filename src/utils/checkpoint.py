"""Checkpoint save/load helpers shared by the training scripts."""

import os

import torch


def save_checkpoint(model, optimizer, epoch: int, cer: float, path: str) -> None:
    """Save model/optimizer state plus the epoch and validation CER."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "cer": cer,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }, path)


def load_checkpoint(model, optimizer, path: str, map_location=None):
    """Load a checkpoint into ``model`` (and ``optimizer`` if given).

    Returns ``(epoch, cer)``.
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["cer"]
