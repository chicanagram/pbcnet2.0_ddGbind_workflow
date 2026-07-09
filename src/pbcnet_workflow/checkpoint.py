"""Portable checkpoint loading for PBCNet2.0.

Two upstream gotchas this fixes:

1. The shipped weights file is ``PBCNet2.pth`` (not ``PBCNet2.0.pth`` as the README
   text says). ``default_checkpoint()`` points at the real file.
2. Upstream ``train.py`` does ``torch.load('/code/PBCNet.pth')`` with no
   ``map_location`` and an absolute path — this crashes on any machine without a GPU
   at cuda index 1. We always pass ``map_location``.

Note: the checkpoint is a *pickled full model object*, not a state_dict. Loading it
therefore requires the upstream model classes to be importable and a compatible DGL
version. ``paths.add_model_code_to_path()`` makes ``models.readout`` importable.
"""
from __future__ import annotations

from pathlib import Path

from .device import resolve_device
from .paths import EXTERNAL_ROOT, add_model_code_to_path


def default_checkpoint() -> Path:
    """Path to the bundled pretrained weights (handles the .pth naming quirk)."""
    cand = [EXTERNAL_ROOT / "PBCNet2.pth", EXTERNAL_ROOT / "PBCNet2.0.pth"]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint found. Looked for: {', '.join(str(c) for c in cand)}"
    )


def load_model(checkpoint: str | Path | None = None, device: str | int | None = "auto"):
    """Load PBCNet2.0 onto the resolved device.

    Args:
        checkpoint: path to a ``.pth``; ``None`` -> bundled pretrained weights.
        device: passed to :func:`resolve_device` (``"auto"`` by default).

    Returns:
        ``(model, device_str)`` with the model in ``eval()`` mode.
    """
    import torch

    add_model_code_to_path()  # so the pickled model's classes resolve
    dev = resolve_device(device)
    ckpt = Path(checkpoint) if checkpoint else default_checkpoint()

    # weights_only=False because upstream pickled the whole nn.Module, not a state_dict.
    model = torch.load(ckpt, map_location=torch.device(dev), weights_only=False)
    model.to(dev)
    model.eval()
    return model, dev
