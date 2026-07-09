"""Device selection that works across CUDA cluster, WSL2, and Apple-Silicon CPU.

Upstream PBCNet2.0 hardcodes ``cuda = "cuda:" + str(args.device)`` with no CPU path.
This module centralises device choice so the same code runs everywhere.

Key constraint: DGL's only GPU backend is CUDA. There is no MPS/Metal backend, so on
macOS we deliberately fall back to CPU even though ``torch.backends.mps`` may be
available — putting DGL graphs on CPU and torch tensors on MPS would break.
"""
from __future__ import annotations

import os


def resolve_device(prefer: str | int | None = None) -> str:
    """Return a torch device string usable for both torch tensors and DGL graphs.

    Args:
        prefer: ``None``/``"auto"`` -> auto-detect. An int ``i`` or ``"cuda:i"`` ->
            that CUDA device if available. ``"cpu"`` -> force CPU. The env var
            ``PBCNET_DEVICE`` overrides everything when set.

    Returns:
        e.g. ``"cuda:0"`` or ``"cpu"``.
    """
    import torch

    env = os.environ.get("PBCNET_DEVICE")
    if env:
        prefer = env

    if prefer in (None, "auto", ""):
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if isinstance(prefer, int):
        prefer = f"cuda:{prefer}"

    prefer = str(prefer)
    if prefer == "cpu":
        return "cpu"
    if prefer.isdigit():
        prefer = f"cuda:{prefer}"

    if prefer.startswith("cuda"):
        if torch.cuda.is_available():
            return prefer
        print(f"[pbcnet] requested '{prefer}' but CUDA is unavailable; using CPU.")
        return "cpu"

    if prefer.startswith("mps"):
        # DGL cannot use MPS — refuse rather than fail cryptically later.
        print("[pbcnet] MPS requested but DGL has no MPS backend; using CPU.")
        return "cpu"

    return prefer


def describe() -> str:
    """Human-readable summary of the runtime, for logs and `python -m ...device`."""
    import torch

    lines = [f"torch      {torch.__version__}"]
    try:
        import dgl

        lines.append(f"dgl        {dgl.__version__}")
    except Exception as exc:  # pragma: no cover - env-dependent
        lines.append(f"dgl        NOT INSTALLED ({exc})")
    lines.append(f"cuda avail {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"cuda dev   {torch.cuda.get_device_name(0)}")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    lines.append(f"mps avail  {bool(mps and mps.is_available())} (unused: DGL is CUDA-only)")
    lines.append(f"selected   {resolve_device('auto')}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
