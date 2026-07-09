"""Centralised paths + sys.path wiring for the vendored upstream code.

Upstream modules use ``sys.path.append`` with assumptions about ``code_path`` and do
``rsplit('/', 1)`` (POSIX-only). Import them through this module so paths resolve on
macOS/Windows too, and so hardcoded ``/home/user-home/...`` / ``/code/...`` locations
are replaced by repo-relative ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

# repo layout: <REPO>/src/pbcnet_workflow/paths.py
REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / "external" / "PBCNet2.0"
MODEL_CODE = EXTERNAL_ROOT / "model_code"
DATA_ROOT = EXTERNAL_ROOT / "data"          # bundled test/benchmark data
RESULTS_ROOT = REPO_ROOT / "results"
CONFIG_ROOT = REPO_ROOT / "configs"


def add_model_code_to_path() -> None:
    """Make upstream packages (``models``, ``Dataloader``, ``utilis`` ...) importable."""
    p = str(MODEL_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
    pe = str(EXTERNAL_ROOT)
    if pe not in sys.path:
        sys.path.insert(0, pe)


if __name__ == "__main__":
    for name in ["REPO_ROOT", "EXTERNAL_ROOT", "MODEL_CODE", "DATA_ROOT"]:
        print(f"{name:14s} {globals()[name]}  exists={Path(globals()[name]).exists()}")
