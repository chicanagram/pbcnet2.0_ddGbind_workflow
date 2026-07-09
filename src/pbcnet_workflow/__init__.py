"""pbcnet_workflow: a portable, cross-platform wrapper around PBCNet2.0.

Submodules
----------
device      auto CPU/CUDA selection (DGL is CUDA-only; MPS unused)
paths       repo-relative paths + sys.path wiring for vendored upstream code
checkpoint  map_location-safe loading of the pretrained model
metrics     ddG-vs-experiment metrics (RMSE/MAE/Spearman/Kendall/Pearson)

The heavier, model-executing pieces live in ``scripts/`` (preprocess, benchmark,
finetune) so this package stays importable even where torch/dgl are absent
(e.g. for analysis/visualization on the Mac).
"""

__version__ = "0.1.0"
