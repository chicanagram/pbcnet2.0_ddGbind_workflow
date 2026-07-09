# CLAUDE.md — working notes for this repo

## What this is
A cross-platform workflow wrapping the vendored **PBCNet2.0** GNN (`external/PBCNet2.0/`),
which predicts **relative** protein–ligand binding affinity (ΔΔG between congeneric
ligands; Siamese/pairwise, FEP lead-opt setting). We add benchmarking, finetuning,
analysis/viz, and (planned) LLM interpretability.

## Model in one breath
`PBCNetv2` = TensorNet equivariant encoder (per-atom Cartesian tensors decomposed into
I/A/S irreps, message passing over ≤5 Å distance + covalent edges) → mask protein atoms →
sum ligand-atom embeddings → Siamese head `FNN(LayerNorm(emb_target − emb_reference))` → ΔΔG.
Key files: `model_code/models/{readout,tensornet}.py`, `Graph2pickle.py` (graph build),
`model_code/{train,Finetune}.py`, `model_code/predict/predict.py`.

## Hard-won facts / gotchas
- Checkpoint file is **`PBCNet2.pth`**, not `PBCNet2.0.pth` (README is wrong). It's a
  **pickled full nn.Module**, not a state_dict — needs upstream classes importable + a
  compatible DGL. Always load with `map_location` (see `src/pbcnet_workflow/checkpoint.py`).
- Upstream hardcodes CUDA (`cuda = "cuda:"+str(device)`), absolute paths (`/code/...`,
  `/home/user-home/...`), and POSIX `rsplit('/')`. Our overlay + one `train.py` patch fix it.
- **DGL is CUDA-only** — no MPS/ROCm. On Mac it's CPU. Never send DGL graphs to MPS.
- Preprocessing (`Graph2pickle`) is CPU-bound, O(n²) over atoms — cache the `.pkl` graphs.
- Column-name inconsistency upstream: `Label` vs `Lable`, `Ligand1/2`, `Ligand1_num`,
  `file_name`, `again_number`. Our scripts define one clean schema (see docs/DATA.md).

## Platforms
Train/benchmark on the **Linux GPU cluster** (or WSL2). Use the **Mac (CPU)** for dev,
inference, analysis, viz, interpretability. Avoid native Windows for the model (use WSL2).
Two conda envs in `env/`; device auto-selected by `pbcnet_workflow.device.resolve_device`.

## Conventions
- New code goes in `src/pbcnet_workflow/` (importable, no torch/dgl at import time) or
  `scripts/` (CLI, may need the full env). Don't edit `external/` except documented
  `# [pbcnet-patch]` lines (log them in `patches/README.md`).
- The **pairs CSV** (target, ligand1_pkl, ligand2_pkl, exp_ddg, [ligand1_id, ref_label]) is
  the common input; the **long predictions CSV** (target, Ligand1, pred_ddg, exp_ddg,
  ref_label) is the common output. Keep to these so analysis/viz stay decoupled.
- Prediction convention: `model(g_target, g_reference)` → ΔΔG(target − reference).

## Local env note
This Mac's `python3` (homebrew) is bare; use **miniconda** (`~/miniconda3`) and the
`pbcnet-cpu` env. Analysis/viz were validated with the conda base interpreter.

## Data classification
Repo assets are **ARES PRIVATE** (org policy). Don't push structures/affinities to hosted
services or LLM APIs without clearance — see docs/INTERPRETABILITY.md §5 caveat.
