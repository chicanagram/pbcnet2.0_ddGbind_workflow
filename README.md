# PBCNet2.0 ΔΔG binding workflow

A cross-platform workflow around **[PBCNet2.0](https://github.com/YuJie-0202/PBCNet2.0)**
(Yu & Sheng) — a Cartesian-tensor equivariant GNN that predicts **relative** protein–ligand
binding affinity (ΔΔG between congeneric ligands, the FEP lead-optimization setting).

This repo vendors the upstream model under `external/PBCNet2.0/` and adds a portable
overlay for: **benchmarking, finetuning, ddG-vs-experiment analysis/visualization**, and a
plan for **LLM-aided feature interpretability** (esp. mutation effects).

> ⚠️ **Data classification (org policy):** treat model weights, structures, and any
> benchmark/experimental data here as **ARES PRIVATE**. Do not upload to external services
> (including hosted LLM APIs) without clearing it against A*STAR data-handling rules.
> See the caveat in [docs/INTERPRETABILITY.md](docs/INTERPRETABILITY.md).

## Platform support at a glance

| Machine | Role | Runs the GNN? |
|---|---|---|
| **Linux cluster (GPU, SSH)** | training, finetuning, full benchmarks | ✅ CUDA |
| **WSL2 Ubuntu** | Linux-equivalent dev; GPU if NVIDIA host | ✅ (WSL CUDA) |
| **Mac M2** | dev, inference, analysis, viz, interpretability | ✅ CPU-only* |
| **Native Windows** | not supported for the model — use WSL2 | ⚠️ avoid |

\* DGL's only GPU backend is CUDA (no Apple MPS / ROCm), so on Apple Silicon it runs on CPU
regardless of the M2 GPU. Fine for inference/small finetunes; **not** for full training.
Analysis & visualization need only pandas/matplotlib/scipy and run anywhere.

## Install

See [env/README.md](env/README.md). Short version:

```bash
# Linux GPU cluster / WSL2
conda env create -f env/environment-linux-cuda.yml && conda activate pbcnet-cuda
# macOS Apple Silicon
conda env create -f env/environment-osx-arm64.yml && conda activate pbcnet-cpu

python -m pbcnet_workflow.device     # sanity: prints versions + selected device
```

(You may need `export PYTHONPATH=$PWD/src` or `pip install -e .` once packaging is added;
the scripts already prepend `src/` to `sys.path`.)

## The four workflows

```bash
# 0. Preprocess SDF+PDB -> DGL .pkl graphs (CPU, slow, cache it)
python scripts/preprocess.py --sdf-dir data/flu_NA/poses --pocket data/flu_NA/pocket.pdb \
                             --outdir data/flu_NA/graphs

# 1. Benchmark: score a pairs CSV -> predictions CSV
python scripts/run_benchmark.py data/flu_NA/pairs.csv --out results/flu_NA_pred.csv

# 2. Finetune (few-shot) on a target, then score a held-out set
python scripts/run_finetune.py --finetune-csv data/flu_NA/train_pairs.csv \
       --eval-csv data/flu_NA/eval_pairs.csv --name flu_NA --epochs 10

# 3. Analysis: RMSE / MAE / Spearman / Kendall / Pearson, pairwise + per-ligand
python scripts/analyze_results.py results/flu_NA_pred.csv --outdir results/analysis

# 3b. Visualization: scatter (±1 kcal band), error hist, per-target metric bars
python scripts/visualize.py results/flu_NA_pred.csv --outdir results/figures

# 4. Interpretability — see docs/INTERPRETABILITY.md (design only, not yet built)
```

Data formats, what ships vs what to source (incl. bring-your-own **flu** data and the
Zenodo training set), are in [docs/DATA.md](docs/DATA.md).

## Layout

```
external/PBCNet2.0/     vendored upstream (patched minimally — see patches/README.md)
src/pbcnet_workflow/    portable overlay: device, paths, checkpoint, metrics
scripts/                preprocess / run_benchmark / run_finetune / analyze / visualize
env/                    per-platform conda envs + notes
configs/                example config (knob reference)
docs/                   DATA.md, INTERPRETABILITY.md
data/  results/  notebooks/
```

## Status / validated

- ✅ Analysis + visualization pipeline validated end-to-end (pure pandas/scipy/matplotlib).
- ⏳ `preprocess.py`, `run_benchmark.py`, `run_finetune.py` are written against the real
  upstream APIs but **need a first run in a full torch+dgl env** (Mac CPU or cluster) to
  confirm graph/checkpoint compatibility. First smoke test: `external/PBCNet2.0/case/`.
- 📝 Interpretability is a design doc only (per plan), no code yet.

Upstream: <https://github.com/YuJie-0202/PBCNet2.0> · License: MIT (upstream) —
"Advancing Ligand Binding Affinity Prediction with Cartesian Tensor-Based Deep Learning".
