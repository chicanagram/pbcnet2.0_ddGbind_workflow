# Environments

One git repo, **per-platform conda environments** (you cannot share a single lockfile —
the torch/DGL binaries differ by OS and by CUDA version).

| File | Platform | Role | DGL GPU? |
|------|----------|------|----------|
| `environment-linux-cuda.yml` | Linux x86-64 + NVIDIA (cluster, WSL2) | training, finetuning, full benchmarks | ✅ CUDA |
| `environment-osx-arm64.yml`  | macOS Apple Silicon (M2) | dev, inference, analysis, viz, interpretability | ❌ CPU-only |

Native **Windows** is intentionally unsupported for running the model — use **WSL2 Ubuntu**
with the Linux env instead (DGL + CUDA on native Windows is historically fragile).

## Quick start

```bash
# On the Linux GPU cluster / WSL2:
conda env create -f env/environment-linux-cuda.yml
conda activate pbcnet-cuda

# On the Mac (M2):
conda env create -f env/environment-osx-arm64.yml
conda activate pbcnet-cpu
```

Verify the install and see which device was selected:

```bash
python -m pbcnet_workflow.device      # prints torch/dgl versions + chosen device
```

## Matching CUDA on the cluster

Check the driver/toolkit with `nvidia-smi`, then edit the two `cu118` occurrences in
`environment-linux-cuda.yml` (the torch extra-index-url and the dgl find-links) to the
matching tag: `cu117`, `cu118`, or `cu121`.

## Reproducing upstream exactly

Upstream PBCNet2.0 was validated on Ubuntu 18.04, CUDA 11.4, Python 3.8, `dgl==1.0.2`.
If a newer DGL causes graph-API errors, recreate the pinned combo:

```bash
conda create -n pbcnet-legacy python=3.8
pip install torch==1.13.1 torchvision torchaudio
pip install dgl==1.0.2 -f https://data.dgl.ai/wheels/cu113/repo.html --no-deps
pip install pandas packaging PyYAML pydantic scipy matplotlib rdkit \
            networkx psutil tqdm scikit-learn biopython
```

## Fallbacks if `dgl` fails to solve on osx-arm64

1. Try a specific conda-forge build: `conda install -c conda-forge "dgl>=1.1,<1.2"`.
2. If unavailable, build from source (needs cmake + a recent clang) per the DGL docs.
3. Last resort: run everything model-related on the cluster and use the Mac only for
   `analyze`/`visualize` (those need only pandas/matplotlib/scipy — no torch/dgl).

## Optional: containers for the cluster

For reproducible cluster runs, an Apptainer/Singularity or Docker image built from the
`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` base + the pip lines above is recommended.
Ask and a `Dockerfile` / `.def` can be added.
