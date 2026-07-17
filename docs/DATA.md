# Datasets & weights — what ships, what to source

> **This repo has been pared down** (423 MB → ~61 MB) to the assets relevant to the
> **flu PA/NA mutation-ΔΔG** use case. Removed from the upstream vendor: `Result_in_paper/`,
> `data/SAR-Diff/`, `data/Selection/`, `data/F-Opt/`, most of `data/FEP/`, and the redundant
> `.maegz` archives. Everything removed is recoverable from the initial git commit or the
> public upstream repo — see "Restoring removed data" below.

## Already in the repo (vendored under `external/PBCNet2.0/`)

| Asset | Location | Notes |
|-------|----------|-------|
| **Pretrained weights** | `external/PBCNet2.0/PBCNet2.pth` (3.1 MB) | ⚠️ README calls it `PBCNet2.0.pth`; the real filename is `PBCNet2.pth`. `pbcnet_workflow.checkpoint` handles this. |
| **Mutation set** ★ | `data/Mutation/<UniProt>/` (50 MB) | **Primary validation target.** 8 proteins (EGFR P00533 L858R/T790M/G719C, ACE P12821, aldose reductase P15121, …), same drug vs WT/mutant pocket, with per-pair experimental ΔΔG in `predict.csv`. Direct analogue of flu-mutation ΔΔG. Build pairs with `scripts/make_mutation_pairs.py`. |
| **FEP smoke-test** | `data/FEP/{pose_graph,direct_input}/eg5` (6.3 MB) | One FEP+ target (eg5) kept so the `run_benchmark`/`make_fep_pairs` pipeline stays testable. The other 15 targets were removed. |
| Example inference | `case/try.ipynb`, `case/toy_data/` (1.7 MB) | eg5 ligands + pocket, minimal "model loads & runs" check. |

So **weights + a mutation validation set + a pipeline smoke test are covered** — no download needed.

## Validate the method on the mutation set (no sourcing needed)

```bash
python scripts/make_mutation_pairs.py                       # -> results/mutation_pairs.csv (65 pairs, 8 targets)
python scripts/run_benchmark.py results/mutation_pairs.csv --out results/mut_pred.csv
python scripts/analyze_results.py results/mut_pred.csv --outdir results/mut_analysis
python scripts/visualize.py       results/mut_pred.csv --outdir results/mut_figs
```

This is the closest in-repo proxy for your flu PA/NA task: how well predicted ΔΔG tracks the
experimental effect of a **pocket mutation** on a drug's binding.

## Restoring removed data

```bash
# a specific path from the initial commit:
git checkout 1e2424f -- external/PBCNet2.0/data/FEP
# or re-vendor everything fresh from upstream:
git clone https://github.com/YuJie-0202/PBCNet2.0 /tmp/pbc && rm -rf /tmp/pbc/.git
```

## Needs downloading

- **Full training data** (only for retraining from scratch): Zenodo record
  **15656365** — <https://zenodo.org/records/15656365> — including `training_clip_862W.zip`.
  ~862k pairs; GPU-only. Preprocess with `scripts/preprocess.py`. You do **not** need this
  for benchmarking, inference, or finetuning.

## The "flu" set is bring-your-own

There is **no influenza dataset in the repo**. To benchmark on flu you must assemble it in
the model's input format. For an influenza target (e.g. **neuraminidase**, or PA/PB1/PB2 of
the polymerase), you need, per congeneric ligand series:

1. **A prepared pocket** — `pocket.pdb` (protein, hydrogens handled as upstream does:
   `Graph2pickle` strips H). One pocket per binding site.
2. **Docked/co-crystal ligand poses** — one `*.sdf` per ligand, aligned in that pocket.
   (Poses matter: the graph uses 3-D coordinates and ≤5 Å contacts.)
3. **Experimental affinities** — IC50/Ki/ΔG per ligand, converted to a common scale, so
   you can form pairwise **exp_ddg = value(target) − value(reference)**.

Then:

```bash
# 1) build graphs
python scripts/preprocess.py --sdf-dir data/flu_NA/poses --pocket data/flu_NA/pocket.pdb \
                             --outdir data/flu_NA/graphs
# 2) write a pairs CSV: target, ligand1_pkl, ligand2_pkl, exp_ddg, ligand1_id, ref_label
#    (all-vs-reference, or all-vs-all within the series)
# 3) score
python scripts/run_benchmark.py data/flu_NA/pairs.csv --out results/flu_NA_pred.csv
```

Candidate public sources for flu ligand/affinity data (verify licensing before use):
BindingDB, ChEMBL (neuraminidase = CHEMBL sets for NA), PDBbind (flu complexes), and the
PDB for co-crystal structures. A small helper to turn a ChEMBL/BindingDB export + docked
poses into the pairs CSV can be added on request.

## Pairs-CSV schema (the workflow's common currency)

`run_benchmark.py` / `run_finetune.py` input:

| column | required | meaning |
|--------|----------|---------|
| `target` | ✓ | system name (groups metrics/plots) |
| `ligand1_pkl` | ✓ | target ligand graph (.pkl) |
| `ligand2_pkl` | ✓ | reference ligand graph (.pkl) |
| `exp_ddg` | for metrics/finetune | experimental ΔΔG(target − reference) |
| `ligand1_id` | optional | label for per-ligand grouping |
| `ref_label` | optional | reference absolute value, for per-ligand ranking |

`analyze_results.py` / `visualize.py` input (produced by the above): `target, Ligand1,
pred_ddg, exp_ddg, ref_label`.
