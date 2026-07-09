# Datasets & weights — what ships, what to source

## Already in the repo (vendored under `external/PBCNet2.0/`)

| Asset | Location | Notes |
|-------|----------|-------|
| **Pretrained weights** | `external/PBCNet2.0/PBCNet2.pth` (3.1 MB) | ⚠️ README calls it `PBCNet2.0.pth`; the real filename is `PBCNet2.pth`. `pbcnet_workflow.checkpoint` handles this. |
| **FEP+ benchmark inputs** | `data/FEP/direct_input/*.csv`, `data/FEP/pose_graph/`, `data/FEP/finetune_input/` | 16 standard targets: PTP1B, Thrombin, Tyk2, CDK2, Jnk1, Bace, MCL1, p38, syk, shp2, pfkfb3, eg5, cdk8, cmet, tnks2, hif2a. |
| **Mutation set** | `data/Mutation/<UniProt>/` | UniProt IDs incl. EGFR (P00533). Same-ligand WT-vs-mutant complexes — the basis for the mutation-effect interpretability work. |
| **Other eval sets** | `data/F-Opt/`, `data/SAR-Diff/`, `data/Selection/` | pose optimization, SAR pairs, virtual-screening selection. |
| Example inference | `case/try.ipynb`, `case/toy_data/` | eg5 ligands + pocket, a good smoke test. |

So **weights + the FEP benchmark are covered** — no download needed for those.

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
