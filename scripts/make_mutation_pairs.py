#!/usr/bin/env python
"""Build a pairs CSV from the bundled Mutation set (WT-vs-mutant ΔΔG).

This is the validation set closest to the flu PA/NA use case: same drug, wild-type vs
mutant pocket, experimental ΔΔG. Each ``data/Mutation/<UniProt>/predict.csv`` has columns
``lig1, lig2, Label, Label1, Label2`` with the ``.pkl`` graphs co-located in that folder.

Emits this workflow's pairs schema (paths relative to --data-root):
    target, ligand1_pkl, ligand2_pkl, exp_ddg, ligand1_id, ref_label

where target = UniProt id (P00533=EGFR, ...), exp_ddg = Label (ΔΔG(lig1 - lig2)),
ligand1_id = the mutant variant name (the axis that varies), ref_label = Label2.

Then:  python scripts/run_benchmark.py results/mutation_pairs.csv --out results/mut_pred.csv
       python scripts/analyze_results.py results/mut_pred.csv --outdir results/mut_analysis

Pure pandas — runs anywhere.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbcnet_workflow.paths import DATA_ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="ALL",
                    help="UniProt id (e.g. P00533) or ALL (default) to combine every target")
    ap.add_argument("--out", default="results/mutation_pairs.csv")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    args = ap.parse_args()

    mut_root = Path(args.data_root) / "Mutation"
    targets = ([args.target] if args.target != "ALL"
               else sorted(p.name for p in mut_root.iterdir() if p.is_dir()))

    frames = []
    for t in targets:
        csv = mut_root / t / "predict.csv"
        if not csv.exists():
            print(f"  skip {t}: no predict.csv")
            continue
        df = pd.read_csv(csv)
        rel = lambda n: str(Path("Mutation") / t / n)  # noqa: E731
        frames.append(pd.DataFrame({
            "target": t,
            "ligand1_pkl": df["lig1"].map(rel),
            "ligand2_pkl": df["lig2"].map(rel),
            "exp_ddg": df["Label"].astype(float),
            "ligand1_id": df["lig2"].map(lambda n: Path(n).stem),  # the mutant variant
            "ref_label": df.get("Label2", float("nan")),
        }))

    if not frames:
        ap.error("no Mutation targets found")
    out = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{len(out)} WT-vs-mutant pairs across {out.target.nunique()} target(s) -> {args.out}")
    print(out.groupby("target").size().to_string())


if __name__ == "__main__":
    main()
