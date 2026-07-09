#!/usr/bin/env python
"""Analyze predicted ΔΔG vs experimental values.

Input: a "long" predictions CSV with (at least) these columns:
    target        system/target name (e.g. Tyk2, EGFR_L858R, flu_NA)
    Ligand1       target ligand id (the one being scored)
    pred_ddg      model-predicted ΔΔG (target - reference), kcal/mol
    exp_ddg       experimental ΔΔG for the same pair
    ref_label     (optional) reference ligand's absolute label, for per-ligand ranking

`scripts/run_benchmark.py` writes exactly this schema. You can also hand-build it.

Output (under --outdir):
    metrics_by_target.csv     one row per target: pairwise + per-ligand metrics
    metrics_overall.csv       pooled across all targets
    per_ligand_<target>.csv   collapsed per-ligand pred/exp for plotting

This script needs only pandas/numpy/scipy — no torch/dgl — so it runs on the Mac.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbcnet_workflow import metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="long predictions CSV (see module docstring)")
    ap.add_argument("--outdir", default="results/analysis", help="output directory")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--ligand-col", default="Ligand1")
    ap.add_argument("--pred-col", default="pred_ddg")
    ap.add_argument("--exp-col", default="exp_ddg")
    ap.add_argument("--ref-label-col", default="ref_label",
                    help="set to '' to skip per-ligand absolute reconstruction")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    missing = {args.target_col, args.pred_col, args.exp_col} - set(df.columns)
    if missing:
        ap.error(f"predictions CSV missing columns: {sorted(missing)}")
    ref_col = args.ref_label_col or None

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for target, sub in df.groupby(args.target_col):
        results[str(target)] = metrics.evaluate_target(
            sub, args.pred_col, args.exp_col, args.ligand_col, ref_col
        )
        per_lig = metrics.per_ligand_from_pairs(
            sub, args.ligand_col, args.pred_col, args.exp_col, ref_col
        )
        per_lig.to_csv(outdir / f"per_ligand_{target}.csv", index=False)

    summary = metrics.summarize_targets(results)
    summary.to_csv(outdir / "metrics_by_target.csv", index=False)

    overall = pd.DataFrame([{
        "target": "OVERALL",
        **{f"pairwise_ddg.{k}": v for k, v in
           metrics.compute(df[args.pred_col].values, df[args.exp_col].values).as_dict().items()},
    }])
    overall.to_csv(outdir / "metrics_overall.csv", index=False)

    pd.set_option("display.width", 160)
    print("Per-target metrics:\n")
    print(summary.round(3).to_string(index=False))
    print(f"\nWrote analysis to {outdir}/")


if __name__ == "__main__":
    main()
