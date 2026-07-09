#!/usr/bin/env python
"""Visualize predicted ΔΔG vs experiment.

Consumes the same long predictions CSV as ``analyze_results.py`` and produces:

    scatter_pairwise.png        pred vs exp ΔΔG, coloured by target, y=x + ±1 kcal band
    scatter_per_target/<t>.png  one clean scatter per target (per-ligand level)
    error_hist.png              distribution of (pred - exp)
    metrics_bar.png             per-target RMSE + Spearman bars

Needs only pandas/numpy/scipy/matplotlib — runs on the Mac.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / SSH-safe
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbcnet_workflow import metrics  # noqa: E402

KCAL_BAND = 1.0  # highlight the ±1 kcal/mol "chemical accuracy" band


def _unit_line(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], "k-", lw=1, alpha=0.6, zorder=0)
    ax.fill_between([lo, hi], [lo - KCAL_BAND, hi - KCAL_BAND],
                    [lo + KCAL_BAND, hi + KCAL_BAND], color="grey", alpha=0.12, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")


def _lims(*arrs):
    v = np.concatenate([np.asarray(a, float) for a in arrs])
    v = v[np.isfinite(v)]
    pad = 0.05 * (v.max() - v.min() + 1e-9)
    return v.min() - pad, v.max() + pad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions")
    ap.add_argument("--outdir", default="results/figures")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--ligand-col", default="Ligand1")
    ap.add_argument("--pred-col", default="pred_ddg")
    ap.add_argument("--exp-col", default="exp_ddg")
    ap.add_argument("--ref-label-col", default="ref_label")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    ref_col = args.ref_label_col or None
    outdir = Path(args.outdir)
    (outdir / "scatter_per_target").mkdir(parents=True, exist_ok=True)

    targets = sorted(df[args.target_col].unique())
    cmap = plt.get_cmap("tab20")

    # --- 1. pooled pairwise scatter, coloured by target ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, t in enumerate(targets):
        sub = df[df[args.target_col] == t]
        ax.scatter(sub[args.exp_col], sub[args.pred_col], s=14, alpha=0.6,
                   color=cmap(i % 20), label=str(t))
    lo, hi = _lims(df[args.exp_col], df[args.pred_col])
    _unit_line(ax, lo, hi)
    ax.set_xlabel("Experimental ΔΔG (kcal/mol)")
    ax.set_ylabel("Predicted ΔΔG (kcal/mol)")
    m = metrics.compute(df[args.pred_col].values, df[args.exp_col].values)
    ax.set_title(f"Pairwise ΔΔG  (n={m.n}, RMSE={m.rmse:.2f}, ρ={m.spearman:.2f})")
    if len(targets) <= 20:
        ax.legend(fontsize=6, ncol=2, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(outdir / "scatter_pairwise.png", dpi=150)
    plt.close(fig)

    # --- 2. per-target per-ligand scatters + collect bar-chart data ---
    bar = []
    for i, t in enumerate(targets):
        sub = df[df[args.target_col] == t]
        pl = metrics.per_ligand_from_pairs(sub, args.ligand_col, args.pred_col,
                                           args.exp_col, ref_col)
        mm = metrics.compute(pl["pred"].values, pl["exp"].values)
        bar.append({"target": str(t), "rmse": mm.rmse, "spearman": mm.spearman, "n": mm.n})

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(pl["exp"], pl["pred"], s=24, alpha=0.75, color=cmap(i % 20))
        if len(pl) >= 2:
            lo, hi = _lims(pl["exp"], pl["pred"])
            _unit_line(ax, lo, hi)
        ax.set_xlabel("Experimental (per ligand)")
        ax.set_ylabel("Predicted (per ligand)")
        ax.set_title(f"{t}  RMSE={mm.rmse:.2f}  ρ={mm.spearman:.2f}")
        fig.tight_layout()
        fig.savefig(outdir / "scatter_per_target" / f"{t}.png", dpi=150)
        plt.close(fig)

    # --- 3. error histogram ---
    err = (df[args.pred_col] - df[args.exp_col]).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(err, bins=40, color="steelblue", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Predicted − Experimental ΔΔG (kcal/mol)")
    ax.set_ylabel("count")
    ax.set_title(f"Signed error  (mean={err.mean():.2f}, std={err.std():.2f})")
    fig.tight_layout()
    fig.savefig(outdir / "error_hist.png", dpi=150)
    plt.close(fig)

    # --- 4. per-target metric bars ---
    bdf = pd.DataFrame(bar)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(6, 0.5 * len(bdf)), 7), sharex=True)
    a1.bar(bdf["target"], bdf["rmse"], color="indianred")
    a1.axhline(KCAL_BAND, color="k", ls="--", lw=1, label=f"{KCAL_BAND} kcal/mol")
    a1.set_ylabel("per-ligand RMSE"); a1.legend(fontsize=8)
    a2.bar(bdf["target"], bdf["spearman"], color="seagreen")
    a2.set_ylabel("per-ligand Spearman ρ"); a2.set_ylim(-0.2, 1.0)
    for ax in (a1, a2):
        ax.tick_params(axis="x", rotation=90, labelsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "metrics_bar.png", dpi=150)
    plt.close(fig)

    print(f"Wrote figures to {outdir}/")


if __name__ == "__main__":
    main()
