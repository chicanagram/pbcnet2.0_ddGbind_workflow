"""Metrics for predicted vs experimental binding affinities.

PBCNet2.0 predicts *relative* affinity (ΔΔG between a target ligand and a reference).
Two evaluation levels matter:

* pairwise ΔΔG   — how well each predicted pair-difference matches experiment;
* per-ligand ΔG  — reconstruct absolute-ish values (pred_ΔΔG + reference_label),
  average over the references each ligand was compared against, then rank. This
  mirrors upstream ``Finetune.py`` (groupby ``Ligand1`` -> mean -> Spearman/Kendall).

Everything here is pure numpy/pandas/scipy so it runs anywhere (incl. the Mac with no
torch/dgl installed).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Metrics:
    n: int
    rmse: float
    mae: float
    pearson: float
    spearman: float
    kendall: float

    def as_dict(self) -> dict:
        return asdict(self)


def _corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if kind == "pearson":
        return float(stats.pearsonr(a, b)[0])
    if kind == "spearman":
        return float(stats.spearmanr(a, b)[0])
    if kind == "kendall":
        return float(stats.kendalltau(a, b)[0])
    raise ValueError(kind)


def compute(pred: np.ndarray, exp: np.ndarray) -> Metrics:
    """Core metrics between two aligned 1-D arrays."""
    pred = np.asarray(pred, dtype=float)
    exp = np.asarray(exp, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(exp)
    pred, exp = pred[mask], exp[mask]
    if len(pred) == 0:
        return Metrics(0, *[float("nan")] * 5)
    return Metrics(
        n=int(len(pred)),
        rmse=float(np.sqrt(np.mean((pred - exp) ** 2))),
        mae=float(np.mean(np.abs(pred - exp))),
        pearson=_corr(pred, exp, "pearson"),
        spearman=_corr(pred, exp, "spearman"),
        kendall=_corr(pred, exp, "kendall"),
    )


def per_ligand_from_pairs(
    df: pd.DataFrame,
    ligand_col: str = "Ligand1",
    pred_ddg_col: str = "pred_ddg",
    exp_ddg_col: str = "exp_ddg",
    ref_label_col: str | None = "ref_label",
) -> pd.DataFrame:
    """Collapse pairwise ΔΔG rows to one row per target ligand.

    If ``ref_label_col`` is given, absolute-ish values are reconstructed as
    ``ΔΔG + reference_label`` before averaging over references (the upstream recipe).
    Otherwise the raw ΔΔG columns are averaged directly.

    Returns a frame with columns ``[ligand, pred, exp]``.
    """
    d = df.copy()
    if ref_label_col and ref_label_col in d.columns:
        d["_pred"] = d[pred_ddg_col].astype(float) + d[ref_label_col].astype(float)
        d["_exp"] = d[exp_ddg_col].astype(float) + d[ref_label_col].astype(float)
    else:
        d["_pred"] = d[pred_ddg_col].astype(float)
        d["_exp"] = d[exp_ddg_col].astype(float)
    g = d.groupby(ligand_col)[["_pred", "_exp"]].mean().reset_index()
    return g.rename(columns={ligand_col: "ligand", "_pred": "pred", "_exp": "exp"})


def evaluate_target(
    df: pd.DataFrame,
    pred_ddg_col: str = "pred_ddg",
    exp_ddg_col: str = "exp_ddg",
    ligand_col: str = "Ligand1",
    ref_label_col: str | None = "ref_label",
) -> dict:
    """Return both pairwise-ΔΔG and per-ligand metrics for one target/system."""
    pair = compute(df[pred_ddg_col].values, df[exp_ddg_col].values)
    per_lig = per_ligand_from_pairs(
        df, ligand_col, pred_ddg_col, exp_ddg_col, ref_label_col
    )
    lig = compute(per_lig["pred"].values, per_lig["exp"].values)
    return {"pairwise_ddg": pair.as_dict(), "per_ligand": lig.as_dict()}


def summarize_targets(results: dict[str, dict]) -> pd.DataFrame:
    """Turn ``{target: evaluate_target(...)}`` into a tidy summary table."""
    rows = []
    for target, res in results.items():
        row = {"target": target}
        for level, m in res.items():
            for k, v in m.items():
                row[f"{level}.{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)
