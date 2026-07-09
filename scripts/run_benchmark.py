#!/usr/bin/env python
"""Run PBCNet2.0 inference over a benchmark set and emit a predictions CSV.

Input: a "pairs" CSV describing target/reference ligand comparisons. Required columns:

    target        system name (Tyk2, EGFR_L858R, flu_NA, ...)
    ligand1_pkl   path to the TARGET ligand's preprocessed .pkl graph
    ligand2_pkl   path to the REFERENCE ligand's preprocessed .pkl graph
    exp_ddg       experimental ΔΔG (target - reference), kcal/mol   [optional for pure prediction]

  optional:
    ligand1_id    label for the target ligand (defaults to ligand1_pkl stem)
    ref_label     reference ligand's absolute experimental value (for per-ligand ranking)

Paths may be absolute or relative to --data-root (default: the vendored data dir).
Build the .pkl graphs first with scripts/preprocess.py.

Output (--out): the long predictions CSV consumed by analyze_results.py / visualize.py:
    target, Ligand1, pred_ddg, exp_ddg, ref_label

Model prediction convention: model(g_target, g_reference) -> ΔΔG(target - reference),
matching the Siamese readout ``FNN(norm(emb_target - emb_reference))``.

Needs the full env (torch + dgl). Runs on CPU (Mac) or CUDA (cluster) automatically.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbcnet_workflow.checkpoint import load_model  # noqa: E402
from pbcnet_workflow.paths import DATA_ROOT  # noqa: E402


def _resolve(p: str, root: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def _load_graph(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def run(df: pd.DataFrame, model, device: str, data_root: Path, batch_size: int) -> pd.DataFrame:
    import dgl
    import torch

    preds, keep = [], []
    rows = df.reset_index(drop=True)
    for start in range(0, len(rows), batch_size):
        chunk = rows.iloc[start:start + batch_size]
        g1s, g2s, idx = [], [], []
        for i, r in chunk.iterrows():
            try:
                g1 = _load_graph(_resolve(r.ligand1_pkl, data_root))
                g2 = _load_graph(_resolve(r.ligand2_pkl, data_root))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip row {i}: {exc}")
                continue
            g1s.append(g1); g2s.append(g2); idx.append(i)
        if not g1s:
            continue
        b1 = dgl.batch(g1s).to(device)
        b2 = dgl.batch(g2s).to(device)
        with torch.no_grad():
            out, _ = model(b1, b2)
        preds.extend(out.squeeze(-1).cpu().numpy().tolist())
        keep.extend(idx)
        print(f"  scored {len(keep)}/{len(rows)}", end="\r")
    print()

    res = rows.loc[keep].copy()
    res["pred_ddg"] = preds
    out = pd.DataFrame({
        "target": res.get("target", "unknown"),
        "Ligand1": res.get("ligand1_id",
                           res["ligand1_pkl"].map(lambda p: Path(p).stem)),
        "pred_ddg": res["pred_ddg"],
    })
    out["exp_ddg"] = res["exp_ddg"].values if "exp_ddg" in res else float("nan")
    out["ref_label"] = res["ref_label"].values if "ref_label" in res else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs_csv", help="benchmark pairs CSV (see module docstring)")
    ap.add_argument("--out", default="results/predictions.csv")
    ap.add_argument("--checkpoint", default=None, help="default: bundled PBCNet2.pth")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda:N")
    ap.add_argument("--data-root", default=str(DATA_ROOT),
                    help="base for relative .pkl paths in the CSV")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs_csv)
    required = {"target", "ligand1_pkl", "ligand2_pkl"}
    missing = required - set(df.columns)
    if missing:
        ap.error(f"pairs CSV missing columns: {sorted(missing)}")

    model, device = load_model(args.checkpoint, args.device)
    print(f"[pbcnet] model on {device}; scoring {len(df)} pairs")

    out = run(df, model, device, Path(args.data_root), args.batch_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} predictions to {args.out}")
    if out["exp_ddg"].notna().any():
        from pbcnet_workflow import metrics
        m = metrics.compute(out["pred_ddg"].values, out["exp_ddg"].values)
        print(f"Quick pairwise: RMSE={m.rmse:.2f}  MAE={m.mae:.2f}  "
              f"Spearman={m.spearman:.2f}  (n={m.n})")


if __name__ == "__main__":
    main()
