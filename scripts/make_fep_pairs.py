#!/usr/bin/env python
"""Convert a bundled upstream FEP CSV into this workflow's clean pairs schema.

Upstream ``external/PBCNet2.0/data/FEP/direct_input/<sys>.csv`` uses columns
``Ligand1, Ligand2, Lable, Lable1, Lable2, Ligand1_num`` where Ligand1/2 are
``<sys>//lig_..._dgl_group.pkl`` paths relative to ``data/FEP/pose_graph/``, and the
graphs already exist there. This emits the pairs CSV that ``run_benchmark.py`` expects:

    target, ligand1_pkl, ligand2_pkl, exp_ddg, ligand1_id, ref_label

with .pkl paths written relative to --data-root (default: the vendored data dir), so
`run_benchmark.py <pairs.csv>` runs with no other setup. Lets you smoke-test on any of
the 16 FEP targets using only in-repo data.
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
    ap.add_argument("system", help="FEP target name, e.g. eg5, Tyk2, MCL1")
    ap.add_argument("--out", default=None, help="default: results/<system>_pairs.csv")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    args = ap.parse_args()

    data_root = Path(args.data_root)
    src = data_root / "FEP" / "direct_input" / f"{args.system}.csv"
    if not src.exists():
        ap.error(f"no such FEP CSV: {src}\n"
                 f"available: {[p.stem for p in sorted((data_root/'FEP'/'direct_input').glob('*.csv'))]}")
    df = pd.read_csv(src)

    def rel(p: str) -> str:
        # upstream stores "<sys>//lig_..._dgl_group.pkl" relative to FEP/pose_graph/
        return str(Path("FEP") / "pose_graph" / p.replace("//", "/"))

    out = pd.DataFrame({
        "target": args.system,
        "ligand1_pkl": df["Ligand1"].map(rel),
        "ligand2_pkl": df["Ligand2"].map(rel),
        "exp_ddg": df["Lable"].astype(float),               # ΔΔG(target - reference)
        "ligand1_id": df.get("Ligand1_num", df["Ligand1"]),
        "ref_label": df.get("Lable2", float("nan")),        # reference absolute label
    })
    out_path = Path(args.out) if args.out else Path("results") / f"{args.system}_pairs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"{len(out)} pairs -> {out_path}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
