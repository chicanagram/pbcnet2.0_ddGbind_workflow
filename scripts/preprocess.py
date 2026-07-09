#!/usr/bin/env python
"""Convert ligand SDF + pocket PDB pairs into the DGL ``.pkl`` graphs PBCNet2.0 eats.

This is a portable wrapper over upstream ``Graph2pickle.py`` (``Graph_Information``):
  * no hardcoded ``/home/user-home/...`` paths (upstream had one in ``__main__``);
  * cross-platform (uses pathlib, not POSIX ``rsplit('/')``);
  * configurable worker count (upstream hardcoded ``Pool(50)``); ``--workers 0`` runs
    serially, which is the safe default on macOS.

Two input modes
---------------
1. --sdf-dir DIR --pocket PDB
       Build one graph per ``*.sdf`` in DIR, all against a single pocket PDB.
2. --manifest CSV
       CSV with columns: ligand_sdf, pocket_pdb, out_pkl (one row per complex).

The graph build is pure-CPU (RDKit + BioPython) and O(n^2) in atom count — it is the
slow step of the whole pipeline, so preprocess once and cache the .pkl files.

Needs: rdkit, dgl, biopython, torch  (i.e. the full env — run on cluster or in pbcnet-cpu).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pbcnet_workflow.paths import add_model_code_to_path, EXTERNAL_ROOT  # noqa: E402


def _load_builder():
    """Import ``Graph_Information`` from the vendored Graph2pickle.py."""
    add_model_code_to_path()
    sys.path.insert(0, str(EXTERNAL_ROOT))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "graph2pickle", str(EXTERNAL_ROOT / "Graph2pickle.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def build_one(builder, ligand_sdf: Path, pocket_pdb: Path, out_pkl: Path) -> tuple[Path, str]:
    import pickle

    try:
        if not ligand_sdf.exists() or not pocket_pdb.exists():
            return out_pkl, "missing input"
        graph = builder.Graph_Information(str(ligand_sdf), str(pocket_pdb))
        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_pkl, "wb") as fh:
            pickle.dump(graph, fh)
        return out_pkl, "ok"
    except Exception as exc:  # noqa: BLE001
        return out_pkl, f"error: {exc}"


def collect_jobs(args) -> list[tuple[Path, Path, Path]]:
    jobs = []
    if args.manifest:
        import pandas as pd

        man = pd.read_csv(args.manifest)
        for _, r in man.iterrows():
            jobs.append((Path(r.ligand_sdf), Path(r.pocket_pdb), Path(r.out_pkl)))
    else:
        sdf_dir = Path(args.sdf_dir)
        pocket = Path(args.pocket)
        out_dir = Path(args.outdir or sdf_dir)
        for sdf in sorted(sdf_dir.glob("*.sdf")):
            jobs.append((sdf, pocket, out_dir / f"{sdf.stem}_dgl_group.pkl"))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sdf-dir", help="directory of *.sdf ligands")
    g.add_argument("--manifest", help="CSV: ligand_sdf,pocket_pdb,out_pkl")
    ap.add_argument("--pocket", help="pocket PDB (required with --sdf-dir)")
    ap.add_argument("--outdir", help="where to write .pkl (default: alongside SDFs)")
    ap.add_argument("--workers", type=int, default=0,
                    help="0=serial (safe on macOS); >0 uses a process pool")
    args = ap.parse_args()
    if args.sdf_dir and not args.pocket:
        ap.error("--pocket is required with --sdf-dir")

    builder = _load_builder()
    jobs = collect_jobs(args)
    print(f"{len(jobs)} graph(s) to build (workers={args.workers})")

    results = []
    if args.workers and args.workers > 1:
        import multiprocessing as mp

        with mp.Pool(args.workers) as pool:
            results = pool.starmap(
                build_one, [(builder, a, b, c) for a, b, c in jobs]
            )
    else:
        for a, b, c in jobs:
            results.append(build_one(builder, a, b, c))

    ok = sum(1 for _, s in results if s == "ok")
    print(f"done: {ok}/{len(results)} succeeded")
    for p, s in results:
        if s != "ok":
            print(f"  FAILED {p.name}: {s}")


if __name__ == "__main__":
    main()
