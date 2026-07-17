#!/usr/bin/env python
"""Extract a docked ligand pose from a complex PDB and save it as SDF.

Why a template is (almost always) needed
-----------------------------------------
PDB files store atoms + coordinates but NOT bond orders or aromaticity. Reading a ligand
straight from PDB therefore yields all-single-bond, non-aromatic connectivity, which makes
RDKit compute wrong hybridization/valence/aromatic flags. ``Graph2pickle.py`` featurizes
atoms from exactly those properties, so a template-free SDF will silently corrupt the graph.

The fix: supply the ligand's known SMILES via --smiles. We extract the ligand's atoms +
3-D coordinates from the PDB, then use RDKit's ``AssignBondOrdersFromTemplate`` to graft the
correct bond orders onto the docked coordinates. Atom order/count must match (heavy atoms).

Usage
-----
  python scripts/extract_ligand.py complex.pdb --resname LIG --smiles "Cc1ccccc1..." \
         --out data/flu_NA/poses/lig1.sdf

  # auto-detect the ligand (largest non-solvent HETATM residue) if you don't know resname:
  python scripts/extract_ligand.py complex.pdb --smiles "..." --out lig1.sdf

Without --smiles it still writes an SDF (bond perception from geometry) but PRINTS A WARNING
— only acceptable if you will re-perceive bonds downstream. Needs: rdkit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOLVENT = {"HOH", "WAT", "NA", "CL", "K", "MG", "ZN", "CA", "SO4", "PO4",
           "GOL", "EDO", "ACT", "DMS", "PEG", "MN", "FE", "CU"}


def _load_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem  # noqa: F401
        return Chem
    except ImportError:
        sys.exit("RDKit not found — activate pbcnet-cpu / pbcnet-cuda first.")


def _ligand_resnames(pdb_path: Path) -> list[str]:
    """HETATM residue names that aren't solvent/ions, by descending atom count."""
    counts: dict[str, int] = {}
    for line in pdb_path.read_text().splitlines():
        if line.startswith("HETATM"):
            resn = line[17:20].strip()
            if resn and resn not in SOLVENT:
                counts[resn] = counts.get(resn, 0) + 1
    return sorted(counts, key=counts.get, reverse=True)


def extract(pdb_path: Path, resname: str | None, smiles: str | None, out: Path,
            name: str | None) -> None:
    Chem = _load_rdkit()

    if resname is None:
        cands = _ligand_resnames(pdb_path)
        if not cands:
            sys.exit("No non-solvent HETATM residue found; pass --resname explicitly.")
        resname = cands[0]
        if len(cands) > 1:
            print(f"[extract] auto-picked ligand resname '{resname}' "
                  f"(other candidates: {cands[1:]})")

    # Pull only the ligand's HETATM lines into a mini-PDB, preserving coordinates.
    lig_lines = [ln for ln in pdb_path.read_text().splitlines()
                 if ln.startswith(("HETATM", "ATOM")) and ln[17:20].strip() == resname]
    if not lig_lines:
        sys.exit(f"No atoms with resname '{resname}' in {pdb_path}")
    block = "\n".join(lig_lines) + "\nEND\n"

    lig = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False, proximityBonding=True)
    if lig is None:
        sys.exit(f"RDKit could not parse ligand '{resname}' from PDB.")

    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            sys.exit(f"Could not parse --smiles: {smiles!r}")
        try:
            lig = Chem.AssignBondOrdersFromTemplate(template, lig)
        except ValueError as exc:
            sys.exit(f"Bond-order assignment failed ({exc}).\n"
                     "  Check: does the SMILES match the ligand? Are all heavy atoms present "
                     "in the PDB (no missing/altloc atoms)? Try stripping Hs from the SMILES.")
        Chem.SanitizeMol(lig)
    else:
        print("[extract] WARNING: no --smiles given; bond orders are geometry-perceived and "
              "likely wrong. Supply --smiles for reliable featurization.")
        try:
            Chem.SanitizeMol(lig)
        except Exception as exc:  # noqa: BLE001
            print(f"[extract] sanitize warning: {exc}")

    lig.SetProp("_Name", name or f"{pdb_path.stem}_{resname}")
    out.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(out))
    w.write(lig)
    w.close()
    n_heavy = lig.GetNumHeavyAtoms()
    print(f"[extract] wrote {out}  ({n_heavy} heavy atoms, resname {resname})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", help="complex PDB containing protein + docked ligand")
    ap.add_argument("--out", required=True, help="output .sdf path")
    ap.add_argument("--resname", default=None, help="ligand 3-letter PDB resname (auto if omitted)")
    ap.add_argument("--smiles", default=None, help="ligand SMILES for bond-order assignment (strongly recommended)")
    ap.add_argument("--name", default=None, help="molecule title written into the SDF")
    args = ap.parse_args()
    extract(Path(args.pdb), args.resname, args.smiles, Path(args.out), args.name)


if __name__ == "__main__":
    main()
