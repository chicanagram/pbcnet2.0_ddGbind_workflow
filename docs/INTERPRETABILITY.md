# LLM-aided interpretability of PBCNet2.0 features (plan — no code yet)

Goal: use the model's internal representations to surface *which physicochemical /
structural features drive a predicted ΔΔG*, with special attention to **mutation
effects** (why a pocket mutation changes a ligand's relative affinity), then hand a
compact, structured summary to an LLM to propose mechanistic hypotheses.

This document is the design. Nothing here is implemented yet; it defines exactly which
tensors to pull from the network and how to turn them into LLM-ready evidence.

---

## 1. What the model actually computes (so we extract the right thing)

Recap of the architecture (`external/PBCNet2.0/model_code/models/`):

```
complex graph  ──►  TensorEmbedding  ──►  Interaction × num_layer  ──►  per-atom scalar emb x[N, H]
(atoms=nodes,        (initial Cartesian    (equivariant tensor msg
 dist/bond edges)     tensor X[N,H,3,3])    passing; returns edge_attr)
                                                   │
                       per-atom X decomposed into irreps  I (scalar), A (vector), S (sym-traceless)
                                                   │
   readout:  mask out protein atoms ─► sum ligand-atom emb ─► emb_mol[H]   (one vector per ligand)
                                                   │
   Siamese head:  ΔΔG = FNN( LayerNorm( emb_mol(target) − emb_mol(reference) ) )
```

Key consequences for interpretability:

- The prediction is driven by the **difference of two pooled ligand embeddings**,
  `emb_target − emb_reference` (`readout.py:91`). That difference vector `[H]` is the
  single most direct handle on "what changed."
- Protein atoms are **masked at readout** (`readout.py:74-75`), so the pooled vector is
  a *ligand-atom* sum — **but** ligand-atom embeddings already encode the pocket, because
  message passing runs over protein–ligand edges (≤5 Å) built in `Graph2pickle.py`. So a
  pocket mutation reaches the prediction *through* changes in ligand-atom embeddings and
  in the protein→ligand edge messages.
- Each atom carries an **irreducible tensor decomposition** — `I` (isotropic/scalar,
  charge-like), `A` (antisymmetric/vector, directional/dipole-like), `S` (symmetric
  traceless, anisotropy/steric-directional). Attributing an effect to `‖I‖` vs `‖A‖` vs
  `‖S‖` is chemically meaningful.
- Message passing returns a per-edge `edge_attr` (`tensornet.py:490`, collected as `ATT`
  in `TensorNet.forward`). It is **not softmax attention** — it's the learned scalar
  weight multiplying the I/A/S messages on each edge — but its magnitude is a usable
  proxy for **per-contact importance**, and edges include protein–ligand contacts.

---

## 2. Features to extract (the "probes")

Implemented later as forward hooks on the vendored model (no upstream edits needed).

| # | Probe | Where | Tensor / shape | Interpretation |
|---|-------|-------|----------------|----------------|
| P1 | Per-atom scalar embedding | `readout._readout`, `g.nodes['atom'].data['emb']` | `[N_atoms, H]` | per-atom contribution to the pooled vector |
| P2 | Per-atom irrep norms per layer | decompose `X` after each `Interaction` | 3 × `[N_atoms, H]` (‖I‖,‖A‖,‖S‖) | scalar vs directional vs steric character |
| P3 | Edge messages | `TensorNet.forward` `ATT[layer]` | `[N_edges, H, 3]` on `int` edges | per-contact (incl. protein–ligand) importance |
| P4 | Pooled ligand embedding | `emb_mol` | `[H]` per ligand | ligand-level fingerprint |
| P5 | Head difference vector | `norm(emb1 − emb2)` | `[H]` per pair | the direct ΔΔG determinant |
| P6 | Gradient attributions | autograd on P1/P5 or on atom `pos` | `[N_atoms]` | signed sensitivity of ΔΔG to each atom / position |

Edge→atom mapping (for P3) comes from `g.edges('int')` (src,dst index arrays); atom→
(ligand vs pocket) comes from `g.nodes['atom'].data['type']` (1=ligand, 0=pocket) and the
residue bookkeeping already computed in `Graph2pickle.py` (`res_idx`, `res_type`).

---

## 3. From tensors to interpretable descriptors

Raw `[N, H]` activations are not human-meaningful. Two bridges:

**(a) Attribution → atoms/contacts → chemistry.**
- Integrated Gradients (or gradient×activation) of ΔΔG w.r.t. P1/P6 gives a signed
  per-atom score. Aggregate ligand atoms to **RDKit substructures / R-groups** (the BRICS
  decomposition already exists in `Graph2pickle.brics_decomp`) → "R-group X at position Y
  contributes +0.6 kcal/mol."
- For P3, rank `int` edges by |message|; keep protein–ligand edges; map the protein atom
  back to its **residue** (via `res_idx`/`res_type`) → "contact to residue LEU858 dominates."

**(b) Latent dimension → descriptor correlation.**
- Compute per-ligand RDKit/interaction descriptors (logP, TPSA, HBD/HBA, formal charge,
  aromatic-ring count, MW, plus counts of H-bond / hydrophobic / π contacts from the graph).
- Correlate each of the `H` embedding dimensions (P4/P5) against these descriptors across
  a dataset → a "dimension ↔ descriptor" dictionary. Then any single prediction's
  difference vector P5 can be read as "moved along the HBD axis and the aromatic axis."

---

## 4. Mutation-effect workflow (the headline use case)

The bundled `data/Mutation/` set (UniProt IDs incl. EGFR **P00533**) is exactly this
shape: same ligand(s) against **wild-type vs mutant** pockets.

For a (ligand, WT-pocket, mutant-pocket) triple:
1. Build both complex graphs (`preprocess.py`).
2. Score both; the model's pocket enters via protein–ligand edges.
3. Diff the probes **WT vs mutant**: ΔP1 (per-atom), ΔP2 (which irrep moved), ΔP3
   (which contacts strengthened/weakened), ΔP5 (head vector shift).
4. Localize the change to residues near the mutation site and to specific ligand R-groups.
5. Package as structured evidence (§5) → LLM hypothesis.

This directly answers "which features matter for this mutation's effect."

---

## 5. LLM stage (modular, backend pluggable — decision deferred)

The LLM never sees raw tensors — only a compact, structured **evidence JSON** per case:

```jsonc
{
  "system": "EGFR", "mutation": "L858R", "ligand": "gefitinib",
  "predicted_ddg": 1.8, "experimental_ddg": 2.1,
  "top_residue_contacts": [{"res": "L858R", "delta_importance": +0.44}, ...],
  "top_ligand_rgroups":  [{"smiles_frag": "c1ccncc1", "attribution": -0.6}, ...],
  "irrep_shift": {"scalar": 0.1, "vector": 0.5, "sym": 0.3},
  "latent_axes_moved": [{"axis": "HBD", "z": +2.1}, {"axis": "aromatic", "z": -1.3}]
}
```

- **Interface:** one `explain(evidence: dict) -> str` function behind a thin adapter, so the
  backend (Anthropic Claude API vs a local/open model via Ollama) is a config switch.
- **Prompt shape:** system role = "structural chemist"; give the evidence + a short schema
  legend; ask for (i) a mechanistic hypothesis, (ii) confidence, (iii) a testable follow-up
  (e.g. a suggested analog or point mutation).
- **Validation:** cross-check LLM claims against held-out experimental ΔΔG and against
  ablations (mask the residue/R-group the LLM fingered → does predicted ΔΔG move as claimed?).
  This closes the loop and guards against plausible-but-wrong narratives.

> ⚠️ **PRIVATE-data caveat.** Sending evidence to a hosted LLM API publishes it. If the
> structures/affinities are ARES PRIVATE, either (a) send only aggregated, de-identified
> descriptors, or (b) use the local-model backend. Decide before wiring a hosted backend.

---

## 6. Build order when we implement

1. `interpret/hooks.py` — forward hooks capturing P1–P5; grad utilities for P6.
2. `interpret/attribution.py` — IG + edge ranking + atom→R-group / atom→residue mapping.
3. `interpret/descriptors.py` — RDKit descriptors + latent-axis correlation dictionary.
4. `interpret/evidence.py` — assemble the evidence JSON (§5).
5. `interpret/llm.py` — pluggable `explain()` adapter (Claude API | local).
6. `interpret/ablation.py` — mask-and-rescore validation loop.
7. `notebooks/interpretability_demo.ipynb` — run the EGFR mutation case end-to-end.
