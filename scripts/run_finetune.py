#!/usr/bin/env python
"""Few-shot finetune PBCNet2.0 on a target, then (optionally) score a held-out set.

This is the portable analogue of upstream ``model_code/Finetune.py``. Upstream hardwires
the FEP ``finetune_input`` reference-split machinery and ``cuda:1``; here we take plain
pairs CSVs and pick the device automatically, so the same command runs on the Mac (CPU,
fine for a handful of reference ligands) or the cluster (CUDA).

Inputs (same "pairs" schema as run_benchmark.py):
    --finetune-csv   pairs to train on   (must include exp_ddg)
    --eval-csv       pairs to score after finetuning (optional; needs exp_ddg for metrics)

Outputs:
    <outdir>/finetuned_<name>.pth    the adapted model
    <outdir>/predictions_<name>.csv  scores on --eval-csv (long schema)
    <outdir>/finetune_log.csv        per-epoch train loss (+ eval metrics if given)

Upstream finetuning defaults we mirror: Adam(lr=1e-5), MSE on ΔΔG, ~10 epochs, small
batches. Because the model was pickled whole, we finetune the loaded object directly.

Needs the full env (torch + dgl).
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


def _batches(df: pd.DataFrame, data_root: Path, batch_size: int, shuffle: bool, seed: int):
    import dgl
    import torch

    rows = df.sample(frac=1.0, random_state=seed) if shuffle else df
    rows = rows.reset_index(drop=True)
    for start in range(0, len(rows), batch_size):
        chunk = rows.iloc[start:start + batch_size]
        g1s, g2s, y = [], [], []
        for _, r in chunk.iterrows():
            try:
                with open(_resolve(r.ligand1_pkl, data_root), "rb") as fh:
                    g1s.append(pickle.load(fh))
                with open(_resolve(r.ligand2_pkl, data_root), "rb") as fh:
                    g2s.append(pickle.load(fh))
                y.append(float(r.exp_ddg))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip row: {exc}")
        if not g1s:
            continue
        yield dgl.batch(g1s), dgl.batch(g2s), torch.tensor(y, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finetune-csv", required=True)
    ap.add_argument("--eval-csv", default=None)
    ap.add_argument("--name", default="model", help="tag for output filenames")
    ap.add_argument("--outdir", default="results/finetune")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model, device = load_model(args.checkpoint, args.device)
    ft = pd.read_csv(args.finetune_csv)
    if "exp_ddg" not in ft.columns:
        ap.error("--finetune-csv must include an exp_ddg column")
    print(f"[pbcnet] finetuning '{args.name}' on {device} "
          f"({len(ft)} pairs, {args.epochs} epochs, lr={args.lr})")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    log = []

    for epoch in range(args.epochs):
        model.train()
        total, nb = 0.0, 0
        for g1, g2, y in _batches(ft, data_root, args.batch_size, True, args.seed + epoch):
            g1, g2, y = g1.to(device), g2.to(device), y.to(device)
            out, out_neg = model(g1, g2)
            loss = loss_fn(out.squeeze(-1).float(), y) + \
                loss_fn(out_neg.squeeze(-1).float(), torch.neg(y))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        row = {"epoch": epoch, "train_loss": total / max(nb, 1)}
        print(f"  epoch {epoch:2d}  loss {row['train_loss']:.4f}")
        log.append(row)

    ckpt_out = outdir / f"finetuned_{args.name}.pth"
    torch.save(model, ckpt_out)
    print(f"Saved finetuned model -> {ckpt_out}")

    if args.eval_csv:
        # reuse the benchmark scorer for a consistent predictions CSV
        from run_benchmark import run  # sibling script
        ev = pd.read_csv(args.eval_csv)
        model.eval()
        pred = run(ev, model, device, data_root, args.batch_size)
        pred_out = outdir / f"predictions_{args.name}.csv"
        pred.to_csv(pred_out, index=False)
        print(f"Wrote eval predictions -> {pred_out}")
        if pred["exp_ddg"].notna().any():
            from pbcnet_workflow import metrics
            m = metrics.compute(pred["pred_ddg"].values, pred["exp_ddg"].values)
            log[-1].update({"eval_rmse": m.rmse, "eval_spearman": m.spearman})

    pd.DataFrame(log).to_csv(outdir / "finetune_log.csv", index=False)


if __name__ == "__main__":
    main()
