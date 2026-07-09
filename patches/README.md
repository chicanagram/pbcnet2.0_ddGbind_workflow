# Portability patches to vendored upstream

Strategy chosen: **vendor + patch**. The upstream tree lives under
`external/PBCNet2.0/`. We keep edits to it minimal and documented here; most portability
lives in our overlay (`src/pbcnet_workflow/`) so upstream stays close to original.

## Blockers found in original upstream (why patches are needed)

| File | Original | Problem | Handled by |
|------|----------|---------|------------|
| `model_code/train.py` | `cuda = "cuda:" + str(args.device)` | no CPU path | patched in-place (see below) + `device.resolve_device` |
| `model_code/train.py` | `torch.load('/code/PBCNet.pth')` (retrain) | absolute path, no `map_location` | patched in-place |
| `model_code/Finetune.py` | `cuda = "cuda:"+...`; `.to(cuda)` | no CPU path | our `run_finetune.py` supersedes it |
| `Graph2pickle.py` | `/home/user-home/...` path, `Pool(50)` in `__main__` | won't run as-is | our `preprocess.py` supersedes it (imports `Graph_Information`, ignores `__main__`) |
| README | refers to `PBCNet2.0.pth` | real file is `PBCNet2.pth` | `checkpoint.default_checkpoint()` |
| all modules | `code_path.rsplit('/', 1)` | POSIX-only path splitting | avoided by importing via `paths.add_model_code_to_path` on POSIX; on Windows use WSL2 |

## Patches actually applied to the tree

- `model_code/train.py`: device line made CPU-aware; retrain checkpoint load made
  repo-relative + `map_location`. Grep for `# [pbcnet-patch]` to find them.

Everything else is handled by the overlay without touching upstream, so re-vendoring a
newer upstream mainly means re-applying the single `train.py` device patch.
