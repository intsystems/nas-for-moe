# cifar100AndSVHN_v2 — SGEM with NLL-surrogate

This is a v2 fork of the SGEM pipeline (`code/cifar100/cifar100_sgem.py`) where
the surrogate predicts **mean per-sample cross-entropy loss** (`val_loss`)
instead of `val_accuracy`. The cluster-aware MoE objective becomes

    L(alpha, R) = sum_m |C_m| * log( sum_k r_mk * exp(-u(alpha_k, R_k)) )

so the per-sample likelihood entering the mixture is `exp(-u)`, and the
sum over clusters is **weighted by cluster size** `|C_m|`.

## EM updates (implemented in `optimize_surrogate_em_v2.py`)

- **E-step.** `q_{mk} \propto r_{mk} * exp(-u_k)`, normalised over `k`
  (computed via log-sum-exp for numerical stability).
- **M-step (r).** Gradient ascent on logits with Gumbel-Softmax samples
  of `R_k`; the q-function adds `c_m * q_{mk} * log(r_{mk})` and
  `c_m * q_{mk} * (-u_k)` (no `log u` because `u` is already a loss /
  log-likelihood quantity).
- **M-step (alpha).** Per expert, sample candidates and accept the one
  with the **lowest** mean predicted `u` (lower predicted loss is better).
- **Active learning.** Acquisition is **LCB = mu - sigma** (instead of
  UCB = mu + sigma in v1): lower predicted loss + larger uncertainty
  ⇒ more promising.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | empty; makes the directory a package |
| `README.md` | this file |
| `utils_v2.py` | `compute_log_likelihood_loss`, `save_observation_v2`, `read_obs_value` |
| `toy_dataset_v2.py` | `ArchSubsetLossDataset` — PyG dataset reading `val_loss` |
| `collect_dataset_v2.py` | thin shim re-exporting v1 utils + v2 `save_observation` |
| `optimize_surrogate_em_v2.py` | EM core adapted for NLL-surrogate (forked from `toy_experiment/optimize_surrogate_em.py`) |
| `cifar100_sgem_v2.py` | main entry; forks `cifar100/cifar100_sgem.py` and patches the evaluator to return `val_loss` |
| `run_v2.sh` | smoke-test launcher |

The original v1 modules are reused **unchanged via import**:
- `cifar100/cifar100_searchspace.py`
- `cifar100/cifar100_final_train.py`
- `cifar100/eval_sgem_per_expert.py`
- everything in `toy_experiment/` (we just override symbols on
  `collect_dataset` at module load time).

## How to run

```bash
# default: GPU 0, seed=322, small budget (smoke test)
./run_v2.sh

# explicit seed & GPU
GPU=1 ./run_v2.sh 42
```

For a full run, increase `--n-em-iterations`, `--n-seed-observations`,
`--cell-train-epochs`, and re-enable `--final-moe-epochs` (default in
`run_v2.sh` is 0 for the smoke test).

## Observation JSON format (v2)

```json
{
  "arch": {...},
  "subset_b": [1, 0, ...],
  "val_loss": 3.4521,
  "val_accuracy": 0.1832
}
```

`val_accuracy` is kept only for diagnostic purposes; the v2 surrogate is
trained on `val_loss`.

## Notes

- The variable name `val_acc` inside `optimize_surrogate_em_v2.py` is kept
  to minimise diff against v1; it actually holds the **v2 loss** returned
  by the patched evaluator. JSON persistence renames it to `val_loss`.
- The data directory defaults to
  `../cifar100/cifar100_svhn_data_semantic_testsplit` (combined CIFAR-100
  + SVHN, 30 clusters, 110 classes).
- Default `--K=2` (one expert per dataset).
