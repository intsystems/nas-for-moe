"""v2 collect helpers — re-export v1 utilities, override save_observation.

The v1 collect_dataset module is large and most of its utilities are reused
as-is. The only thing that changes for v2 is the persistence format
(val_loss instead of val_accuracy), so we override save_observation.

`cifar100_sgem_v2.py` monkey-patches v1's `collect_dataset.save_observation`
and `collect_dataset.evaluate_architecture_on_subset` to v2 versions, so
imports inside optimize_surrogate_em_v2 still work.
"""
from toy_experiment.collect_dataset import (  # noqa: F401
    set_seed,
    evaluate_architecture_on_subset,
)
from utils_v2 import save_observation_v2 as save_observation  # noqa: F401
