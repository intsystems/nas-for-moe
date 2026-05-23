# NAS for MoE

Research project on Neural Architecture Search (NAS) for Mixture-of-Experts (MoE) models.

## Project Structure

Now I am working with `code/toy_experiment` folder

### Pipeline scripts (in execution order)

1. **Data generation** — `toy_generate_dataset.py` / `toy_generate_dataset_multi.py` / `toy_generate_dataset_imbalanced.py`
2. **Clustering** — `toy_clustering.py`
3. **Surrogate dataset collection** — `collect_dataset.py`
4. **Surrogate retraining** — `retrain_surrogate.py`
5. **MoE optimization** — `optimize_sampling.py` / `optimize_gradient.py` / `optimize_em.py`

See @docs/pipeline.md for full pipeline description.

### Support modules

- `optimize_expert_assignments.py` — shared utilities for optimization methods (OptimizationResult, graph/surrogate helpers, CLI args)
- `toy_dataset.py` — PyG dataset for surrogate function
- `toy_graph.py` — graph encoding of architectures
- `toy_searchspace.py` — search space definition

### Other

- `code/nas_moe/` — core library (dataset, graph, VAE, surrogate, MoE arch)
- `code/*.ipynb` — legacy experiment notebooks (do not modify)
- `docker/` — Dockerfile and launch script

## Environment

Working directly on the GPU server. See:
- @docs/server.md — server specs (Jarvis, main GPU machine)
- @docs/rsm.md — альтернативные серверы rsm3/4/5/6 (подключение через `ssh rsm6`)
- @docs/docker.md — container setup and usage
- @docs/pipeline.md — полный пайплайн toy_experiment (порядок запуска, входы/выходы)
- @docs/toy-experiment-context.md — общий контекст toy_experiment (суррогат, кластеры, MoE)
- @docs/surrogate.md — суррогатная функция: вход/выход, MC Dropout, active learning
- @docs/theory.md — математическая постановка и методы оптимизации MoE
- @docs/data-directory.md — описание датасета в `data/` (параметры генерации, кластеризация, файлы)
- @docs/cifar100-clustering-variants.md — варианты кластеризации CIFAR-100 (random vs semantic), версии layout'а (v1 / `_testsplit`), **протокол оценки: поиск архитектур на val, финал — обучение на train∪val + тест на отложенном test (`test_acc`)**, конвенция имён прогонов, сохранённые obs-датасеты
- @docs/experiments-log.md — журнал экспериментов CIFAR-100 (val_acc по методам, ключевые выводы, TODO)

**All experiments must run inside the Docker container.** Before running any `docker exec`, ensure the container is running — if not, start it with `docker start nas-for-moe`. See @docs/docker.md for details.

```bash
docker start nas-for-moe          # ensure container is running
docker exec nas-for-moe <command> # run commands inside
```

## Code Style

- Python 3, follow PEP 8
- 4-space indentation
- Type hints on function signatures when not obvious
- Keep modules focused

## Key Modules

- `vae.py` — VAE for architecture encoding
- `surrogate.py` — surrogate model predicting arch performance
- `moe_arch.py` — MoE architecture definition
- `graph.py` — graph representation of architectures
- `nni_utils.py` — NNI integration for NAS


## Conventions

- Trained models saved as `.pth` files (e.g. `best_vae_model.pth`)
- Do not commit model checkpoints
- Do not commit data or datasets


# Pay Attention

- If you have any questions or ambiguities, please, ask me.

use context7

Записывай в свои конфигурационные файлы (папка .claude) важные архитектурны решения, например, где сохранены два разных варианта кластеризации. Также трекируй результаты экспериментов в каком-то виде. Делай и сохраняй какие-то выводы.
 