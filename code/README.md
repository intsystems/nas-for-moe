# Code — NAS for MoE experiments

Код к проекту «Neural Architecture Search для Mixture-of-Experts». Метод —
**SAGEM**: суррогат-функция предсказывает val-точность архитектуры эксперта на
заданном подмножестве кластеров данных, а EM-алгоритм совместно подбирает
архитектуры экспертов `α_k` и матрицу маршрутизации кластеров `r`.

Все запуски выполняются **в Docker-контейнере** (см. `../.claude/docs/docker.md`).

## Структура

| Папка | Что внутри |
|---|---|
| `toy_experiment/` | Контролируемый 2D-эксперимент (linear + ring) |
| `cifar100/` | CIFAR-100 и гетерогенный микс **CIFAR-100 + SVHN** |
| `mnist/` | Вспомогательные MoE-эксперименты на MNIST |
| `nas_moe/` | Базовая библиотека (граф, суррогат, VAE, MoE-арх) |
| `data/` | Готовые датасеты (в git не коммитятся) |
| `runs/`, `cifar100/runs_testsplit/` | Результаты прогонов (`.json`, логи) — не коммитятся |

## toy_experiment

Пайплайн (порядок запуска), подробности — в `../.claude/docs/pipeline.md`:

1. `toy_generate_dataset.py` — генерация 2D-данных (4 класса: linear + ring).
2. `toy_clustering.py` — KMeans + train/val split.
3. `collect_dataset.py` — active-learning сбор датасета суррогата.
4. `optimize_surrogate_em.py` — **основной метод** (EM-оптимизация `α_k` и `r`).

Вспомогательное: `toy_searchspace.py`, `toy_graph.py`, `toy_dataset.py`,
`optimize_expert_assignments.py` (общие утилиты), `plot_*` (визуализация).

## cifar100 / CIFAR-100 + SVHN

Протокол оценки и варианты кластеризации — в
`../.claude/docs/cifar100-clustering-variants.md`. Кратко: поиск архитектур на
`train`→`val`, финальная модель обучается на `train ∪ val` и **один раз**
оценивается на отложенном `test` (метрика `test_acc`).

**Подготовка данных** (3-way split train/val/test, semantic-кластеры на
ResNet-50 эмбеддингах):
- `prepare_cifar100.py` — random/KMeans-кластеризация по пикселям.
- `prepare_cifar100_semantic.py` — semantic-кластеризация.
- `prepare_cifar100_svhn.py` — микс CIFAR-100 + SVHN (`source_ids.npy`,
  `ideal_split_by_source` в `meta.json`; ожидаемый оптимум при `K=2`).

**Методы и бейзлайны:**
- `cifar100_sgem.py` — SAGEM (наш метод). Флаги: `--phase-c-uniform-mix`,
  `--load-balance-weight`, `--n-r-gradient-steps 0` (ablation: фиксированная
  случайная `r`), `--final-moe-mode {learnable,cluster,both}`.
- `cifar100_random_moe_cluster.py` / `cifar100_random_moe_learnable.py` —
  Random-MoE с cluster- / learnable-гейтом.
- `cifar100_darts_search.py` + `cifar100_finetune_arch.py` — DARTS-поиск и
  retrain единичной архитектуры.
- `cifar100_darts_moe_baseline.py` — `K` копий одной DARTS-архитектуры
  (DARTS×K) с learnable-гейтом.
- `cifar100_random_single_arch_baseline.py` — случайная одиночная сеть.

**Оценка / визуализация:** `eval_sgem_per_expert.py`, `eval_surrogate.py`,
`plot_cifar100_clusters.py` (представители кластеров с разметкой экспертов;
`--max-clusters`, `--no-labels`), `plot_cifar100_svhn_semantic.py` (t-SNE).

**Драйвер-скрипты прогонов** `run_*_svhn*.sh` — как именно запускались
многосидовые эксперименты CIFAR-100 + SVHN на сервере rsm6 (через
`docker exec`). Пути в них абсолютные (`/pbabkin/...`) — это запись фактических
запусков, при переносе скорректировать под своё окружение.
