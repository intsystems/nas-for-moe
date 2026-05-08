# Toy Experiment — Pipeline

Полный пайплайн от генерации данных до оптимизации MoE.

## Обзор

```
toy_generate_dataset*.py  →  toy_clustering.py  →  collect_dataset.py  →  retrain_surrogate.py  →  optimize_*.py
       (1)                        (2)                    (3)                     (4)                    (5)
```

---

## Шаг 1: Генерация данных

Генерация синтетических 2D-данных для бинарной классификации.

| Скрипт | Данные | Output dir по умолчанию |
|---|---|---|
| `toy_generate_dataset.py` | 1 linear cloud + 1 ring cloud (balanced) | `data/` |
| `toy_generate_dataset_multi.py` | 10 linear + 10 ring clouds (на сетке) | `data_multi/` |
| `toy_generate_dataset_imbalanced.py` | 1 linear (200) + 1 ring (1800) | `data_imbalanced/` |

**Выход:** `data_X.npy` (features [N, 2]), `data_y.npy` (labels [N]), `dataset.png` (визуализация).

**Пример:**
```bash
docker exec nas-for-moe python /pbabkin/main/mipt/nas-for-moe/code/toy_experiment/toy_generate_dataset_multi.py \
    --n-linear 10 --n-ring 10 --n-samples 200 --seed 322
```

---

## Шаг 2: Кластеризация

`toy_clustering.py` — KMeans кластеризация с train/val split.

**Вход:** `data_X.npy`, `data_y.npy` из шага 1.

**Алгоритм:**
1. Train/val split (по умолчанию 80/20).
2. KMeans на train-данных.
3. Валидационные точки назначаются кластеру по ближайшему центроиду.

**Выход (в ту же директорию):**
- `train_indices.npy`, `val_indices.npy` — индексы split
- `train_cluster_ids.npy`, `val_cluster_ids.npy` — метки кластеров
- `cluster_centers.npy` — центроиды KMeans
- `clusters.png` — визуализация

**Пример:**
```bash
docker exec nas-for-moe python /pbabkin/main/mipt/nas-for-moe/code/toy_experiment/toy_clustering.py \
    --data-dir ./data_multi --n-clusters 20 --seed 322
```

---

## Шаг 3: Сбор датасета для суррогата

`collect_dataset.py` — active learning loop: обучение архитектур на подмножествах кластеров, сбор пар `(alpha, b) -> val_accuracy`.

**Вход:**
- Данные и кластеризация из шагов 1-2 (`--data-dir`).
- Пространство поиска архитектур (`toy_searchspace.py`).

**Алгоритм:**
1. Seed-датасет: случайные пары (архитектура, бинарный вектор `b`).
2. Обучение реальных моделей, измерение val accuracy.
3. Обучение суррогата на собранном датасете.
4. Генерация кандидатов, оценка через MC Dropout (UCB = mu + sigma).
5. Выбор лучшего кандидата, реальное обучение, добавление в датасет.
6. Повтор с шага 3.

**Выход:** директория с `obs_*.json` файлами (каждый — одно наблюдение: config, bool_vector, val_accuracy) и `surrogate_*.pth` (чекпоинт суррогата).

**Пример:**
```bash
docker exec nas-for-moe python /pbabkin/main/mipt/nas-for-moe/code/toy_experiment/collect_dataset.py \
    --data-dir ./data_multi --save-dir ./model_dataset_1
```

---

## Шаг 4: Переобучение суррогата

`retrain_surrogate.py` — обучение суррогатной функции на всём собранном датасете с нуля (grid search по гиперпараметрам).

**Вход:**
- `obs_*.json` из шага 3 (`obs_dir` в коде, по умолчанию `./model_dataset_1`).
- `cluster_centers.npy` из шага 2 (для определения M).

**Алгоритм:**
1. Загрузка всех наблюдений.
2. Перебор нескольких конфигураций (dropout, hidden_dim, heads, lr, epochs, patience).
3. Для каждой: train/val split, обучение, оценка (R2, Spearman, MAE, RMSE).
4. Сохранение лучшей модели.

**Выход:** `surrogate_k{M}_n{N}_best.pth` — лучший чекпоинт суррогата.

**Пример:**
```bash
docker exec nas-for-moe python /pbabkin/main/mipt/nas-for-moe/code/toy_experiment/retrain_surrogate.py
```

---

## Шаг 5: Оптимизация назначения экспертов

Три метода оптимизации MoE log-likelihood: `∏_m Σ_k r_mk · u(α_k, R_k) → max`.

Все три метода используют общие утилиты из `optimize_expert_assignments.py`.

### Метод 1: Sampling Search (`optimize_sampling.py`)

- Сэмплирование жёстких назначений кластеров → подбор архитектур → выбор лучшего.
- Суррогат вызывается на дискретных бинарных `R_k`.
- Простой, без аппроксимаций.

### Метод 2: Gradient Optimization (`optimize_gradient.py`)

- Параметризация `r` через logits + softmax.
- Gumbel-Softmax (straight-through) для передачи `R_k` в суррогат.
- Архитектуры пересэмплируются каждые N шагов.
- Temperature annealing: `tau` снижается от `tau_start` до `tau_end`.

### Метод 3: EM Algorithm (`optimize_em.py`)

- **E-step:** `q_mk ∝ r_mk · u(α_k, R_k)`, нормализация по k.
- **M-step (r):** градиентная оптимизация logits с Gumbel-Softmax.
- **M-step (α):** сэмплирование архитектур.

**Общие аргументы всех методов:**
- `--surrogate-path` — путь к `.pth` суррогата
- `--data-dir` — директория с данными
- `--cluster-dir` — директория с кластеризацией (по умолчанию = data-dir)
- `--M` — число кластеров, `--K` — число экспертов
- `--n-arch-candidates` — число кандидатов архитектур при сэмплировании
- `--save-results` — путь для сохранения результатов в JSON

**Выход:** `OptimizationResult` — лучшие архитектуры `α_1..K`, матрица `r [M,K]`, дискретное назначение, objective, история. Также реальная оценка (переобучение на кластерах).

**Пример:**
```bash
docker exec nas-for-moe python /pbabkin/main/mipt/nas-for-moe/code/toy_experiment/optimize_sampling.py \
    --surrogate-path ./model_dataset_1/surrogate_k20_n100_best.pth \
    --data-dir ./data_multi --M 20 --K 3 --save-results results_sampling.json
```

---

## Типичный полный запуск

```bash
# 1. Генерация данных
python toy_generate_dataset_multi.py --output-dir ./data_multi

# 2. Кластеризация
python toy_clustering.py --data-dir ./data_multi --n-clusters 20

# 3. Сбор датасета (active learning)
python collect_dataset.py --data-dir ./data_multi --save-dir ./model_dataset_1

# 4. Переобучение суррогата (опционально, если нужен лучший суррогат)
python retrain_surrogate.py

# 5. Оптимизация (три метода)
python optimize_sampling.py --surrogate-path ./model_dataset_1/surrogate_best.pth --data-dir ./data_multi --M 20 --K 3
python optimize_gradient.py --surrogate-path ./model_dataset_1/surrogate_best.pth --data-dir ./data_multi --M 20 --K 3
python optimize_em.py --surrogate-path ./model_dataset_1/surrogate_best.pth --data-dir ./data_multi --M 20 --K 3
```
