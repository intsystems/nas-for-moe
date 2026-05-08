# CIFAR-100: варианты кластеризации и расположение данных

В `code/cifar100/` сосуществуют **два варианта кластеризации** датасета.
Текущая работа идёт с **semantic**.

## Варианты

| Вариант | Директория данных | Подготовка | M (clusters) |
|---|---|---|---|
| Random / KMeans | `cifar100_data/` | `prepare_cifar100.py` | 30 |
| Semantic | `cifar100_data_semantic/` | `prepare_cifar100_semantic.py` | 30 |

Содержимое директорий идентично по структуре: `data_X.npy`, `data_y.npy`,
`train_indices.npy`, `val_indices.npy`, `train_cluster_ids.npy`,
`val_cluster_ids.npy`, `cluster_centers.npy`, `meta.json`.

## Имена прогонов в `runs/`

- `*_long_semantic*` / `*_semantic*` → semantic clustering
- остальные (`lb005`, `lb030`, `lb1000_mix050`, …) → random clustering

## Сохранённые obs-датасеты (для `--initial-obs-dir`)

В `code/cifar100/runs/`:

| Директория | N obs | Кластеризация | Особенности |
|---|---|---|---|
| `cifar100_sgem_obs_K3_lb5000_long_semantic` | ~1050 | semantic | pc-mix=0 |
| `cifar100_sgem_obs_K3_lb5000_long_semantic_pc05` | ~1391 | semantic | pc-mix=0.5 |
| `cifar100_sgem_obs_K3_lb10000_long_semantic` | ~1050 | semantic | pc-mix=0 |
| `cifar100_sgem_obs_K3_lb15000_long_semantic` | ~1050 | semantic | pc-mix=0 |

`*.pth` чекпоинтов суррогата в этих директориях **нет** —
`--initial-surrogate-path` использовать неоткуда, при загрузке obs суррогат
переобучается с нуля.

## Конвенция имён прогонов

- `K{n}` — число экспертов
- `lb{NNN}` в **старых** прогонах (mix050-серия) ≈ `--load-balance-weight`
  (`lb000`=0.0, `lb150`=0.15, `lb500`=0.5, `lb1000`=1.0).
- `lb{NNNN}_long_semantic` в новых (5000/10000/15000) — **не** load-balance-weight,
  параметр неизвестен (значение args в JSON не сохраняется — баг).
- `pc05` — `--phase-c-uniform-mix 0.5`
- `lbw{X}` — load-balance-weight (введено в новой серии, например `pc05_lbw1`)

## Известная проблема

`cifar100_sgem.py` **не сохраняет `vars(args)`** в результирующий JSON
(`results_*_sgem*.json` секция `cifar100_sgem` содержит только `configs`,
`r_matrix`, `hard_assignments`, `objective_value`, `history`, `phase_a_history`,
`final_moe`). Это делает невозможным восстановление гиперпараметров
завершённых прогонов после факта.
