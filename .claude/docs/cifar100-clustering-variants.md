# CIFAR-100: варианты кластеризации и расположение данных

В `code/cifar100/` сосуществуют **два варианта кластеризации** датасета
(random/KMeans и semantic) × **две версии layout'а** (старая `v1` без test,
новая `_testsplit` с отложенным test). Текущая работа идёт с
**semantic + _testsplit**.

## Протокол оценки (инвариант — действует для всех методов: random/DARTS/SGEM)

**Подбор/поиск архитектур идёт на валидационной выборке. Финальная
найденная архитектура обучается заново и тестируется на отложенной
тестовой выборке — единственный раз, без подглядывания по эпохам.**

Конкретно:
- **этап поиска** — обучение на `train`, выбор лучшей архитектуры/конфигурации
  по `val` (random search 30ep → best by val_acc; DARTS supernet: веса W ←
  train, α ← val; SGEM: суррогат обучается на obs, где архитектуры тренируются
  на подмножествах `train`, а таргет = их `val`-точность). **`test` на этом
  этапе НЕ используется нигде.**
- **финальная модель** — обучение на `train ∪ val` фиксированное число эпох
  (обычно 100), затем одна оценка на `test`. Метрика результата — `test_acc`
  (ключ `final_moe.*` в JSON). Per-epoch в test не подглядываем.

Это требует наличия отдельного `test`-сплита → см. версию `_testsplit` ниже.

## Версии layout'а

| Версия | Директории данных | Прогоны | Особенности |
|---|---|---|---|
| **v1 (старая)** | `cifar100_data/`, `cifar100_data_semantic/` | `code/cifar100/runs/` | split train/val (80/20), test нет; метрика `val_acc`; есть собранные `cifar100_sgem_obs_*` |
| **_testsplit (новая, с 2026-05-12)** | `cifar100_data_testsplit/`, `cifar100_data_semantic_testsplit/` | `code/cifar100/runs_testsplit/` | split train/val/test (0.7/0.15/0.15), отложенный test; метрика финала `test_acc`; cluster IDs другие → старые obs несовместимы |

Старые v1-артефакты **не трогаем** — лежат по прежним путям. Новые
эксперименты пишутся **только** в `*_testsplit/` директории и
`runs_testsplit/`, чтобы не перемешивались со старыми.

Дефолты `--data-dir` / `--save-results` / `--save-dir` / `--output-dir`
в скриптах `code/cifar100/` уже переключены на `_testsplit`-пути
(semantic-вариант), так что обычно достаточно `python cifar100_sgem.py
--device cuda:0` без указания путей. Скрипты с `required=True` для
`--data-dir` (`cifar100_darts_search.py`, `cifar100_finetune_arch.py`,
`cifar100_final_train.py`, `eval_sgem_per_expert.py`) путь по-прежнему
требуют явно — передавайте `./cifar100_data_semantic_testsplit`.

## Варианты кластеризации

| Вариант | Подготовка | M (clusters) | v1-dir | _testsplit-dir |
|---|---|---|---|---|
| Random / KMeans (PCA на пикселях) | `prepare_cifar100.py` | 30 | `cifar100_data/` | `cifar100_data_testsplit/` |
| Semantic (ResNet-50 эмбеддинги) | `prepare_cifar100_semantic.py` | 30 | `cifar100_data_semantic/` | `cifar100_data_semantic_testsplit/` |

Содержимое директорий идентично по структуре: `data_X.npy`, `data_y.npy`,
`train_indices.npy`, `val_indices.npy`, `test_indices.npy`,
`train_cluster_ids.npy`, `val_cluster_ids.npy`, `test_cluster_ids.npy`,
`cluster_centers.npy`, `meta.json` (pixel-вариант дополнительно содержит
`pca_components.npy`, `pca_mean.npy`). В v1-директориях файлов
`test_indices.npy` / `test_cluster_ids.npy` **нет** — поэтому новый код
(`load_cifar100_meta`) на них упадёт с понятной ошибкой; используйте
`_testsplit`-директории.

Сгенерировать новую версию:
```bash
# в docker-контейнере, из code/cifar100/
python prepare_cifar100_semantic.py --output-dir ./cifar100_data_semantic_testsplit \
    --fraction 0.7 --n-clusters 30 --val-size 0.15 --test-size 0.15 --seed 322 --device cuda:0
python prepare_cifar100.py --output-dir ./cifar100_data_testsplit \
    --fraction 0.7 --n-clusters 30 --val-size 0.15 --test-size 0.15 --seed 322
```

## Микс-датасет CIFAR-100 + SVHN (`prepare_cifar100_svhn.py`, с 2026-05-12)

Кастомный датасет = смесь CIFAR-100 и SVHN в заданной пропорции
(`--cifar-frac`, дефолт 0.5/0.5; размер микса `--total-samples`, дефолт 42000).
Цель — проверить, выделит ли SGEM **разбиение кластеров по источнику данных**
(один эксперт → CIFAR-100, другой → SVHN). SVHN — максимально далёкий домен,
semantic-кластеры (ResNet-50 эмбеддинги) почти не смешивают источники в один
кластер → «идеальное разбиение» чётко определено.

- Метки SVHN сдвинуты на **+100** → классы `0..99` = CIFAR-100, `100..109` = SVHN,
  `num_classes = 110` (берётся из `meta.json`, downstream-скрипты вроде
  `cifar100_sgem.py` подхватывают автоматически).
- Layout как у `_testsplit`-семантического (3-way split, `cluster_centers` в
  2048-d пространстве эмбеддингов) + доп. файл **`source_ids.npy`** (`[N]` int64,
  0 = CIFAR, 1 = SVHN).
- `meta.json` содержит `ideal_split_by_source` (списки CIFAR-/SVHN-мажоритарных
  кластеров и их purity) — это ожидаемый оптимум при K=2. Скрипт печатает
  состав каждого кластера по источнику и warning, если средняя purity < 0.9.
- Дефолтный `--output-dir`: `./cifar100_svhn_data_semantic_testsplit`.

```bash
# в docker-контейнере, из code/cifar100/
python prepare_cifar100_svhn.py --output-dir ./cifar100_svhn_data_semantic_testsplit \
    --total-samples 42000 --cifar-frac 0.5 --n-clusters 30 \
    --val-size 0.15 --test-size 0.15 --seed 322 --device cuda:0
```

## Трёхчастный split (с 2026-05-12)

Данные делятся на **train / val / test** (по умолчанию 0.7/0.15/0.15 от
`fraction`-подвыборки, см. `prepare_cifar100*.py --val-size --test-size`).
KMeans-кластеры обучаются **только на train-эмбеддингах**; val и test
размечаются ближайшим центроидом — test нигде не участвует в кластеризации
или поиске архитектур.

Использование выборок в пайплайнах:
- **поиск** (random search 30ep / DARTS supernet / сбор obs суррогата SGEM):
  обучение на `train`, оценка/выбор по `val`. У DARTS веса W ← train,
  α ← val (раньше был 50/50 split самого train).
- **финальная модель** (`--final-epochs`/`--final`): обучение на `train ∪ val`,
  единственная оценка на `test` после последней эпохи (без подглядывания).
  Метрика в JSON — `test_acc` (а не `val_acc`).

> **Важно:** перегенерация данных под 3-way split меняет cluster IDs →
> все ранее собранные obs-датасеты (`cifar100_sgem_obs_*`) становятся
> несовместимыми, их нельзя переиспользовать через `--initial-obs-dir`.
> Аналогично, старые `results_*` с `val_acc` несопоставимы с новыми `test_acc`.

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
