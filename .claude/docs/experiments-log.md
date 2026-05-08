# Experiments Log (CIFAR-100, semantic clustering)

Журнал результатов и выводов по экспериментам. Все на M=30, K=3,
init_channels=16, semantic-кластеризации, если не указано иначе.

## Сравнительная таблица: best val_acc

| Метод | Прогон / источник | val_acc | Заметки |
|---|---|---|---|
| Random MoE + cluster-gating (search 30ep) | `results_cifar100_random_moe_cluster_K3_semantic_n200.json` | 0.3832 | 200 кандидатов × 30 эпох; mean=0.367, std≈0.01 |
| **Random MoE + cluster-gating (retrain 100ep)** | `results_cifar100_random_moe_cluster_K3_semantic_n200_final100.json` | **0.4162** | retrain best (index 90, counts 10/8/12); +3.3 п.п. vs search; всё ещё ниже DARTS (0.4288) |
| Random MoE + learnable gating (search 30ep) | `results_cifar100_random_moe_K3_semantic_n200_final100.json` | 0.4252 | 200 кандидатов; best index 196 |
| **Random MoE + learnable gating (retrain 100ep)** 🏆 | same | **0.4596** | 37k params; best of all methods |
| SGEM `pc05_lbw1` (final-moe/learnable) | `results_cifar100_sgem_K3_semantic_pc05_lbw1.json` | 0.4155 | коллапс `[0,30,0]` с iter 4; lbw=1.0 не помог |
| SGEM `pc05_lbw1` (final-moe/cluster) | same | 0.3848 | hard_assignment коллапсный → fact-only E1 |
| DARTS search (50 ep) | `results_cifar100_darts_search_semantic.json` | — | search only, 6 мин |
| DARTS search (200 ep) | `results_cifar100_darts_search_semantic_long.json` | — | search only, 32 мин; нашёл 3×skip_connect + parallel inputs (DARTS-collapse-to-skip?) |
| **DARTS retrain (single arch)** | `results_cifar100_darts_finetune_semantic.json` | **0.4288** | 100 эпох; взят config с 50-ep search; n_params=13.3k |
| SGEM `lb5000_long_semantic` (pc-mix=0) | `results_cifar100_sgem_K3_lb5000_long_semantic.json` | TBD | сошёлся к [9,11,10] |
| SGEM `lb10000_long_semantic` (pc-mix=0) | `results_cifar100_sgem_K3_lb10000_long_semantic.json` | TBD | сошёлся к [10,10,10] |
| SGEM `pc05` (pc-mix=0.5, lbw=0.5) | `cifar100_sgem_K3_lb5000_long_semantic_pc05` | **прерван** | коллапс [0,30,0] с iter 10 до iter 66+ |
| SGEM `pc05_lbw1` (pc-mix=0.5, lbw=1.0) | `runs/sgem_K3_semantic_pc05_lbw1.log` | **запущен 2026-05-05** | старт [7,17,6], 50 EM iters; reuses 1391 obs из pc05 |
| SGEM `pc05_lbw25` (final-moe/learnable) | `results_cifar100_sgem_K3_semantic_pc05_lbw25.json` (rsm6) | 0.4292 | split [3,23,4]; obj=-32.89; lbw=25 на грани, E1 ещё доминирует |
| SGEM `pc05_lbw25` (final-moe/cluster) | same | 0.4121 | — |
| **SGEM `pc05_lbw50` (final-moe/learnable)** 🏆 | `results_cifar100_sgem_K3_semantic_pc05_lbw50.json` (rsm6) | **0.4758** | split **[8,13,9]** сбалансирован; obj=-45.70; верхний конец seed-разброса |
| SGEM `pc05_lbw50` (final-moe/cluster) | same | 0.4365 | — |
| SGEM `pc05_lbw50_seed1` (learnable) | `results_cifar100_sgem_K3_semantic_pc05_lbw50_seed1.json` (rsm6, 2026-05-06) | 0.4542 | split [11,9,10]; obj=-37.10 |
| SGEM `pc05_lbw50_seed1` (cluster) | same | 0.4101 | — |
| SGEM `pc05_lbw50_seed2` (learnable) | `results_cifar100_sgem_K3_semantic_pc05_lbw50_seed2.json` (rsm6, 2026-05-06) | 0.4513 | split [11,7,12]; obj=-42.40 |
| SGEM `pc05_lbw50_seed2` (cluster) | same | 0.4130 | — |
| SGEM `pc05_lbw50_seed3` (learnable) | `results_cifar100_sgem_K3_semantic_pc05_lbw50_seed3.json` (rsm6, 2026-05-07) | 0.4380 | split [9,10,11]; obj=-40.36 |
| SGEM `pc05_lbw50_seed3` (cluster) | same | 0.4251 | — |
| SGEM `pc05_lbw50_seed4` (learnable) | `results_cifar100_sgem_K3_semantic_pc05_lbw50_seed4.json` (rsm6, 2026-05-07) | 0.4490 | split [9,10,11]; obj=-40.91 |
| SGEM `pc05_lbw50_seed4` (cluster) | same | 0.4221 | — |
| **DARTS × K (same arch, learnable gate), seed=322** 🏆 | `results_cifar100_darts_moe_K3_semantic.json` (rsm6, 2026-05-08) | **0.4721** | 42.8k params; cluster=0.4089 |
| DARTS × K, seed=1 | `results_cifar100_darts_moe_K3_semantic_seed1.json` (rsm6, 2026-05-08) | 0.4780 | cluster=0.4169 |
| DARTS × K, seed=2 | `results_cifar100_darts_moe_K3_semantic_seed2.json` (rsm6, 2026-05-08) | 0.4839 | cluster=0.4093 |
| DARTS × K, seed=3 | `results_cifar100_darts_moe_K3_semantic_seed3.json` (rsm6, 2026-05-09) | 0.4725 | cluster=0.4169 |
| DARTS × K, seed=4 | `results_cifar100_darts_moe_K3_semantic_seed4.json` (rsm6, 2026-05-09) | 0.4755 | cluster=0.4225 |

> Чтобы заполнить пропуски — извлечь `objective_value`, `final_moe.val_acc` или
> `per_expert_eval.weighted_val_acc` из соответствующих JSON.

## Ёмкость сети (общая для всех методов)

- `CIFAR100Net` = stem(3→16) → cell1 → cell2 → fixed reduction(stride 2)
  → GAP → FC(100). **Всегда 2 normal cells.**
- `ClusterGatedMoE` (random/SGEM final) = `K` экспертов, каждый = `CIFAR100Net`
  → итого `K·2` ячеек на MoE-сеть.
- DARTS supernet тоже строит 2 cell с общими label'ами и независимыми весами;
  retrain собирает ту же `CIFAR100Net` с найденным `config`.

## Ключевые выводы

### 1. Random baseline даёт ~0.37 val_acc
- 200 кандидатов с random cluster gating: mean=0.367, best=0.383.
- Разброс top-worst всего ~3.4 п.п. — пространство пологое, любые методы
  должны заметно бить эту планку, чтобы оправдать сложность.
- Лучший hard_assignment в random — почти равномерный {10,8,12}; экстремальные
  расклады не выигрывают.

### 1b. DARTS бьёт random-MoE: 0.429 vs 0.383 (+4.6 п.п.)
- Один эксперт (CIFAR100Net, 2 cell, 13.3k params) после retrain на 100 эпох
  даёт **0.4288**, при этом параметров вдвое меньше, чем у MoE с K=3.
- Вывод: на этом search space качество архитектуры важнее наличия
  cluster-специализации (по крайней мере при random gating).
- 200-эпоховый DARTS-search вернул архитектуру с 3×skip_connect и всеми
  входами от stem — типичный симптом DARTS-collapse-to-skip; retrain пока
  делался с 50-эпохового search'а.

### 2. `phase-c-uniform-mix=0.5` рискованно при низком `load-balance-weight`
- Серия `*_mix050` (Apr 29-30, random clustering): при `lbw∈{0.0, 0.15, 0.5}`
  splits деградировали до `[7,22,1] / [9,19,2] / [10,20,0]`. Только `lbw=1.0`
  держал [5,16,9] — **на random clustering**.
- Серия semantic + pc05 (May 4-5):
  - `lbw=0.5` → полный коллапс `[0,30,0]` с iter 10.
  - `lbw=1.0` → коллапс **тот же** `[0,30,0]` с iter 4. **Удвоение lbw не помогло
    на semantic-кластеризации.**
- Гипотеза: pc-mix=0.5 заливает датасет суррогата большими подмножествами `b`,
  суррогат начинает уверенно оценивать "сильного" эксперта высоко на любых R,
  E-step стягивает массу к нему, штраф `lbw·load_balance` слишком слаб.
- На semantic clustering эффект сильнее (коллапс происходит на 2x раньше),
  возможно из-за того что один из экспертов реально лучше остальных
  на большинстве semantic-кластеров.

### 2b. Learnable gating > cluster gating (random MoE)
- Random MoE+learnable retrain 100ep: **0.4596** (37k params)
- Random MoE+cluster retrain 100ep: 0.4162 (30k params)
- Разница 4.3 п.п. — свободный софтмакс-гейт лучше random hard-assignment
  даже без всякой cluster-специализации.
- Random MoE+learnable также бьёт DARTS-single (0.4596 vs 0.4288, +3.1 п.п.) —
  но это не про специализацию, а про большее число параметров.

### 3. `ew=0` в логах SGEM — это **не баг**
- В логе печатается `current_entropy_weight` (отдельный регуляризатор,
  захардкожен в 0.0 для CIFAR-100), а не `load_balance_weight`.
- Логирование `load_balance_weight` отсутствует — желательно добавить
  отдельный print в `optimize_surrogate_em.py`, чтобы видеть масштаб
  штрафа vs `q_function`.

### 4. Args не сохраняются в JSON-результатах SGEM
- `cifar100_sgem` JSON содержит только `configs`/`r_matrix`/`hard_assignments`/
  `objective_value`/`history`/`phase_a_history`/`final_moe`.
- Из-за этого после факта **невозможно** восстановить, с какими
  гиперпараметрами запускался прогон. Желательно дописать `vars(args)` в
  результат.

### 5. Диагностика M-step scales (2026-05-05, lbw=1.0, pc-mix=0.5, 5 EM iters)

После добавления `[M-step scales]` логов в `optimize_surrogate_em.py:953`,
получены численные масштабы слагаемых loss = -q_function + lbw·LB:

| Iter | Split | -q_function | q_log_r | q_log_u | LB | lbw·LB |
|---|---|---|---|---|---|---|
| 1 | [5,19,6] | 65.04 | -18.4 | -46.6 | 1.23 | 1.23 |
| 2 | [5,19,6] | 75.79 | -26.7 | -49.1 | 2.32 | 2.32 |
| 3 | [0,28,2] | 26.63 | -7.4 | -19.2 | 2.97 | 2.97 |
| 4 | [0,30,0] | 27.61 | -0.5 | -27.1 | 3.00 | 3.00 |
| 5 | [0,30,0] | 30.52 | -0.1 | -30.4 | 3.00 | 3.00 |

**Главное:**
- `LB ∈ [1, K=3]`, **насыщается на K** при коллапсе → Δ_LB между balanced
  и collapsed всего ~1.77.
- `-q_function` в коллапсе **уменьшается на ~35**: коллапс это **глобальный
  оптимум** по q_function. Объяснение: при коллапсе все кластеры идут к
  одному «сильному» эксперту, и суррогат предсказывает u≈0.36 для него на
  любом R, тогда как при балансе среднее u≈0.21.
- Чтобы штраф перевешивал: `lbw · 1.77 > 35` → нужно **lbw ≥ 20**.
  При `lbw=8` штраф даёт прирост лишь 14 — недостаточно (что наблюдалось).
- `q_log_r → 0` при one-hot r — этот член НЕ мешает коллапсу.

**Корень проблемы — смещение датасета суррогата:**
- pc-mix=0.5 заливает датасет наблюдениями с большими |b| для всех K экспертов.
- Один из экспертов случайно лучше других на больших |b| → суррогат запоминает
  это паттерн → E-step стягивает массу к нему → датасет ещё больше смещается.
- Это самоподтверждающийся коллапс. Просто увеличение lbw временное решение
  (вытянет r из коллапса, но суррогат продолжит указывать туда).

**Возможные решения:**
1. lbw ≥ 25–30 (workaround: forces балансовое решение, но субоптимальное по q).
2. Симметризация обучения в Phase C (равно K архитектур × K b-векторов).
3. Augmentation суррогата (permutation invariance относительно k).
4. Отказ от pc-mix > 0.

### 6. Подтверждение прогноза lbw (2026-05-05/06, rsm6)

Запущены два прогона на rsm6 для проверки гипотезы "lbw ≥ 20 ломает коллапс":

- **`lbw=25`** → split [3, 23, 4]: workaround **недостаточен**, E1 всё ещё
  доминирует. Прогноз был на пограничной ситуации — численно подтвердилось.
  val_acc/learnable=0.4292 (≈DARTS-single).
- **`lbw=50`** → split [8, 13, 9]: **сработало**, баланс держится. obj хуже
  (-45.70 vs -32.89), но real val_acc/learnable=**0.4758** — новый абсолютный
  максимум, бьёт random-MoE-learnable (0.4596) на +1.6 п.п. и DARTS-single
  (0.4288) на +4.7 п.п. **Это первый результат, где SGEM реально оправдывает
  свою сложность относительно baselines.**

Подтверждает основной вывод раздела 5: q_function смещён в пользу коллапса,
сильный балансовый штраф восстанавливает разумное решение и улучшает реальное
качество. Ценой того, что objective перестаёт коррелировать с val_acc — именно
прогон с худшим objective дал лучший val_acc.

### 6b. Воспроизводимость lbw=50 (2026-05-06, rsm6)

Два повторных прогона `pc05_lbw50` с другими seed-ами для проверки, насколько
0.4758 был типичным результатом:

| seed | hard split | obj | val_acc/learnable | val_acc/cluster |
|---|---|---|---|---|
| исходный | [8, 13, 9] | -45.70 | **0.4758** | 0.4365 |
| seed1 | [11, 9, 10] | -37.10 | 0.4542 | 0.4101 |
| seed2 | [11, 7, 12] | -42.40 | 0.4513 | 0.4130 |
| **mean** | balanced | — | **≈0.4604** | ≈0.4199 |

**Выводы:**
- Баланс **держится во всех трёх seed-ах** → `lbw=50` надёжно ломает коллапс
  на semantic clustering, это не результат удачного seed-а.
- Но 0.4758 — **верхний конец разброса**. Среднее learnable val_acc ≈ 0.4604
  фактически совпадает с random-MoE-learnable (0.4596).
- Преимущество SGEM над baselines из раздела 6 уменьшается до ~0.001 п.п. на
  среднем seed-е; «обгон» random-MoE наблюдается только на лучшем seed-е.
- DARTS-single (0.4288) SGEM по-прежнему уверенно бьёт во всех seed-ах
  (+2.3 — +4.7 п.п.).
- objective и val_acc антикоррелируют внутри seed-разброса (best obj=-37.10
  → худший val_acc, worst obj=-45.70 → лучший val_acc). Это согласуется с
  выводом раздела 5: q_function смещена, и более «правильное» по q решение
  не даёт лучшего реального качества.

**Скорректированное заключение:** SGEM с pc-mix=0.5 + lbw=50 на semantic
clustering работает **сравнимо** с random-MoE-learnable, не лучше его в
среднем. Чтобы оправдать сложность метода, нужны либо: больше seed-ов (текущая
выборка 3 — мало), либо устранение смещения суррогата (раздел 5, варианты
2–4) ради честного objective.

### 6c. SGEM `pc05_lbw50` на 5 seed-ах (2026-05-07, rsm6)

Добавлены seed3 и seed4 — теперь полная выборка из 5 seed-ов.

| seed | hard split | obj | val_acc/learnable | val_acc/cluster |
|---|---|---|---|---|
| исходный | [8, 13, 9] | -45.70 | 0.4758 | 0.4365 |
| seed1 | [11, 9, 10] | -37.10 | 0.4542 | 0.4101 |
| seed2 | [11, 7, 12] | -42.40 | 0.4513 | 0.4130 |
| seed3 | [9, 10, 11] | -40.36 | 0.4380 | 0.4251 |
| seed4 | [9, 10, 11] | -40.91 | 0.4490 | 0.4221 |
| **mean** | balanced | — | **≈0.4537** | ≈0.4214 |

Среднее на 5 seed-ах **0.4537** (было 0.4604 на 3 seed-ах) — ниже
random-MoE-learnable (0.4596). SGEM **не обгоняет** random-MoE даже на
лучшем seed-е статистически значимо (нужна оценка CI).

### 7. DARTS × K (same arch, learnable gate) — новый сильнейший baseline (2026-05-08/09, rsm6)

Запущен `cifar100_darts_moe_baseline.py`: K=3 эксперта с **одинаковой**
DARTS-найденной архитектурой (config из `results_cifar100_darts_finetune_semantic.json`,
50-эпоховый search), learnable softmax-gate, retrain 100 ep.

| seed | val_acc/learnable | val_acc/cluster |
|---|---|---|
| 322 | 0.4721 | 0.4089 |
| 1   | 0.4780 | 0.4169 |
| 2   | 0.4839 | 0.4093 |
| 3   | 0.4725 | 0.4169 |
| 4   | 0.4755 | 0.4225 |
| **mean** | **0.4764** | 0.4149 |

**Это новый абсолютный максимум среди baseline'ов (5 seed-ов).** Существенно
больше random-MoE-learnable (0.4596, 1 seed) и SGEM-lbw50 (0.4537 mean, 5 seeds).
Cluster-gating ровно у границы DARTS-single (0.4288) — выигрыш от MoE
полностью уносит learnable gate.

**Выводы для статьи:**
- Архитектурная гетерогенность экспертов **не нужна** — `K` копий одной
  DARTS-архитектуры с learnable gate работают **лучше**, чем SGEM с
  гетерогенными экспертами.
- SGEM на текущей формулировке **не оправдывает** свою сложность: уступает
  и random-MoE-learnable, и DARTS×K.
- Возможно, прирост MoE вообще не от cluster-специализации, а просто от
  ансамбля + learnable gate. Чтобы это проверить, нужен ablation
  «DARTS×K с одинаковыми весами» (но это сводится к single-arch с скейлом).
- Гипотеза: текущая постановка SGEM (pc-mix=0.5 + lbw=50) даёт балансовое,
  но **архитектурно бедное** решение, потому что суррогат смещён, и архитектуры
  α_k подбираются под предсказания смещённой суррогатной функции, а не под
  реальный val_acc.

## TODO для будущих прогонов

### Инфраструктура / диагностика

- [ ] Сохранять `vars(args)` в JSON (одна правка в `cifar100_sgem.py`).
- [x] Логировать `load_balance` и `q_function` отдельно на M-step
      (сделано 2026-05-05 в `optimize_surrogate_em.py:961`).
- [ ] Заполнить пропуски в таблице val_acc (SAGEM прогоны без pc-mix
      с retrain 100ep).
- [ ] Запустить SAGEM с `lbw=25` или `lbw=50`, pc-mix=0.5 — проверить,
      удерживает ли такой штраф баланс.

### Эксперименты, нужные для законченности статьи

#### (1) Воспроизводимость SAGEM (must-have)

- [ ] Прогнать `pc05_lbw50` на **≥ 5–10 seed-ах** (сейчас всего 3,
      mean ≈ 0.4604 ≈ random-MoE-learnable=0.4596 — статзначимости нет).
- [ ] Посчитать **mean ± std** и **доверительный интервал**
      (95% CI, paired t-test или Wilcoxon) для val_acc/learnable
      vs random-MoE-learnable.
- [ ] Если SAGEM ≠ random-MoE статистически — обновить таблицу в
      `experiments.tex` и заключение.

#### (2) Ablation по компонентам метода

- [ ] **(a) Фиксированное случайное разбиение кластеров по экспертам.**
      Тот же SAGEM-pipeline, но r-матрица один раз случайно
      инициализируется как hard-assignment и **не обновляется**
      (M-step по r отключён). Архитектуры всё равно ищутся через
      суррогат + active learning. Цель: показать, что обновление
      routing-матрицы реально вносит вклад в val_acc, а не только
      сам факт независимого поиска архитектур на (случайно
      порезанных) подкластерах.
- [ ] **(b) Скорость сходимости / график «val_acc vs EM iter».**
      Логировать на каждой EM-итерации $t$:
      - текущий objective $L^{(t)}$,
      - текущий surrogate-error прокси (R²/Spearman на held-out),
      - val_acc «proxy» (короткое retrain выбранных архитектур
        на текущем split, например 20 эпох).
      Построить кривую val_acc(t) для SAGEM, для ablation (a),
      и для random-MoE (горизонтальная линия). Это эмпирически
      иллюстрирует Theorem 1 (сходимость $L_{\mathrm{true}}$).

#### (3) Полный набор baseline'ов для CIFAR-100

Сейчас в таблице есть Random-MoE, DARTS-single, SAGEM. **Добавить**:

- [ ] **Random architecture** (single net, не MoE) — sanity check,
      чтобы понять, что DARTS вообще что-то ищет. Один случайный
      `CIFAR100Net`, retrain 100 ep, ≥ 5 seeds.
- [x] **DARTS × K** — `K = 3` экспертов с **одинаковой**
      DARTS-найденной архитектурой, learnable softmax-gate, retrain
      100 ep, 5 seeds (2026-05-08/09, rsm6). **mean=0.4764** —
      бьёт все остальные методы; см. раздел 7.
- [ ] Финальная таблица должна выглядеть примерно так:
      | Method | val_acc (mean ± std, n seeds) |
      |---|---|
      | Random arch (single) | ... |
      | Random-MoE (cluster gate) | ... |
      | Random-MoE (learnable gate) | ... |
      | DARTS (single) | ... |
      | DARTS × K (same arch, learnable gate) | ... |
      | SAGEM (ablation: fixed random r) | ... |
      | **SAGEM (full)** | ... |

#### Приоритет

1. (3) Random + DARTS×K baselines — без них непонятно, что вообще
   меряем.
2. (1) Воспроизводимость SAGEM на ≥ 5 seeds.
3. (2a) Ablation с фиксированным random r.
4. (2b) Кривая val_acc(t).
