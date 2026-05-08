# EM v4 — успешные параметры (4-class dataset)

Запуск `optimize_surrogate_em.py`, нашедший **ideal split 20/20** (2026-04-02).

## Команда запуска

```bash
docker exec nas-for-moe python /pbabkin/main/mipt/nas-for-moe/code/toy_experiment/optimize_surrogate_em.py \
    --data-dir /pbabkin/main/mipt/nas-for-moe/code/toy_experiment/data \
    --M 20 --K 2 --n-nodes 3 \
    --n-em-iterations 40 \
    --n-arch-candidates 200 \
    --n-r-gradient-steps 20 \
    --r-lr 0.03 --tau 0.5 \
    --entropy-weight 0.02 --entropy-weight-end 0.02 \
    --max-logit-spread 1.5 \
    --surrogate-retrain-every 3 \
    --n-new-observations 15 \
    --initial-obs-dir /pbabkin/main/mipt/nas-for-moe/code/toy_experiment/surrogate_em_4class_v2 \
    --per-cluster-eval \
    --obs-save-dir /pbabkin/main/mipt/nas-for-moe/code/toy_experiment/surrogate_em_4class_v4 \
    --save-results /pbabkin/main/mipt/nas-for-moe/code/toy_experiment/results_surrogate_em_4class_v4.json \
    --seed 322
```

## Ключевые решения

| Параметр | Значение | Почему |
|----------|----------|--------|
| initial-obs-dir | surrogate_em_4class_v2 (350 obs) | Warm start — суррогату нужно много данных для точности |
| r-lr | 0.03 | Консервативные обновления r предотвращают осцилляции |
| n-r-gradient-steps | 20 | Меньше шагов — меньше риск уйти далеко за одну итерацию |
| entropy-weight | 0.02 (const) | Постоянная — annealing вызывал нестабильность |
| max-logit-spread | 1.5 | Предотвращает collapse, но позволяет разделение |
| surrogate-retrain-every | 3 | Частое обновление суррогата |
| n-new-observations | 15 | Больше новых данных за S-step |
| per-cluster-eval | true | **Критично** — без этого EM всегда коллапсирует |
| init-assignment | не задан (random) | EM сам нашёл ideal split |

## Результаты

- **Split [7, 13]** — 20/20 совпадение с ideal (ring vs linear)
- EM сошёлся к ideal split к итерации 6, удерживал 30 итераций
- Найденные архитектуры слабые (73.9% MoE acc)
- С reference архитектурами (linear + rbf) на этом split: **95.6% MoE acc**

## Что не работало (v1-v3)

| Запуск | Проблема |
|--------|----------|
| v1 (100 seed, ew 0.1→0) | Нестабилен, drift от [13,7] к [17,3] |
| v2 (300 seed, clip=1.0) | Слишком жёсткий clipping, застрял на [10,10] |
| v3 (ideal init, 160 obs) | Даже с ideal init ушёл к [15,5] |
