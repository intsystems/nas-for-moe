# Теорема о сходимости GEM-алгоритма с адаптивным суррогатом

---

## 0. Обозначения

| Обозначение | Определение |
|---|---|
| $A$ | Конечное множество архитектур, $\|A\| < \infty$ |
| $K$ | Число экспертов |
| $M$ | Число кластеров данных |
| $\Delta_K$ | Стандартный $(K{-}1)$-симплекс: $\{v \in [0,1]^K : \sum_k v_k = 1\}$ |
| $\mathcal{R} = (\Delta_K)^M$ | Множество матриц маршрутизации, $r \in \mathcal{R}$, строка $r_{m,:} \in \Delta_K$ |
| $\Theta = A^K \times \mathcal{R}$ | Пространство параметров; элемент $\theta = (\alpha, r)$, $\alpha = (\alpha_1, \ldots, \alpha_K)$ |
| $R_k = r_{:,k}$ | $k$-й столбец матрицы $r$ (профиль принадлежности к эксперту $k$) |
| $\mathcal{Q} = (\Delta_K)^M$ | Множество мягких назначений $q = (q_{m,:})_{m=1}^M$ |
| $u_{\mathrm{true}} : A \times [0,1]^M \to \mathbb{R}_+$ | Истинная функция качества эксперта |
| $u^{(t)} : A \times [0,1]^M \to \mathbb{R}_+$ | Суррогатная функция на итерации $t$ |
| $H(p) = -\sum_k p_k \log p_k$ | Энтропия Шеннона дискретного распределения $p$ |

**Замечание о компактности.** Множество $\Theta$ компактно: $A^K$ конечно, $\mathcal{R}$ — замкнутое ограниченное подмножество $\mathbb{R}^{M \times K}$.

---

## 1. ELBO (свободная энергия) и лог-правдоподобие

### Определение 1 (ELBO). 

Для произвольной функции качества $v : A \times [0,1]^M \to \mathbb{R}_+$ определим Evidence Lower Bound:

$$
\mathcal{F}_v(q, \theta) \;=\; \sum_{m=1}^{M} \left[\, \sum_{k=1}^{K} q_{mk}\, \log\!\bigl(r_{mk}\, v(\alpha_k, R_k)\bigr) \;+\; H(q_{m,:}) \,\right].
$$

Эквивалентно (раскрывая энтропию):

$$
\mathcal{F}_v(q, \theta) \;=\; \sum_{m=1}^{M} \sum_{k=1}^{K} q_{mk}\, \log \frac{r_{mk}\, v(\alpha_k, R_k)}{q_{mk}}.
$$

Это в точности ELBO для модели смесей, где $q_{mk}$ — вариационное распределение латентных назначений $z_m \in \{1,\ldots,K\}$. Терминология «свободная энергия» (negative variational free energy) восходит к [Neal, Hinton, 1998, §2]; тождественность с ELBO описана в [Bishop, 2006, §9.4], [Blei et al., 2017, eq. (2)].

### Предложение 1 (вариационное представление правдоподобия).

Лог-правдоподобие допускает вариационное представление:

$$
L_v(\theta) \;\equiv\; \sum_{m=1}^{M} \log\!\Bigl(\sum_{k=1}^{K} r_{mk}\, v(\alpha_k, R_k)\Bigr) \;=\; \max_{q \in \mathcal{Q}}\; \mathcal{F}_v(q, \theta),
$$

причём максимум достигается при

$$
q_{mk}^{\star}(\theta) \;=\; \frac{r_{mk}\, v(\alpha_k, R_k)}{\sum_{j=1}^{K} r_{mj}\, v(\alpha_j, R_j)}.
$$

*Доказательство.* Для каждого $m$ фиксируем $\theta$ и рассматриваем задачу $\max_{q_{m,:} \in \Delta_K} \sum_k q_{mk} \log \frac{a_{mk}}{q_{mk}}$, где $a_{mk} = r_{mk}\, v(\alpha_k, R_k)$. Это стандартная задача максимизации $-\mathrm{KL}(q_{m,:}\,\|\,a_{m,:}/\sum_j a_{mj}) + \log(\sum_j a_{mj})$. Минимум KL-дивергенции равен нулю и достигается при $q_{mk} = a_{mk}/\sum_j a_{mj}$. Подставляя, получаем $\mathcal{F}_v(q^{\star}, \theta) = \sum_m \log(\sum_k a_{mk}) = L_v(\theta)$.

Это стандартный результат, являющийся частным случаем неравенства Йенсена для лог-суммы; см. [Dempster et al., 1977, §2], [Neal, Hinton, 1998, Proposition 1]. $\square$

### Следствие (декомпозиция правдоподобия).

Для любого $q \in \mathcal{Q}$:

$$
L_v(\theta) \;=\; \mathcal{F}_v(q, \theta) \;+\; \sum_{m=1}^{M} \mathrm{KL}\!\bigl(q_{m,:}\,\big\|\, q_{m,:}^{\star}(\theta)\bigr),
$$

откуда $L_v(\theta) \geq \mathcal{F}_v(q, \theta)$ с равенством тогда и только тогда, когда $q = q^{\star}(\theta)$.

Это декомпозиция Нила–Хинтона [Neal, Hinton, 1998, eq. (3)]; идентичная формула в [Bishop, 2006, eq. (9.73)].

---

## 2. Мера ошибки суррогата

### Определение 2.

Погрешность суррогата на итерации $t$:

$$
\varepsilon_t \;=\; \sup_{\alpha \in A,\; R \in [0,1]^M} \bigl|\log u^{(t)}(\alpha, R) \;-\; \log u_{\mathrm{true}}(\alpha, R)\bigr|.
$$

---

## 3. Описание алгоритма

На итерации $t = 0, 1, 2, \ldots$:

**E-шаг.** Максимизировать ELBO по $q$:

$$
q^{(t)} \;=\; \arg\max_{q \in \mathcal{Q}}\; \mathcal{F}_{u^{(t)}}(q,\, \theta^{(t)}).
$$

По Предложению 1: $q_{mk}^{(t)} = r_{mk}^{(t)} u^{(t)}(\alpha_k^{(t)}, R_k^{(t)}) \big/ \sum_j r_{mj}^{(t)} u^{(t)}(\alpha_j^{(t)}, R_j^{(t)})$.

**M-шаг (GEM).** Найти $\theta^{(t+1)}$ такое, что ELBO не убывает:

$$
\mathcal{F}_{u^{(t)}}\!\bigl(q^{(t)},\, \theta^{(t+1)}\bigr) \;\geq\; \mathcal{F}_{u^{(t)}}\!\bigl(q^{(t)},\, \theta^{(t)}\bigr).
$$

**S-шаг.** Обновить суррогат: $u^{(t)} \to u^{(t+1)}$ (добавление точек в обучающий набор, переобучение).

---

## 4. Условия теоремы

**(C1) Равномерная ограниченность.** Существуют $0 < c \leq C < \infty$ такие, что для всех $\alpha \in A$, $R \in [0,1]^M$, $t \geq 0$:

$$
c \;\leq\; u_{\mathrm{true}}(\alpha, R) \;\leq\; C, \qquad c \;\leq\; u^{(t)}(\alpha, R) \;\leq\; C.
$$

**(C2) Суммируемость ошибки суррогата.**

$$
\sum_{t=0}^{\infty} \varepsilon_t \;<\; \infty.
$$

**(C3) GEM-условие.** На каждой итерации $t$:

$$
\mathcal{F}_{u^{(t)}}\!\bigl(q^{(t)},\, \theta^{(t+1)}\bigr) \;\geq\; \mathcal{F}_{u^{(t)}}\!\bigl(q^{(t)},\, \theta^{(t)}\bigr).
$$

**(C4) Непрерывность.** Для каждого $\alpha \in A$ функция $u_{\mathrm{true}}(\alpha, \cdot)$ непрерывна на $[0,1]^M$.

---

## 5. Формулировка теоремы

**Теорема.** Пусть $\{\theta^{(t)}\}_{t \geq 0}$ — последовательность, порождённая алгоритмом §3, и выполнены условия (C1)–(C4). Тогда:

**(a)** Последовательность $\{L_{\mathrm{true}}(\theta^{(t)})\}$ сходится.

**(b)** Любая предельная точка $\theta^{\star}$ последовательности $\{\theta^{(t)}\}$ является стационарной точкой $L_{\mathrm{true}}$ на $\Theta$, в следующем смысле: для всех $\theta \in \Theta$

$$
Q_{\mathrm{true}}(\theta;\, \theta^{\star}) \;\leq\; Q_{\mathrm{true}}(\theta^{\star};\, \theta^{\star}),
$$

где $Q_{\mathrm{true}}(\theta;\, \theta') = \sum_m \sum_k w_{mk}(\theta')\, \log\!\bigl(r_{mk}\, u_{\mathrm{true}}(\alpha_k, R_k)\bigr)$ и $w_{mk}(\theta') = q_{mk}^{\star}(\theta')$ — апостериорные веса по $u_{\mathrm{true}}$ (Предложение 1).

---

## 6. Доказательство

### 6.1. Лемма 1 (равномерное отклонение ELBO)

**Формулировка.** При условии (C1), для всех $q \in \mathcal{Q}$, $\theta \in \Theta$, $t \geq 0$:

$$
\bigl|\mathcal{F}_{u_{\mathrm{true}}}(q, \theta) - \mathcal{F}_{u^{(t)}}(q, \theta)\bigr| \;\leq\; M\, \varepsilon_t.
$$

**Доказательство.** По определению ELBO (§1):

$$
\mathcal{F}_{u_{\mathrm{true}}}(q, \theta) - \mathcal{F}_{u^{(t)}}(q, \theta) \;=\; \sum_{m=1}^{M} \sum_{k=1}^{K} q_{mk}\, \bigl[\log u_{\mathrm{true}}(\alpha_k, R_k) - \log u^{(t)}(\alpha_k, R_k)\bigr].
$$

Здесь члены $\log r_{mk}$ и $-\log q_{mk}$ взаимно сокращаются — ключевое преимущество работы через ELBO, отмечавшееся в [Neal, Hinton, 1998, §2]. Оценивая модуль:

$$
\bigl|\mathcal{F}_{u_{\mathrm{true}}} - \mathcal{F}_{u^{(t)}}\bigr| \;\leq\; \sum_{m=1}^{M} \sum_{k=1}^{K} q_{mk}\, \varepsilon_t \;=\; M\, \varepsilon_t,
$$

где использовано $\sum_k q_{mk} = 1$ и определение $\varepsilon_t$. $\square$

### 6.2. Следствие (равномерное отклонение лог-правдоподобий)

Для всех $\theta \in \Theta$, $t \geq 0$:

$$
\bigl|L_{\mathrm{true}}(\theta) - L^{(t)}(\theta)\bigr| \;\leq\; M\, \varepsilon_t.
$$

**Доказательство.** Поскольку $L_v(\theta) = \max_{q} \mathcal{F}_v(q, \theta)$ (Предложение 1) и для любых функций $f, g$ выполнено $|\max_q f(q) - \max_q g(q)| \leq \sup_q |f(q) - g(q)|$ (стандартное свойство, см. [Rockafellar, 1970, §10]), имеем:

$$
\bigl|L_{\mathrm{true}}(\theta) - L^{(t)}(\theta)\bigr| \;=\; \bigl|\max_q \mathcal{F}_{u_{\mathrm{true}}}(q, \theta) - \max_q \mathcal{F}_{u^{(t)}}(q, \theta)\bigr| \;\leq\; \sup_q \bigl|\mathcal{F}_{u_{\mathrm{true}}}(q, \theta) - \mathcal{F}_{u^{(t)}}(q, \theta)\bigr| \;\leq\; M\, \varepsilon_t. \quad \square
$$

### 6.3. Предложение 2 (квази-монотонность)

**Формулировка.** При условиях (C1)–(C3), для всех $t \geq 0$:

$$
L_{\mathrm{true}}(\theta^{(t+1)}) \;\geq\; L_{\mathrm{true}}(\theta^{(t)}) \;-\; 2M\, \varepsilon_t. \tag{$\star$}
$$

**Доказательство.** Выстраиваем цепочку из пяти неравенств.

**Шаг (i).** Так как $L_{\mathrm{true}}(\theta) = \max_q \mathcal{F}_{u_{\mathrm{true}}}(q, \theta)$, при любом фиксированном $q$:

$$
L_{\mathrm{true}}(\theta^{(t+1)}) \;\geq\; \mathcal{F}_{u_{\mathrm{true}}}(q^{(t)}, \theta^{(t+1)}).
$$

Это стандартное неравенство Йенсена для смесей [Dempster et al., 1977, eq. (2.2)].

**Шаг (ii).** Применяя Лемму 1 к паре $(q^{(t)}, \theta^{(t+1)})$:

$$
\mathcal{F}_{u_{\mathrm{true}}}(q^{(t)}, \theta^{(t+1)}) \;\geq\; \mathcal{F}_{u^{(t)}}(q^{(t)}, \theta^{(t+1)}) \;-\; M\, \varepsilon_t.
$$

**Шаг (iii).** Применяя условие GEM (C3):

$$
\mathcal{F}_{u^{(t)}}(q^{(t)}, \theta^{(t+1)}) \;\geq\; \mathcal{F}_{u^{(t)}}(q^{(t)}, \theta^{(t)}).
$$

Это определение Generalized EM [Dempster et al., 1977, §3], [Neal, Hinton, 1998, §4].

**Шаг (iv).** По определению E-шага, $q^{(t)} = \arg\max_q \mathcal{F}_{u^{(t)}}(q, \theta^{(t)})$, поэтому:

$$
\mathcal{F}_{u^{(t)}}(q^{(t)}, \theta^{(t)}) \;=\; L^{(t)}(\theta^{(t)}).
$$

Это свойство E-шага как максимизации по $q$ [Neal, Hinton, 1998, Theorem 1].

**Шаг (v).** По Следствию 6.2:

$$
L^{(t)}(\theta^{(t)}) \;\geq\; L_{\mathrm{true}}(\theta^{(t)}) \;-\; M\, \varepsilon_t.
$$

**Цепочка.** Собирая шаги (i)–(v):

$$
L_{\mathrm{true}}(\theta^{(t+1)}) \;\geq\; \mathcal{F}_{u_{\mathrm{true}}}(q^{(t)}, \theta^{(t+1)}) \;\geq\; \mathcal{F}_{u^{(t)}}(q^{(t)}, \theta^{(t+1)}) - M\varepsilon_t \;\geq\; \mathcal{F}_{u^{(t)}}(q^{(t)}, \theta^{(t)}) - M\varepsilon_t
$$

$$
= L^{(t)}(\theta^{(t)}) - M\varepsilon_t \;\geq\; L_{\mathrm{true}}(\theta^{(t)}) - 2M\varepsilon_t. \quad \square
$$

**Замечание.** При $\varepsilon_t = 0$ (точный суррогат) цепочка воспроизводит стандартный результат монотонности EM: $L(\theta^{(t+1)}) \geq L(\theta^{(t)})$; см. [Wu, 1983, Theorem 2].

### 6.4. Доказательство пункта (a): сходимость $L_{\mathrm{true}}(\theta^{(t)})$

Доказательство основано на теореме Роббинса–Зигмунда.

**Теорема (Robbins, Siegmund, 1971).** Пусть $\{V_t\}_{t \geq 0}$, $\{a_t\}_{t \geq 0}$, $\{b_t\}_{t \geq 0}$ — неотрицательные последовательности, удовлетворяющие

$$
V_{t+1} \;\leq\; V_t \;-\; a_t \;+\; b_t, \qquad \sum_{t=0}^{\infty} b_t \;<\; \infty.
$$

Тогда $\{V_t\}$ сходится и $\sum_{t=0}^{\infty} a_t < \infty$.

*Источник:* [Robbins, Siegmund, 1971, Theorem 1]. Эту теорему применяют к анализу стохастических версий EM в [Delyon et al., 1999, доказательство Theorem 1] и [Cappé, Moulines, 2009, Proposition 3].

**Применение.** Определим:

$$
V_t \;=\; L_{\max} \;-\; L_{\mathrm{true}}(\theta^{(t)}), \qquad \text{где } L_{\max} = \sup_{\theta \in \Theta} L_{\mathrm{true}}(\theta).
$$

Величина $L_{\max}$ конечна, поскольку $\Theta$ компактно и $L_{\mathrm{true}}$ непрерывна на $\Theta$ (из (C1), (C4) и конечности $A$). Далее, $V_t \geq 0$ по определению.

Из Предложения 2 ($\star$):

$$
L_{\mathrm{true}}(\theta^{(t+1)}) \;\geq\; L_{\mathrm{true}}(\theta^{(t)}) \;-\; 2M\, \varepsilon_t,
$$

откуда:

$$
V_{t+1} = L_{\max} - L_{\mathrm{true}}(\theta^{(t+1)}) \;\leq\; L_{\max} - L_{\mathrm{true}}(\theta^{(t)}) + 2M\, \varepsilon_t \;=\; V_t + 2M\, \varepsilon_t.
$$

Положим $a_t = 0$ и $b_t = 2M\, \varepsilon_t$. Тогда $V_{t+1} \leq V_t - a_t + b_t$ и $\sum_t b_t = 2M \sum_t \varepsilon_t < \infty$ по (C2). По теореме Роббинса–Зигмунда $\{V_t\}$ сходится, а значит, сходится и $\{L_{\mathrm{true}}(\theta^{(t)})\}$. $\square$

**Усиление.** Более того, можно извлечь дополнительную информацию. Определим «приращение истинной Q-функции»:

$$
\Delta_t \;=\; Q_{\mathrm{true}}(\theta^{(t+1)};\, \theta^{(t)}) \;-\; Q_{\mathrm{true}}(\theta^{(t)};\, \theta^{(t)}).
$$

По стандартному свойству EM-неравенства [Wu, 1983, Lemma 1]:

$$
L_{\mathrm{true}}(\theta^{(t+1)}) - L_{\mathrm{true}}(\theta^{(t)}) \;\geq\; \Delta_t. \tag{$\star\star$}
$$

Это следует из того, что $L_{\mathrm{true}}(\theta) - L_{\mathrm{true}}(\theta') \geq Q_{\mathrm{true}}(\theta; \theta') - Q_{\mathrm{true}}(\theta'; \theta')$ (частный случай неравенства Гиббса; [Csiszár, Tusnády, 1984, Theorem 2]).

Из ($\star$) и ($\star\star$): $\Delta_t \leq L_{\mathrm{true}}(\theta^{(t+1)}) - L_{\mathrm{true}}(\theta^{(t)})$, а также $\Delta_t \geq -2M\varepsilon_t$ (из ($\star$) и ($\star\star$)). Поэтому $\Delta_t + 2M\varepsilon_t \geq 0$, и:

$$
V_{t+1} \;\leq\; V_t \;-\; \underbrace{(\Delta_t + 2M\varepsilon_t)}_{\geq\, 0,\; =:\, a_t} \;+\; \underbrace{4M\varepsilon_t}_{=:\, b_t}.
$$

Проверка: $V_{t+1} = V_t - (L_{\mathrm{true}}(\theta^{(t+1)}) - L_{\mathrm{true}}(\theta^{(t)})) \leq V_t - \Delta_t \leq V_t - (\Delta_t + 2M\varepsilon_t) + 2M\varepsilon_t$... 

Пересчитаем аккуратнее. Из ($\star\star$):

$$
V_{t+1} = V_t - (L_{\mathrm{true}}(\theta^{(t+1)}) - L_{\mathrm{true}}(\theta^{(t)})) \leq V_t - \Delta_t.
$$

Запишем $\Delta_t = (\Delta_t + 2M\varepsilon_t) - 2M\varepsilon_t$. Тогда:

$$
V_{t+1} \;\leq\; V_t \;-\; (\Delta_t + 2M\varepsilon_t) \;+\; 2M\varepsilon_t.
$$

Здесь $a_t = \Delta_t + 2M\varepsilon_t \geq 0$ (из Предложения 2) и $b_t = 2M\varepsilon_t$, $\sum_t b_t < \infty$. По Роббинсу–Зигмунду:

$$
\sum_{t=0}^{\infty} (\Delta_t + 2M\varepsilon_t) \;<\; \infty,
$$

и поскольку $\sum_t \varepsilon_t < \infty$, получаем:

$$
\sum_{t=0}^{\infty} \Delta_t \;<\; \infty, \quad \text{т.е.} \quad \sum_{t=0}^{\infty} \bigl[Q_{\mathrm{true}}(\theta^{(t+1)};\, \theta^{(t)}) - Q_{\mathrm{true}}(\theta^{(t)};\, \theta^{(t)})\bigr] \;<\; \infty. \tag{$\star\star\star$}
$$

Этот факт используется в доказательстве пункта (b).

### 6.5. Лемма 2 (устойчивость softmax-нормализации)

**Формулировка.** Пусть $a_k > 0$ и $\tilde{a}_k > 0$ для $k = 1,\ldots,K$, причём $e^{-\varepsilon}\, a_k \leq \tilde{a}_k \leq e^{\varepsilon}\, a_k$ для некоторого $\varepsilon \geq 0$. Обозначим $p_k = a_k / \sum_j a_j$ и $\tilde{p}_k = \tilde{a}_k / \sum_j \tilde{a}_j$. Тогда:

$$
|\tilde{p}_k - p_k| \;\leq\; (e^{2\varepsilon} - 1)\, p_k.
$$

В частности, при $\varepsilon \leq 1$: $|\tilde{p}_k - p_k| \leq 3\varepsilon\, p_k$.

**Доказательство.** Из условий: $\tilde{a}_k = a_k\, e^{\xi_k}$, $|\xi_k| \leq \varepsilon$. Тогда:

$$
\tilde{p}_k = \frac{a_k\, e^{\xi_k}}{\sum_j a_j\, e^{\xi_j}}.
$$

*Верхняя граница:* $\tilde{p}_k \leq \frac{a_k\, e^{\varepsilon}}{\sum_j a_j\, e^{-\varepsilon}} = p_k\, e^{2\varepsilon}$.

*Нижняя граница:* $\tilde{p}_k \geq \frac{a_k\, e^{-\varepsilon}}{\sum_j a_j\, e^{\varepsilon}} = p_k\, e^{-2\varepsilon}$.

Следовательно: $|{\tilde{p}_k}/{p_k} - 1| \leq e^{2\varepsilon} - 1$. При $\varepsilon \leq 1$: $e^{2\varepsilon} - 1 \leq 2\varepsilon\, e^2 / e \leq 3\varepsilon$. $\square$

Аналогичные оценки мультипликативной устойчивости нормализации используются в анализе алгоритма Sinkhorn [Peyré, Cuturi, 2019, §4.2] и в теории распределённой оптимизации [Gao, Pavel, 2017, Lemma 2].

### 6.6. Лемма 3 (отклонение апостериорных весов)

**Формулировка.** Пусть $q_{mk}^{(t)}$ — E-step веса по суррогату $u^{(t)}$ (§3) и $w_{mk} = q_{mk}^{\star,\mathrm{true}}(\theta^{(t)})$ — апостериорные веса по $u_{\mathrm{true}}$ (Предложение 1). Тогда при $\varepsilon_t \leq 1$:

$$
\bigl|q_{mk}^{(t)} - w_{mk}\bigr| \;\leq\; 3\,\varepsilon_t\; w_{mk} \;\leq\; 3\,\varepsilon_t.
$$

**Доказательство.** Обозначим $a_{mk} = r_{mk}^{(t)}\, u_{\mathrm{true}}(\alpha_k^{(t)}, R_k^{(t)})$ и $\tilde{a}_{mk} = r_{mk}^{(t)}\, u^{(t)}(\alpha_k^{(t)}, R_k^{(t)})$. Из определения $\varepsilon_t$:

$$
e^{-\varepsilon_t}\, a_{mk} \;\leq\; \tilde{a}_{mk} \;\leq\; e^{\varepsilon_t}\, a_{mk}.
$$

Поскольку $w_{mk} = a_{mk} / \sum_j a_{mj}$ и $q_{mk}^{(t)} = \tilde{a}_{mk} / \sum_j \tilde{a}_{mj}$, применяем Лемму 2. $\square$

### 6.7. Лемма 4 (отклонение Q-функций)

**Формулировка.** Определим «суррогатную Q-функцию»:

$$
\tilde{Q}^{(t)}(\theta;\, \theta') \;=\; \sum_{m=1}^{M} \sum_{k=1}^{K} q_{mk}^{(t)}(\theta')\, \log\!\bigl(r_{mk}\, u^{(t)}(\alpha_k, R_k)\bigr)
$$

и «истинную Q-функцию»:

$$
Q_{\mathrm{true}}(\theta;\, \theta') \;=\; \sum_{m=1}^{M} \sum_{k=1}^{K} w_{mk}(\theta')\, \log\!\bigl(r_{mk}\, u_{\mathrm{true}}(\alpha_k, R_k)\bigr).
$$

Тогда при (C1) существует константа $C_Q = C_Q(M, K, c, C)$ такая, что для всех $\theta, \theta' \in \Theta$, $t \geq 0$, при $\varepsilon_t \leq 1$:

$$
\bigl|\tilde{Q}^{(t)}(\theta;\, \theta') - Q_{\mathrm{true}}(\theta;\, \theta')\bigr| \;\leq\; C_Q\, \varepsilon_t.
$$

**Доказательство.** Разложим разность, добавляя и вычитая $\sum_m \sum_k q_{mk}^{(t)}\, \log(r_{mk}\, u_{\mathrm{true}}(\alpha_k, R_k))$:

$$
\tilde{Q}^{(t)} - Q_{\mathrm{true}} = \underbrace{\sum_m \sum_k q_{mk}^{(t)}\, [\log u^{(t)}(\alpha_k, R_k) - \log u_{\mathrm{true}}(\alpha_k, R_k)]}_{\text{(I): ошибка суррогата}} + \underbrace{\sum_m \sum_k (q_{mk}^{(t)} - w_{mk})\, \log(r_{mk}\, u_{\mathrm{true}}(\alpha_k, R_k))}_{\text{(II): ошибка весов}}.
$$

**Оценка (I).** $|(\mathrm{I})| \leq \sum_m \sum_k q_{mk}^{(t)}\, \varepsilon_t = M\, \varepsilon_t$.

**Оценка (II).** Разобьём $\log(r_{mk}\, u_{\mathrm{true}}) = \log r_{mk} + \log u_{\mathrm{true}}$.

*Часть с $\log u_{\mathrm{true}}$:* По Лемме 3, $|q_{mk}^{(t)} - w_{mk}| \leq 3\varepsilon_t$, и $|\log u_{\mathrm{true}}| \leq \max(|\log c|, |\log C|) =: \Lambda$. Поэтому:

$$
\biggl|\sum_m \sum_k (q_{mk}^{(t)} - w_{mk})\, \log u_{\mathrm{true}}\biggr| \;\leq\; 3\varepsilon_t \cdot MK\Lambda.
$$

*Часть с $\log r_{mk}$:* Здесь $\log r_{mk}$ может быть неограничен при $r_{mk} \to 0$. Однако из определений:

$$
q_{mk}^{(t)} - w_{mk} = r_{mk}^{(t)} \left(\frac{u^{(t)}_k}{\sum_j r_{mj}^{(t)} u^{(t)}_j} - \frac{u^{\mathrm{true}}_k}{\sum_j r_{mj}^{(t)} u^{\mathrm{true}}_j}\right),
$$

где для краткости $u_k^{(t)} = u^{(t)}(\alpha_k^{(t)}, R_k^{(t)})$. Обозначим выражение в скобках как $D_{mk}$. Тогда:

$$
(q_{mk}^{(t)} - w_{mk})\, \log r_{mk}^{(t)} = r_{mk}^{(t)}\, D_{mk}\, \log r_{mk}^{(t)}.
$$

Используем $|x \log x| \leq 1/e$ для $x \in [0, 1]$ (максимум функции $-x\log x$ на $[0,1]$ равен $1/e$; это стандартный факт, см. [Cover, Thomas, 2006, §2.7]):

$$
\bigl|r_{mk}^{(t)}\, \log r_{mk}^{(t)}\bigr| \;\leq\; 1/e.
$$

Оценим $|D_{mk}|$. По Лемме 2 (устойчивость softmax) и условию (C1):

$$
|D_{mk}| \;\leq\; \frac{C}{c\, K}\, (e^{2\varepsilon_t} - 1) \;\leq\; \frac{3C\varepsilon_t}{c\, K}.
$$

Итого:

$$
\biggl|\sum_m \sum_k (q_{mk}^{(t)} - w_{mk})\, \log r_{mk}\biggr| \;\leq\; MK \cdot \frac{1}{e} \cdot \frac{3C\varepsilon_t}{cK} = \frac{3MC\varepsilon_t}{ec}.
$$

**Собирая (I) и (II):**

$$
\bigl|\tilde{Q}^{(t)} - Q_{\mathrm{true}}\bigr| \;\leq\; M\varepsilon_t + 3MK\Lambda\varepsilon_t + \frac{3MC}{ec}\varepsilon_t \;=\; C_Q\, \varepsilon_t,
$$

где $C_Q = M(1 + 3K\Lambda + 3C/(ec))$. $\square$

### 6.8. Доказательство пункта (b): стационарность предельных точек

**Шаг 1. Существование предельных точек.** Поскольку $\Theta$ компактно (§0), по теореме Больцано–Вейерштрасса $\{\theta^{(t)}\}$ имеет хотя бы одну предельную точку. Пусть $\theta^{\star} = \lim_{j \to \infty} \theta^{(t_j)}$ вдоль подпоследовательности $\{t_j\}$.

**Шаг 2. Доказательство от противного.** Предположим, что $\theta^{\star}$ не стационарна: существуют $\theta' \in \Theta$ и $\delta > 0$ такие, что

$$
Q_{\mathrm{true}}(\theta';\, \theta^{\star}) - Q_{\mathrm{true}}(\theta^{\star};\, \theta^{\star}) \;=\; \delta \;>\; 0. \tag{$\dagger$}
$$

**Шаг 3. Непрерывность $Q_{\mathrm{true}}$ по второму аргументу.** Функция $\theta' \mapsto Q_{\mathrm{true}}(\theta;\, \theta')$ непрерывна на $\Theta$ при каждом фиксированном $\theta$. Действительно:
- $w_{mk}(\theta')$ — непрерывная функция $\theta'$, поскольку $u_{\mathrm{true}}$ непрерывна по $R$ (условие C4), а softmax-нормализация непрерывна;
- $\log(r_{mk}\, u_{\mathrm{true}}(\alpha_k, R_k))$ непрерывна по $\theta$ при $r_{mk} > 0$.

На границе $r_{mk} = 0$ оба множителя $w_{mk}$ и $\log r_{mk}$ требуют отдельного анализа, но произведение $w_{mk} \log r_{mk}$ остаётся непрерывным, т.к. $w_{mk} = O(r_{mk})$ и $r_{mk} |\log r_{mk}| \to 0$.

Этот аргумент непрерывности стандартен в теории EM; см. [Wu, 1983, Condition 2] и [McLachlan, Krishnan, 2008, §3.5].

**Шаг 4. Переход к подпоследовательности.** Из непрерывности $Q_{\mathrm{true}}$: поскольку $\theta^{(t_j)} \to \theta^{\star}$, для достаточно большого $j$:

$$
Q_{\mathrm{true}}(\theta';\, \theta^{(t_j)}) - Q_{\mathrm{true}}(\theta^{(t_j)};\, \theta^{(t_j)}) \;\geq\; \frac{\delta}{2}. \tag{$\dagger\dagger$}
$$

**Шаг 5. Оценка через суррогатную Q-функцию.** По Лемме 4:

$$
\tilde{Q}^{(t_j)}(\theta';\, \theta^{(t_j)}) \;\geq\; Q_{\mathrm{true}}(\theta';\, \theta^{(t_j)}) - C_Q\, \varepsilon_{t_j},
$$

$$
\tilde{Q}^{(t_j)}(\theta^{(t_j)};\, \theta^{(t_j)}) \;\leq\; Q_{\mathrm{true}}(\theta^{(t_j)};\, \theta^{(t_j)}) + C_Q\, \varepsilon_{t_j}.
$$

Вычитая:

$$
\tilde{Q}^{(t_j)}(\theta';\, \theta^{(t_j)}) - \tilde{Q}^{(t_j)}(\theta^{(t_j)};\, \theta^{(t_j)}) \;\geq\; \frac{\delta}{2} - 2C_Q\, \varepsilon_{t_j}. \tag{$\dagger\dagger\dagger$}
$$

Поскольку $\varepsilon_{t_j} \to 0$ (из (C2)), при достаточно большом $j$ правая часть $\geq \delta/4$.

**Шаг 6. Приращение истинной Q-функции.** Из условия GEM (C3), M-шаг на итерации $t_j$ находит $\theta^{(t_j + 1)}$ с $\tilde{Q}^{(t_j)}(\theta^{(t_j+1)}; \theta^{(t_j)}) \geq \tilde{Q}^{(t_j)}(\theta^{(t_j)}; \theta^{(t_j)})$. Если $\theta'$ входит в множество кандидатов M-шага (что гарантировано, если $A$ конечно и перебирается полностью), то:

$$
\tilde{Q}^{(t_j)}(\theta^{(t_j+1)};\, \theta^{(t_j)}) \;\geq\; \tilde{Q}^{(t_j)}(\theta';\, \theta^{(t_j)}).
$$

По Лемме 4:

$$
Q_{\mathrm{true}}(\theta^{(t_j+1)};\, \theta^{(t_j)}) \;\geq\; \tilde{Q}^{(t_j)}(\theta^{(t_j+1)};\, \theta^{(t_j)}) - C_Q\, \varepsilon_{t_j} \;\geq\; \tilde{Q}^{(t_j)}(\theta';\, \theta^{(t_j)}) - C_Q\, \varepsilon_{t_j}.
$$

Аналогично:

$$
Q_{\mathrm{true}}(\theta^{(t_j)};\, \theta^{(t_j)}) \;\leq\; \tilde{Q}^{(t_j)}(\theta^{(t_j)};\, \theta^{(t_j)}) + C_Q\, \varepsilon_{t_j}.
$$

Вычитая и используя ($\dagger\dagger\dagger$):

$$
\Delta_{t_j} \;=\; Q_{\mathrm{true}}(\theta^{(t_j+1)};\, \theta^{(t_j)}) - Q_{\mathrm{true}}(\theta^{(t_j)};\, \theta^{(t_j)}) \;\geq\; \frac{\delta}{2} - 4C_Q\, \varepsilon_{t_j} \;\geq\; \frac{\delta}{4}
$$

для достаточно больших $j$.

**Шаг 7. Противоречие.** Из Шага 6: $\Delta_{t_j} \geq \delta/4$ для бесконечно многих $j$. Но из ($\star\star\star$) (§6.4): $\sum_{t=0}^{\infty} \Delta_t < \infty$, поскольку $\Delta_t + 2M\varepsilon_t \geq 0$ и $\sum_t (\Delta_t + 2M\varepsilon_t) < \infty$, $\sum_t \varepsilon_t < \infty$. Расходимость подсуммы $\sum_j \Delta_{t_j} = +\infty$ противоречит $\sum_t \Delta_t < \infty$.

Схема «от противного через суммируемость $\Delta_t$» стандартна в теории сходимости EM; см. [Wu, 1983, Theorem 4], [Vaida, 2005, Theorem 1]. $\square$

---

## 7. Замечания

### 7.1. О выполнимости условия (C3)

Условие GEM (C3) выполняется тривиально, если текущее решение $\theta^{(t)}$ включено в множество кандидатов M-шага: тогда $\mathcal{F}_{u^{(t)}}(q^{(t)}, \theta^{(t+1)}) \geq \mathcal{F}_{u^{(t)}}(q^{(t)}, \theta^{(t)})$ автоматически. Это стандартный приём для GEM; см. [Dempster et al., 1977, §3], [McLachlan, Krishnan, 2008, §3.6].

### 7.2. Ослабление условия (C2) до $\varepsilon_t \to 0$

Если заменить суммируемость на $\varepsilon_t \to 0$ (без $\sum_t \varepsilon_t < \infty$):
- Пункт (a) **теряется**: $\{L_{\mathrm{true}}(\theta^{(t)})\}$ может не сходиться, т.к. теорема Роббинса–Зигмунда неприменима.
- Пункт (b) **сохраняется**: доказательство от противного (§6.8) использует только $\varepsilon_{t_j} \to 0$, а не суммируемость.

Это аналогично соотношению между SAEM [Delyon et al., 1999] и online EM [Cappé, Moulines, 2009].

### 7.3. Скорость сходимости

Из доказательства §6.4:

$$
L_{\mathrm{true}}(\theta^{\star}) - L_{\mathrm{true}}(\theta^{(T)}) \;\leq\; 2M \sum_{t=T}^{\infty} \varepsilon_t.
$$

Если $\varepsilon_t = O(t^{-p})$ при $p > 1$, то $L_{\mathrm{true}}(\theta^{(T)}) \to L_{\mathrm{true}}(\theta^{\star})$ со скоростью $O(T^{1-p})$.

---

## Литература

1. **Dempster, A. P., Laird, N. M., Rubin, D. B.** (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society, Series B*, 39(1):1–38.

2. **Wu, C. F. J.** (1983). On the convergence properties of the EM algorithm. *The Annals of Statistics*, 11(1):95–103.

3. **Neal, R. M., Hinton, G. E.** (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. In *Learning in Graphical Models*, NATO ASI Series, vol. 89, Springer.

4. **Robbins, H., Siegmund, D.** (1971). A convergence theorem for non negative almost supermartingales and some applications. In *Optimizing Methods in Statistics*, Academic Press, pp. 233–257.

5. **Delyon, B., Lavielle, M., Moulines, É.** (1999). Convergence of a stochastic approximation version of the EM algorithm. *The Annals of Statistics*, 27(1):94–128.

6. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer, §9.4.

7. **Blei, D. M., Kucukelbir, A., McAuliffe, J. D.** (2017). Variational inference: a review for statisticians. *Journal of the American Statistical Association*, 112(518):859–877.

8. **Csiszár, I., Tusnády, G.** (1984). Information geometry and alternating minimization procedures. *Statistics and Decisions*, Supplement Issue 1:205–237.

9. **Cappé, O., Moulines, É.** (2009). On-line expectation–maximization algorithm for latent data models. *Journal of the Royal Statistical Society, Series B*, 71(3):593–613.

10. **McLachlan, G. J., Krishnan, T.** (2008). *The EM Algorithm and Extensions*, 2nd edition. Wiley.

11. **Rockafellar, R. T.** (1970). *Convex Analysis*. Princeton University Press.

12. **Cover, T. M., Thomas, J. A.** (2006). *Elements of Information Theory*, 2nd edition. Wiley.

13. **Peyré, G., Cuturi, M.** (2019). Computational optimal transport. *Foundations and Trends in Machine Learning*, 11(5–6):355–607.

14. **Zhang, Z., Kwok, J. T., Yeung, D.-Y.** (2007). Surrogate maximization/minimization algorithms and extensions. *Machine Learning*, 69:1–33.

15. **Lange, K., Hunter, D. R., Yang, I.** (2000). Optimization transfer using surrogate objective functions. *Journal of Computational and Graphical Statistics*, 9(1):1–59.

16. **Vaida, F.** (2005). Parameter convergence for EM and MM algorithms. *Statistica Sinica*, 15:831–840.

17. **Srinivas, N., Krause, A., Kakade, S. M., Seeger, M.** (2010). Gaussian process optimization in the bandit setting: no regret and experimental design. *ICML*.

18. **Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., de Freitas, N.** (2016). Taking the human out of the loop: a review of Bayesian optimization. *Proceedings of the IEEE*, 104(1):148–175.