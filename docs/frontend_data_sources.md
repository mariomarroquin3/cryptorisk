# De dónde sale cada número del frontend

Referencia técnica: para cada pieza de UI en `web/`, qué hook la alimenta, qué
endpoint de la API responde, qué función hace el cálculo, y si el dato es
**congelado** (estudio original, nunca cambia), **incremental** (walk-forward
en vivo, `backtests_live.parquet`), **en vivo** (re-fit del modelo en memoria,
en cada request) o **precio de mercado** (Binance, vía `live_tail`/`live_price`).

Generado a partir de una revisión completa de `web/app/*/page.tsx`,
`web/components/*.tsx`, `src/cryptorisk/api/app.py` y
`src/cryptorisk/api/data.py`. Los números de línea citados pueden desviarse
con el tiempo; si algo no coincide, el código manda.

Convención de signo (vale para toda la API y todo el frontend): VaR/ES son
**log-returns negativos**. El precio correspondiente se obtiene con
`last_close * exp(x)` — en el servidor vía `app.py::_price()`, y replicado en
el cliente en `page.tsx::buildBand` donde hace falta dibujar la banda
histórica.

"En vivo" significa siempre lo mismo en esta app: un re-fit en memoria del
proceso FastAPI sobre los ~500 días más recientes de `price_history.parquet`
extendido con klines de Binance — nunca escribe a disco, y corre bajo un
semáforo (`_WHATIF_SLOTS`, máx. 2 fits concurrentes) por la memoria limitada
del host gratuito. MS-GARCH nunca se re-fitea en vivo (corre offline en R);
toda pantalla con fan-out de modelos lo excluye a propósito.

---

## `/` — Overview (`web/app/page.tsx`)

### MarketWatch (`web/components/MarketWatch.tsx`)
Tabla resumen por asset: precio, 24h, 30d, VaR, ES, VaR price, mejor modelo.

| Dato | Hook → endpoint | Cálculo | Naturaleza |
|---|---|---|---|
| Precio, 24h | `usePrice(asset)` → `/price/{asset}` | `D.live_price` (Binance `/api/v3/ticker/24hr`, ttl 15s) | precio de mercado |
| Sparkline 30d | `usePrices(asset,30)` → `/prices/{asset}?limit=30` | `D.load_price_window` | precio de mercado (extendido) |
| Mejor modelo | `useModelsComparison(asset,alpha)` → `/models/comparison` | `eval_fz0_mcs.csv` | congelado |
| VaR / ES (última fila) | `useBacktests(asset, best, alpha, 1, live=true)` → `/backtests?...&live=true` | `D.load_backtests_live()` | congelado + incremental |
| "VaR price" | (cliente) `anchor * exp(last.var)`, `anchor` = close del día anterior a la fila VaR (`MarketWatch.tsx` L28-33) | — | derivado en cliente |

### Selectores (Asset / Confidence / Model)
`useConfig()` → `/config` → `load_config()` (YAML de config). No numérico.

### Cards: spot, 24h high/low
`usePrice(asset)` → `/price/{asset}` → `D.live_price` → Binance. Precio de mercado puro.

### Cards: "Implied range" / "VaR floor" / "ES floor"
`useForecast(asset, alpha, model)` → `/forecast/{asset}?model=&alpha=` (`app.py::forecast`, L229-288).

- Si `D.today_forecast(...)` (`data.py` L410-465) devuelve resultado: **en vivo**
  — re-fit del modelo elegido sobre la ventana de 500 obs, `dist.var(alpha)`,
  `dist.es(alpha)`, `dist.ppf(1-alpha)`; precios = `last_close * exp(var/es/upper)`.
- Si no (modelo sin re-fit en vivo, p. ej. LSTM-Vol sin torch en el host): cae
  a la última fila de `D.load_backtests_live()` para ese (asset, model, alpha)
  — **congelado/incremental**, `source: "frozen_backtest"`.
- `RiskZoneCard` (zona Basel) usa `useCoverage` → `/coverage` →
  `eval_coverage.csv` (**congelado**), columna `basel_zone` ya calculada en el
  pipeline offline (no se recalcula acá).

### Frase narrativa ("Right now, {model} estimates...")
Mismos campos de `forecast` de arriba (var_price, es_price); solo texto.

### `WhyThisVar` (descompone el VaR en 3 piezas)
- `modelVar` = `useForecast(...)` (igual que arriba, en vivo).
- Línea base = `useBacktests(asset, "HS", alpha, 1)` **sin** `live=true` →
  solo `D.load_backtests()` (**congelado puro**).
- `shift = modelVar - baseVar` (cliente).
- Rank / in_mcs / gap vs. mejor = `useModelsComparison` (congelado
  `eval_fz0_mcs.csv`); `gap = row.fz0_mean - best.fz0_mean` (cliente).

### `ModelSpread` ("Model agreement today")
- `useModelsLatest(asset,alpha)` → `/models/latest` → `D.latest_by_model`
  (`data.py` L201-219): agrupa `load_backtests_live()` por modelo, toma la
  fila más reciente de cada uno. **Congelado + incremental.**
- `inMcs` de `useModelsComparison` (congelado).
- Filtro "fresh vs. stale": lógica de cliente (`page.tsx` L149-157),
  comparando `date` contra la más reciente de todas.

### Badge "Live re-fit" / "Frozen backtest"
Campo `forecast.source` de `/forecast/{asset}` (arriba).

### Bloque "VaR already hit today" (intradía)
`forecast.intraday`, de `_intraday()` (`app.py` L291-307) →
`D.intraday_status` → `live_tail.today_so_far` (`data.py` L289-292) —
**precio de mercado** (velas Binance de 5 min del día UTC en curso) contra
`last_close`; `var_breached = st.low_ret < var`.

### `PriceChart`
- Línea de precio: `usePrices(asset,200)` → `/prices/{asset}?limit=200` →
  `D.load_price_window` (precio de mercado extendido).
- Bandas VaR/ES: construidas en el cliente (`buildBand()`, `page.tsx` L39-61)
  = `priorClose * exp(row.var / row.es)` sobre `useBacktests(asset, model,
  alpha, 180, live=true)` (congelado + incremental).
- Puntos "live" sueltos: el forecast en vivo de arriba, dibujado aparte
  cuando el walk-forward aún no llegó a ese día (`liveBandPoints()`, L72-89).

### `ConeChart` ×2 (Jump-Diffusion, GARCH-EVT)
`forecast.cone`, de `_ensemble_cone()` (`app.py` L124-144, ttl 600s), sobre
`D.window_returns(asset)` (ventana en vivo de 500 retornos) vía
`cryptorisk.api.cone`:

- **Jump-Diffusion** (`cone.py::jump_diffusion_cone`, L60-83): Merton, 20k
  trayectorias Monte Carlo, `n_jumps ~ Poisson(lam*h)`,
  `sim = h*(mu - lam*k_bar) + sigma*sqrt(h)*eps + jump`; VaR/ES/upper son
  percentiles empíricos de esa simulación.
- **GARCH-EVT** (`cone.py::garch_evt_cone`, L86-119): GARCH(1,1)-t vía
  `arch`, varianza acumulada a h días (suma de forecasts por paso), cola GPD
  reescalada.

Ambos **en vivo**, independientes del modelo elegido en el selector.

### `RegimeDistribution` (curvas Student-t normal vs. crisis)
`forecast.regime_summary`, de `cone.py::regime_summary()` (L154-171) sobre
`msgarch_regime_params.csv` (**congelado**, del script R): `stat_vol`, `nu`,
`p_stay` por régimen (ratio crisis/normal acotado a [1.4, 3.0]). Las curvas
en sí las dibuja el cliente (`web/lib/studentT.ts::scaledStudentTDensity`) —
son una densidad matemática parametrizada por esos números, no datos crudos
del servidor.

---

## `/capital` — Capital & Decision

Todo **congelado**, salvo `BreachHeadroom` (mixta).

| Dato | Hook → endpoint | Fuente | Naturaleza |
|---|---|---|---|
| Capital stack + tabla | `useCapital(asset)` → `/capital` | `decision_capital.csv`; `basel_zone` se recalcula en el endpoint con `basel_zone_and_addon()` | congelado |
| Estimation-risk bars | `useEstimationRisk(asset)` → `/estimation-risk` | `decision_estimation_risk.csv` | congelado |
| Position limits | `useLimits(asset)` → `/limits` | `decision_limits.csv` | congelado |
| Perp hedge cards | `useHedge(asset)` → `/hedge` | `decision_hedge.csv` | congelado |

El modelo mostrado en el stack = el de menor `capital_usd` entre los
`in_mcs` (o el primero si ninguno califica).

### `BreachHeadroom` (`web/components/BreachHeadroom.tsx`)
- Zona Basel / excepciones / headroom: `useBaselHeadroom(asset)` →
  `/decision/basel-headroom` → `D.basel_headroom()` (`data.py` L150-198) —
  sobre `load_backtests_live()` (congelado+incremental), cuenta excepciones
  99% de los últimos 250 días, `basel_zone_and_addon(x)` / `(x+1)`, capital
  con `es_capital(...)` de `decision.capital`. **Cálculo en vivo sobre datos
  incrementales**, no lee `decision_capital.csv`.
- "Distance to breach": `useBreachDistanceAll(asset, models)` → fan-out a
  `/decision/breach-distance/{asset}?model=&alpha=0.01` por modelo →
  `breach_distance()` (`app.py` L385-406) → `D.today_forecast` (**en vivo**)
  + `D.live_price` (**precio de mercado**) →
  `dist_var = var_price/spot - 1`, `dist_es = es_price/spot - 1`.

### `ModelTip` (usado en toda la app)
`useModelsInfo()` → `/models/info` → texto estático de
`cryptorisk.models.notes` (`MODEL_NOTES`, `METHOD_FAMILY`). No numérico.

---

## `/explain` — Explainability

| Sección | Hook → endpoint | Fuente | Naturaleza |
|---|---|---|---|
| 1. Calibración (`BacktestBand`, `CumulativeViolations`, `RollingHitRate`, `PitHistogram`) | `useBacktests(asset, model, alpha, days)` **sin** `live=true` → `/backtests` | `D.load_backtests()` | congelado |
| 2. Coverage / FZ0 (`HitRateBars`, `FamilyRankChart`) | `useCoverage` / `useModelsComparison` | `eval_coverage.csv` / `eval_fz0_mcs.csv` | congelado |
| 3a. RF-QR importancia/diagnóstico/inputs | `useRfExplain(asset)` → `/explain/rf` | `explain_rf_importance.csv`, `explain_rf_diagnostics.csv`, `explain_rf_inputs.csv` (offline, refit cada 20 días OOS) | congelado |
| 3b. `RfPdpChart` | `useRfPdp(asset, feature, alpha)` → `/explain/rf/pdp` → `D.rf_partial_dependence()` (`data.py` L378-406) | re-fit de RF-QR sobre ventana actual, barre un feature dejando el resto fijo | **en vivo** |
| 4. LSTM importancia | `useLstmExplain(asset)` → `/explain/lstm` | `explain_lstm_importance.csv`, agregado mean/std/count por (lag,feature) en el endpoint | congelado |
| 4b. `LstmLocalPanel` (ocultación por día) | `useLstmLocal(asset,alpha)` → `/explain/lstm/local` | `explain_lstm_local.csv` (el host de la API no puede entrenar la red; corre offline con `study.run_lstm_local`) | congelado/precomputado |
| 5. `VarChangePanel` | `useVarChangeAll` (fan-out) → `/explain/var-change/{asset}` → `D.var_change_attribution()` (`data.py` L556-599) | 4 re-fits (prev/plus/minus/now), descompone el cambio del VaR en "día nuevo entra" vs. "día viejo sale" (`new = 0.5*((plus-prev)+(now-minus))`) | **en vivo** |
| 6. `ConformalPanel` | `useConformalSummary` / `useConformalPath` | `conformal_summary.csv`, `conformal_level_path.csv` (de `study.run_conformal`) | congelado |

---

## `/live` — Live Record

Todo **incremental**: solo `backtests_live.parquet` (los días desde el fin de
la muestra congelada del estudio), nunca las tablas de evaluación offline.

`useLiveTrackRecord(asset,alpha)` → `/live/track-record` →
`D.live_track_record()` (`data.py` L92-147), por modelo:

- `violations` = suma de `violation`; `expected = n*alpha`; `hit_rate = k/n`.
- `p_at_least`: test binomial (`stats.binomtest`).
- `mean_fz0`: `fz0_loss()` de `backtest.scoring`.
- `min_margin = min(realized - var)`.
- Por día (agregado sobre todos los modelos): `var_min/max/median`, `es_median`.

`RealizedVsBand` / `LiveFz0Bars` (`web/components/LiveCharts.tsx`) grafican
esas filas directamente.

---

## `/models` — Model Comparison

Todo **congelado**.

| Tabla | Endpoint | Fuente |
|---|---|---|
| FZ0 / MCS | `/models/comparison` | `eval_fz0_mcs.csv` |
| Coverage (Kupiec/Christoffersen/DQ) | `/coverage` | `eval_coverage.csv` |
| ES tests (Acerbi-Szekely Z1/Z2) | `/es-tests` | `eval_es.csv` |
| GW-CPA (Giacomini-White) | `/gw-cpa` | `eval_gw_cpa.csv` |

Todas las columnas ya vienen calculadas por el pipeline offline; la API solo lee.

---

## `/portfolio` — Portfolio

- Tabla principal + copulas/directs: `usePortfolioEval(alpha)` →
  `/portfolio/eval` → `portfolio_eval.csv` filtrado a `asset=="PORTFOLIO"`
  (**congelado**, de `study.run_portfolio`).
- Bullets "Read": `usePortfolioNarrative(alpha)` → `/portfolio/narrative` →
  reusa `_read_bullets()` de `study.run_portfolio` sobre las mismas filas
  (reformatea, no calcula de nuevo).
- "Live basket composition": `usePortfolioComposition()` →
  `/portfolio/composition` → pesos de `config.yaml` + `D.live_price(a)` por
  activo (**precio de mercado**).
- "Basket 24h change": 100% cliente, `Σ weight[a] * change_pct[a]`
  (`page.tsx` L101-103).

---

## `/regimes` — Regimes

`useRegimes(asset,3000)` → `/regimes/{asset}` → `regimes()` (`app.py`
L661-674), agrega 4 fuentes:

| Campo | Fuente | Naturaleza |
|---|---|---|
| `series` (`D.regime_series()`, L295-305) | `msgarch_pred_{asset}.csv` (script R) | congelado |
| `correlations` | `regime_identification.csv` | congelado |
| `regime_summary` | `cone.regime_summary()` → `msgarch_regime_params.csv` | congelado |
| `stats` (`D.regime_stats()`, L308-350) | combina `msgarch_pred_{asset}.csv` + precio extendido; separa Normal/Crisis por `P>0.5`, calcula mean/sd/skew/kurt/vol_ratio/test de Levene | **en vivo** (cacheado 300s) |
| `band` (`D.pcrisis_band()`, L353-358) | `msgarch_pcrisis_band_{asset}.csv` (bootstrap de `msgarch/bootstrap_pcrisis.R`) | congelado |

`usePrices(asset,3000)` (precio de mercado extendido) alimenta la serie de
precio y `rollingAbsMean`/`rollingVol` (media móvil 21d, 100% cliente,
`page.tsx` L31-52).

`DurationHistogram` calcula las rachas (run-lengths) de la serie booleana
`P>0.5` enteramente en el cliente.

---

## `/whatif` — What-if

Todo **en vivo**, nunca toca `data/results/`.

`useWhatIfAll(asset, models, shock, alpha)` (fan-out, ≤4 en paralelo) → por
modelo `/whatif/{asset}?model=&shock=&alpha=` → `whatif()` (`app.py`
L310-350):

- `base` = `D.today_forecast(...)` (forecast en vivo actual).
- `hit` = `D.whatif_forecast(asset, model, log1p(shock), alphas)`
  (`data.py` L534-552): re-fit sobre la ventana menos el último día, más 1
  día hipotético con retorno `shock` apendeado. Las variables realizadas de
  ese día hipotético se estiman en `_shock_day_realized()` (`data.py`
  L468-484): `rv = shock²`, `bv = rv * ratio_mediana_bv/rv_ventana`,
  `rsv_pos/neg` según el signo, `jump = max(rv-bv, 0)`, `rq = shock⁴/3`.
- `flat` = mismo mecanismo con `shock=0` (referencia "día plano").
- Respuesta: `baseline`, `flat`, `shocked` (var/es);
  `baseline_var_price`/`shocked_var_price` = `_price(last_close, var)` /
  `_price(shocked_close, var)`.

`ReactionBars` y la tabla son cliente puro sobre esas filas: `dpp = shocked -
flat`, `effect = shocked/flat`.

MS-GARCH queda excluido explícitamente (no re-fiteable en vivo).

---

## Tabla resumen: qué archivo alimenta qué página

| Archivo (`data/results/` salvo nota) | Páginas |
|---|---|
| `price_history.parquet` (+ klines Binance incrementales vía `live_tail`) | Overview, Portfolio (composition), Regimes (precio+stats), What-if/Explain (ventanas de re-fit), MarketWatch |
| Binance `/ticker/24hr` (en vivo, no parquet) | Overview (spot/24h), Portfolio (composition), MarketWatch |
| `backtests.parquet` | Overview (banda histórica), WhyThisVar (HS), Explain (backtests), Capital/BreachHeadroom (base congelada) |
| `backtests_live.parquet` | Overview (banda+fallback de forecast), MarketWatch, Live, Capital/BreachHeadroom (headroom en vivo), `/models/latest` |
| `eval_fz0_mcs.csv` | Overview (comparison/WhyThisVar), Models, Explain (family rank), MarketWatch |
| `eval_coverage.csv` | Overview (zona Basel), Models, Explain |
| `eval_es.csv` | Models (ES tests) |
| `eval_gw_cpa.csv` | Models (GW-CPA) |
| `decision_capital.csv` | Capital |
| `decision_estimation_risk.csv` | Capital |
| `decision_limits.csv` | Capital |
| `decision_hedge.csv` | Capital |
| `portfolio_eval.csv` | Portfolio |
| `regime_identification.csv` | Regimes (correlations) |
| `msgarch_pred_{asset}.csv` | Regimes (series, stats) |
| `msgarch_regime_params.csv` | Overview (RegimeDistribution), Regimes (TransitionMatrix, summary) |
| `msgarch_pcrisis_band_{asset}.csv` | Regimes (band) |
| `explain_rf_importance/diagnostics/inputs.csv` | Explain (sección RF-QR) |
| `explain_lstm_importance.csv` | Explain (sección LSTM) |
| `explain_lstm_local.csv` | Explain (LstmLocalPanel) |
| `conformal_summary.csv`, `conformal_level_path.csv` | Explain (sección ACI) |
| *(ninguno — re-fit en memoria vía `models.registry`)* | What-if, VarChangePanel, ConeChart, RfPdpChart, BreachHeadroom (breach-distance), Overview forecast (cuando `source: live_refit`) |
| `config.yaml` (fuera de `data/results/`) | Selectores Asset/Alpha en todas las páginas, pesos de Portfolio |

---

## Notas generales

- `src/cryptorisk/api/data.py` es deliberadamente independiente de
  `cryptorisk.dashboard.data` (Streamlit): duplica la lógica a propósito en
  vez de compartir una capa común (ver su propio docstring).
- "Congelado" (`backtests.parquet`, los `eval_*.csv`, `decision_*.csv`,
  `explain_*` offline) es el registro científico del estudio original — no
  se toca nunca al añadir funcionalidad "en vivo".
- "Incremental" (`backtests_live.parquet`) es el mismo walk-forward, pero
  extendido día a día por el job de GitHub Actions desde que terminó el
  estudio.
- "En vivo" es siempre un re-fit real en memoria del proceso de la API, no
  una lectura de disco.
