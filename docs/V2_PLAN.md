# v2 — Estudio comparativo de modelos de riesgo para criptoactivos

**Documento de arranque.** Define alcance, datos, modelos, protocolo de evaluación,
arquitectura y plan por fases. La v1 (este repo) queda como prueba de concepto;
la v2 es un estudio comparativo con rigor de *model validation*.

---

## 1. Objetivo y pregunta

**Pregunta central:** de las familias de modelos de volatilidad/cola aplicables a
criptoactivos, ¿cuál produce las mejores estimaciones de VaR y Expected Shortfall
para una posición de tesorería, medido fuera de muestra y con significancia
estadística?

**Entregable:** un informe de validación que (a) rankea los modelos con un Model
Confidence Set sobre pérdidas estrictamente consistentes, (b) documenta dónde
falla cada uno y por qué, (c) traduce la elección de modelo a capital, límites y
un análisis de cobertura, y (d) deja el pipeline reproducible con un comando.

**Fuera de alcance (v2):** trading/alpha, ejecución, un servicio en producción
24/7. La capa de decisión es analítica, no operativa.

### Hipótesis a contrastar (los "grupos de casos")

| Familia | Representante(s) | Qué se contrasta |
|---|---|---|
| No paramétrica | Historical Simulation, Age-weighted HS | Robusta pero lenta en reaccionar; referencia regulatoria |
| Suavizado exponencial | EWMA / RiskMetrics (λ=0.94) | Baseline de industria, un parámetro |
| GARCH un régimen | GARCH(1,1)-t, GJR-GARCH-t, EGARCH-t | Clustering + leverage; ¿agrupa violaciones? |
| Cambio de régimen | MS-GARCH 2 estados (sGARCH-t) | Mejor densidad in-sample; ¿se identifica fuera de muestra? |
| Semiparamétrica de cola | FHS (Filtered Historical Simulation), GARCH-EVT (McNeil–Frey) | Estándar de mesa; separa dinámica de vol de forma de cola |
| Discontinua | Merton jump-diffusion; saltos observados (Lee–Mykland) | Colas por saltos de un día; ¿sobre-ajusta al 97.5%? |
| Medidas realizadas (usa intradía) | HAR-RV, HARQ, Realized GARCH | ¿La vol realizada intradía mejora el pronóstico 1-día? |
| Cuantil directo | CAViaR, CAViaR-X | Sin supuesto distribucional; conditioning exógeno |
| Exógena / "-X" | GARCH-X con driver realizado u on-chain | ¿Aportan funding/OI/flujos al riesgo de cola? |

FHS, las medidas realizadas, CAViaR y los "-X" son las adiciones frente a v1.

---

## 2. Universo de datos

### Activos
- **Núcleo (Fase 1):** BTC y ETH, spot USD. El riesgo cripto es de portfolio; un
  solo activo no representa una tesorería real.
- **Extensión (Fase 7):** cesta (BTC, ETH, SOL, BNB) con pesos fijos → VaR de
  portfolio y dependencia de cola vía cópula.

### Bloques de datos

| Bloque | Frecuencia | Contenido | Fuente sugerida | Uso |
|---|---|---|---|---|
| **Precio diario** | 1d | OHLCV, log-retornos | dos fuentes independientes reconciliadas (índice CoinMetrics/CoinGecko + exchange agregado) | insumo de todos los modelos |
| **Intradía** | 5 min | klines BTCUSDT / ETHUSDT | `data.binance.vision` (histórico gratuito desde 2017) | RV, bipower variation, semivarianza, componente de salto, modelos realizados |
| **Contexto macro/on-chain** | 1d | DXY, S&P500, tasa Fed, CPI (lag), **hashrate, difficulty** | FRED + Yahoo + blockchain.com (heredado de v1) | **solo descriptivo**, nunca feature del VaR diario |
| **Microestructura de mercado** | 1d (o 8h) | funding rate de perpetuos, open interest, exchange netflows, cambio de oferta de stablecoins | Coinglass / CoinMetrics / Glassnode | bloque exógeno para modelos "-X", definición de sub-períodos de estrés, análisis de régimen |

**Nota sobre on-chain:** hashrate y difficulty son series muy persistentes y sin
poder predictivo sobre la volatilidad del retorno del día siguiente — se
conservan como contexto (igual que macro), no como regresor. La microestructura
(funding, OI, flujos, stablecoins) sí se relaciona con cascadas de liquidación y
riesgo de cola de corto plazo, y entra como bloque exógeno opcional.

### Limpieza de intradía
- Reglas estándar de limpieza de alta frecuencia (Barndorff-Nielsen, Hansen,
  Lunde & Shephard, 2009): quitar barras fuera de horario/con precio cero,
  outliers vs. mediana móvil, entradas duplicadas.
- Muestreo a 5 min (no tick) para acotar el ruido de microestructura; RV con
  submuestreo y kernel realizado (Zhou) como robustez.
- Solo el par USDT más líquido por activo.

### Point-in-time / calidad
- **Vintage:** cada extracción se guarda con timestamp; las revisiones de datos
  no se sobrescriben, se versionan. Un backtest a fecha *t* solo ve el vintage
  disponible en *t*.
- **Checks automáticos** (`quality.py`): gaps de calendario, |retorno| > 40%
  diario flaggeado, volumen cero / exchange caído, divergencia entre fuentes
  sobre umbral, saltos de nivel entre vintages.
- **Almacenamiento:** DuckDB (un archivo, sin servidor). Esquema:
  `prices_daily(asset, date, source, o,h,l,c, volume, vintage_ts)`,
  `returns_daily(asset, date, log_return)`,
  `bars_5m(asset, ts, o,h,l,c, volume)`,
  `realized_daily(asset, date, rv, bv, rsv_pos, rsv_neg, jump)`,
  `context_daily(date, dxy, spx, fed, cpi_lag, hashrate, difficulty)`,
  `microstructure_daily(asset, date, funding, oi, netflow, stbl_supply_chg)`.
  Parquet solo para exportes.

---

## 3. Modelos — especificación y referencias

Todos se estiman sobre log-retornos diarios (los realizados usan además las
medidas intradía), se enchufan al mismo motor walk-forward y devuelven la
densidad predictiva 1-paso (o al menos el par (VaR, ES)) para
**α ∈ {0.975, 0.99}** — FRTB usa ES 97.5%.

| ID | Modelo | Referencia |
|---|---|---|
| HS | Historical Simulation (ventana rodante) | Basel; Pritsker (2006) |
| AWHS | Age-weighted HS (decay 0.98–0.99) | Boudoukh, Richardson & Whitelaw (1998) |
| EWMA | RiskMetrics, σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}, λ=0.94 | J.P. Morgan (1996) |
| GARCH-t | GARCH(1,1), innovación t estandarizada | Bollerslev (1987) |
| GJR-t | GJR-GARCH(1,1)-t (leverage) | Glosten, Jagannathan & Runkle (1993) |
| EGARCH-t | EGARCH(1,1)-t | Nelson (1991) |
| GARCH-X | GARCH(1,1)-t + driver exógeno en la ecuación de varianza (RV rezagada, o funding/OI) | Engle (2002); Han & Kristensen (2014) |
| MSG | MS-GARCH 2 estados, sGARCH-t por régimen (vía R/MSGARCH) | Haas, Mittnik & Paolella (2004); Ardia et al. (2019) |
| FHS | GARCH(1,1)-t filtra σ_t; VaR = σ_{t+1}·Q_α(residuos empíricos, bootstrap) | Barone-Adesi, Giannopoulos & Vosper (1999) |
| GEVT | GARCH-EVT / McNeil–Frey: GARCH filtra; GPD-POT sobre residuos estandarizados | McNeil & Frey (2000) |
| MJD | Merton jump-diffusion, MLE de (λ, μ_J, σ_J) sobre la ventana (vectorizado) | Merton (1976) |
| HAR-RV | HAR sobre log-RV (diario, semanal, mensual) → σ²_{t+1} | Corsi (2009) |
| HARQ | HAR-RV con corrección por error de medición (realized quarticity) | Bollerslev, Patton & Quaedvlieg (2016) |
| RGARCH | Realized GARCH(1,1): ecuación de medición liga RV con σ²_t | Hansen, Huang & Shek (2012) |
| CAViaR | Cuantil condicional directo (SAV / asimétrico), sin distribución | Engle & Manganelli (2004) |
| CAViaR-X | CAViaR con regresor exógeno (RV, funding, netflow) | Engle & Manganelli (2004) ext. |

**Detección de saltos (intradía):** test de Barndorff-Nielsen & Shephard (2006)
[RV vs bipower variation] y Lee & Mykland (2008) para localizar saltos día a día.
Alimenta la calibración de MJD y una variante "saltos observados".

**Evaluación de vol como subproducto:** con intradía se puede evaluar el
pronóstico de σ²_{t+1} de *cada* modelo contra la RV realizada (§5.4), no solo
vía tests de VaR.

**Nota sobre MS-GARCH (hallazgo de v1):** buena densidad predictiva y VaR
competitivo, pero la **probabilidad de régimen filtrada no se identifica** en
ventanas rodantes de ~500 obs (corr ≈ 0 con |retorno| fuera de muestra; ≈ 0.74
in-sample con un único ajuste; specs restringidas y `FitMCMC` no lo rescatan).
En v2 se estudia formalmente (§5.5), no se esconde.

---

## 4. Motor de backtesting

Se reutiliza y endurece `risk_models/walk_forward.py`:

- **Esquema:** ventana rodante de longitud fija *W* **y** expanding (ambos
  reportados). Período OOS fijado de antemano y **congelado** antes de mirar
  resultados (evita p-hacking de la ventana).
- **Re-estimación:** `refit_every` por modelo, justificado y documentado (GARCH y
  familia: 1; MS-GARCH: 20 con re-filtrado diario vía `newdata`; jumps: 5; HAR:
  1, es OLS y es barato).
- **Salida:** para cada (modelo, activo, α, fecha): VaR, ES, σ²_{t+1}, retorno
  realizado, y —cuando el modelo lo permite— la densidad predictiva completa
  (para PIT y scoring).
- **Sin lookahead:** el modelo en *t* solo ve datos ≤ *t*−1 y el vintage de *t*−1.
- **Paralelización:** el walk-forward por (modelo, activo) es *embarrassingly
  parallel* → `joblib` / multiprocessing.

---

## 5. Protocolo de evaluación

Lo que separa un estudio serio de "corrí unos backtests". Cinco capas.

### 5.1 Backtests de cobertura de VaR (nivel de rechazo 5%)

| Test | H₀ | Estadístico |
|---|---|---|
| **Kupiec UC** (POF) | tasa de violación = α | LR_uc ~ χ²(1) |
| **Christoffersen IND** | violaciones independientes (no clustering 1-lag) | LR_ind ~ χ²(1) |
| **Christoffersen CC** | UC + IND conjuntamente | LR_cc = LR_uc + LR_ind ~ χ²(2) |
| **Engle–Manganelli DQ** | Hit_t no predecible por su historia ni por VaR_t | Wald ~ χ²(q) |
| **Semáforo de Basilea** | zona verde/amarilla/roja por nº de excepciones en 250d | conteo → multiplicador |

DQ es el más discriminante: `Hit_t = 1{r_t < VaR_t} − α` se regresa sobre
`[1, Hit_{t-1..t-p}, VaR_t]`; bajo H₀ todos los coeficientes son 0.

### 5.2 Backtests de Expected Shortfall

El ES no es *elicitable* de forma aislada:

- **Acerbi–Székely (2014), test Z₂** (recomendado):

  Z₂ = ( 1 / (Nα) · Σ_t [ r_t · 1{r_t < VaR_t} / ES_t ] ) + 1

  Bajo H₀ (ES bien especificado), E[Z₂] = 0; p-valor por simulación bajo la
  densidad predictiva del modelo. Z₁ como complemento (condicionado a violación).
- **ES-regression backtest** (Bayer & Dimitriadis, 2022): más potente; usa la
  representación conjunta (VaR, ES).

### 5.3 Ranking comparativo con significancia — el núcleo

Ordenar modelos y decir si la diferencia es real.

- **Pérdida conjunta FZ0** (Fissler–Ziegel; Patton, Ziegel & Chen, 2019),
  estrictamente consistente para (VaR, ES) al nivel α. Con v_t = VaR_t < 0,
  e_t = ES_t < 0:

  L^{FZ0}_t = (1/(α·e_t))·1{r_t ≤ v_t}·(v_t − r_t) + (v_t/e_t) + ln(−e_t) − 1

  (menor = mejor). Es la serie de pérdida que se compara.
- **Diebold–Mariano** pareado sobre {L^{FZ0}_t} para "modelo A vs B".
- **Model Confidence Set** (Hansen, Lunde & Nason, 2011): de la matriz de
  pérdidas {L_{i,t}}, con bloque-bootstrap estacionario, devuelve el conjunto de
  modelos que contiene al "mejor" con confianza 90%. **Resultado principal del
  informe.**
- **CPA condicional** (Giacomini–White, 2006) para condicionar el ranking al
  régimen (calma vs. estrés), usando la microestructura como condicionante.

### 5.4 Evaluación del pronóstico de volatilidad (habilitada por intradía)

- Proxy de verdad: RV realizada 5-min (con submuestreo).
- Pérdidas robustas al proxy: **QLIKE** y MSE sobre (σ²_{t+1} pronosticada, RV_{t+1}).
- MCS también sobre estas pérdidas → ¿qué modelos pronostican mejor la varianza,
  independientemente del VaR?
- Regresión de Mincer–Zarnowitz de RV sobre σ² pronosticada (α=0, β=1).

### 5.5 Sub-períodos y estudio de identificación MS-GARCH

- **Sub-períodos** definidos *ex-ante*: COVID (2020-03), Luna/UST (2022-05), FTX
  (2022-11), y ventanas de calma comparables. Repetir 5.1–5.4; reportar si el MCS
  cambia.
- **Identificación de régimen vs. longitud de ventana** (contribución
  metodológica): para *W* ∈ {500, 750, 1000, 1500, 2000, expanding}, medir
  corr(P(alta vol)_filtrada, |r|) y corr con RV mensual, in-sample vs
  walk-forward; añadir specs restringidas (regímenes comparten α, β) y `FitMCMC`.
  Salida: curva "cuántos datos necesita el modelo de 2 regímenes para separar los
  estados". (v1 ya tiene el andamiaje: `regime_window_exp.R`,
  `constrained_regime.R`.)

### 5.6 Diagnósticos de densidad
- **PIT:** u_t = F_t(r_t); bajo modelo correcto u_t ~ U(0,1) iid.
- **Berkowitz** sobre Φ⁻¹(u_t) (LR de normalidad + independencia).

---

## 6. Capa de decisión — versión completa

Traducir el mejor modelo (y el peor del MCS) a números de gestión de riesgo.

### 6.1 Capital (estilo FRTB — Internal Models Approach)
- **ES 97.5%** con **liquidity horizons**: para cripto, LH ≥ 10 días hábiles;
  escalado √(LH/10)·add-ons por factor iliquido. Reportar ES 10d por raíz-del-
  tiempo **y** por simulación de trayectorias, comparados.
- **Multiplicador interno** m_c (Basilea: 1.5 + add-on por zona del semáforo).
  Capital = m_c · ES_{97.5%, LH}.
- **Add-on de riesgo de modelo:** reserva = diferencia de capital entre el mejor
  y el peor modelo del MCS (o percentil alto de la distribución de capital sobre
  modelos admisibles). Análogo a una AVA de *model risk* (prudent valuation,
  Reg. Delegada UE 2016/101).

### 6.2 Marco de límites
- Límite de posición nocional *N\** tal que `ES_{99%, 1d}(N*) ≤ presupuesto de
  riesgo`. Tabla de *N\** por modelo.
- **Utilización:** serie temporal de `ES_{99%,1d}(N_t) / presupuesto`.
- **Backtest del marco de límites:** sobre el período OOS, ¿cuántas veces se
  habría excedido el límite fijado con cada modelo? ¿Coinciden los excesos con
  pérdidas reales grandes? Un marco que "nunca se acerca" es tan malo como uno
  que se viola siempre.

### 6.3 P&L attribution
- Descomposición del P&L diario simulado en: drift, difusión (σ), salto,
  residual.
- **PLA test estilo FRTB:** ratios de Spearman y Kolmogorov–Smirnov entre el P&L
  del modelo (RTPL) y el P&L "real" (HPL, aquí el retorno realizado × nocional).
  Zona verde/amarilla/roja. Mide si el modelo *explica* el P&L, no solo si acota
  la cola.

### 6.4 Cobertura con perpetuos
- **Ratio de cobertura:** minimum-variance (β = Cov(spot, perp)/Var(perp)) y uno
  que minimiza el ES de la cartera cubierta.
- **Costo:** funding rate anualizado del perpetuo sobre el nocional cubierto.
- **Beneficio:** reducción de ES_{97.5%} y de capital, cubierto vs. no cubierto.
- **Riesgo de base:** desviación spot–perp, su vol, y su comportamiento en los
  sub-períodos de estrés.

---

## 7. Arquitectura del repositorio

```
cryptorisk/
  data/
    ingest/
      prices_daily.py       # 2 fuentes, reconciliación, vintage
      binance_klines.py     # 5-min desde data.binance.vision
      context.py            # macro + hashrate/difficulty (descriptivo)
      microstructure.py     # funding, OI, netflows, stablecoins
    realized.py             # RV, BV, semivarianza, salto (BNS / Lee–Mykland)
    store.py                # DuckDB: esquemas de §2
    quality.py              # checks + reporte de anomalías
  models/
    base.py                 # Protocolo: fit_predict(ctx) -> PredictiveDist
    hs.py awhs.py ewma.py
    garch.py garch_x.py gjr.py egarch.py
    msgarch_bridge.py       # subprocess + CSV al script R (de v1)
    fhs.py garch_evt.py jump.py
    har.py harq.py realized_garch.py
    caviar.py
  backtest/
    engine.py               # walk-forward de v1, endurecido + paralelo
    coverage.py             # Kupiec, Christoffersen, DQ, Basilea
    es_tests.py             # Acerbi–Székely Z1/Z2, ES-regression
    scoring.py              # FZ0, QLIKE, Diebold–Mariano, MCS, Giacomini–White
    pit.py                  # PIT + Berkowitz
  study/
    run_backtests.py        # orquesta todo -> data/results/
    vol_forecast_eval.py
    subperiods.py
    regime_identification.py
    report.py               # tablas/figuras del informe
  decision/
    capital.py              # ES-IMA, LH, multiplicador, add-on de modelo
    limits.py               # N*, utilización, backtest del marco
    pnl_attribution.py      # PLA test
    hedge.py                # ratio, costo funding, ES cubierto, base
  config/
    study.yaml              # activos, período OOS, ventanas, α, refit_every, semillas
  msgarch/                  # scripts R (de v1)
  tests/                    # unit test por cada test estadístico
  docs/
    methodology.md          # el informe escrito
    model_cards/            # ficha por modelo: supuestos, estimación, límites
  pyproject.toml            # deps PINNEADAS (uv o poetry)
  Makefile                  # make data | realized | backtest | report
```

**Reglas:**
- Toda aleatoriedad con semilla fija en `config/study.yaml`.
- El pipeline entero corre con `make report` y es determinista.
- Cada test estadístico tiene un unit test contra un caso con respuesta conocida
  (Kupiec sobre Bernoulli simulada; FZ0 contra el óptimo teórico; MCS sobre
  modelos con orden conocido).
- R se aísla detrás de `msgarch_bridge.py` (subprocess + CSV, patrón de v1);
  **no** se reescribe MS-GARCH en Python.
- `PredictiveDist`: interfaz común con `.var(alpha)`, `.es(alpha)`, `.sigma2()`,
  y opcionalmente `.cdf(x)` / `.ppf(u)` para PIT y scoring de densidad.

---

## 8. Plan por fases

| Fase | Contenido | Criterio de "hecho" |
|---|---|---|
| **0 — Setup** | `pyproject.toml` pinneado, estructura de repo, CI mínima, `config/study.yaml`, semillas | `make test` verde en checkout limpio |
| **1 — Datos** | Ingesta BTC+ETH diario (2 fuentes) + klines 5-min + contexto + microestructura; store DuckDB con vintage; `realized.py`; `quality.py` | Series reconciliadas 2018→hoy, RV/BV/salto calculados, reporte de calidad sin flags sin explicar |
| **2 — Modelos + motor** | Portar HS/EWMA/GARCH-t/GEVT/MJD de v1; añadir AWHS, GJR, EGARCH, FHS, GARCH-X, HAR, HARQ, RGARCH, CAViaR(-X); MS-GARCH vía puente R; interfaz `PredictiveDist` | Cada modelo produce (VaR, ES, σ², densidad) 1-paso para BTC y ETH, con unit test de sanidad |
| **3 — Backtesting** | `coverage.py` (+ DQ), `es_tests.py`, `scoring.py` (FZ0, **MCS**), `vol_forecast_eval.py` (QLIKE vs RV), PIT/Berkowitz | MCS del par (VaR,ES) y MCS de vol, para BTC y ETH, α∈{97.5,99}% |
| **4 — Sub-períodos + identificación** | Ranking condicional calma/estrés (CPA con microestructura); estudio W vs identificación de régimen | Curva de identificación + tabla de MCS por sub-período |
| **5 — Decisión** | `capital.py`, `limits.py` (+ backtest del marco), `pnl_attribution.py` (PLA test), `hedge.py` | Capital, N* y PLA por modelo; add-on de riesgo de modelo en USD; análisis de hedge cubierto vs no |
| **6 — Informe** | `docs/methodology.md` + model cards + figuras; visor de resultados opcional | Documento revisable por un supervisor; reproducible desde cero |
| **7 (opcional) — Portfolio** | Cesta multi-activo, VaR de portfolio, cópula para dependencia de cola | VaR de cesta con las cinco capas de tests |

Fases 0–3 = MVP publicable. 4–6 = nivel *model validation*. 7 = extensión de
investigación.

---

## 9. Qué se reutiliza de v1

- **Motor walk-forward** (`walk_forward.py`) — endurecer + paralelizar, no reescribir.
- **GARCH-EVT** (`evt.py`, McNeil–Frey) — pasa casi tal cual.
- **Tests de cobertura** (`backtest.py`: Kupiec, Christoffersen) — correctos;
  añadir DQ y semáforo.
- **Puente R/MSGARCH** (`fit_msgarch_walkforward.R` + `run_backtest_msgarch.py`) —
  patrón subprocess+CSV, `normalize_sign`, `State()$FiltProb`, `prob_crisis_insample`.
- **Estimación de saltos vectorizada** (`jump_diffusion.py`).
- **Andamiaje del estudio de régimen** (`regime_window_exp.R`, `constrained_regime.R`).
- **ETL de contexto** (`fetch_btc/macro/onchain.py`) → reclasificado a bloque
  descriptivo.
- **Convenciones:** signo de VaR, filtro 2018+, descarte del día incompleto,
  reconfiguración UTF-8, coerción numérica de CSVs de R.
- **Hallazgos documentados** (README §Limitaciones) → sección "lecciones de v1" +
  model cards.

Se **deja atrás:** el dashboard Streamlit multipágina como entregable central
(pasa a visor de resultados de Fase 6, si acaso), y la mezcla
descriptivo/predictivo en una sola vista.

---

## 10. Riesgos del proyecto

| Riesgo | Mitigación |
|---|---|
| Calidad de datos intradía (prints malos, cortes de exchange) | Reglas de limpieza BNHLS (2009); par USDT líquido; muestreo 5-min; kernel realizado como robustez |
| MCS poco potente con pocas obs OOS | Período largo (2019→hoy ≈ 2600 días × 2 activos); bloque-bootstrap; reportar potencia |
| Runtime alto (≈15 modelos × 2 activos × 2 ventanas × refit diario) | Walk-forward paralelo por (modelo, activo); cachear estimaciones costosas; MS-GARCH con refit_every=20 |
| MS-GARCH en Python sin R | Se mantiene el puente R; subprocess aislado, ya funciona |
| Sobre-ingeniería de la capa de datos | DuckDB + un módulo por fuente; nada de microservicios |
| Alcance que crece | Fases 0–3 congeladas como MVP; 4+ solo con 0–3 cerradas |
| Microestructura (funding/OI) con historia corta o cara | Es bloque **opcional**; los modelos "-X" y el CPA condicional degradan a "sin exógeno" si falta |

---

*Siguiente paso: revisar/ajustar este documento, luego Fase 0 — estructura de
repo, `pyproject.toml` pinneado, `config/study.yaml`, CI.*
