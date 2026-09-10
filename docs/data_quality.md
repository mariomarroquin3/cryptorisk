# Data layer & quality report (Phase 1)

## Sources

| Block | Source | Notes |
|---|---|---|
| Daily price (canonical) | **CoinMetrics** community `PriceUSD` | Multi-venue USD **reference rate**, independent of any single exchange. This is the series returns are computed from. |
| Daily price (cross-check) | **Binance** spot klines (`BTCUSDT`, `ETHUSDT`) | Exchange OHLCV. Used for the divergence check and for volume. |
| Intraday | **Binance** 5-minute bars (`data.binance.vision`) | ~910k bars/asset, 2018-01 → last complete month. |
| Realized measures | derived from the 5-min bars | RV (native + subsampled), bipower variation, realized semivariance ±, jump (BNS Z-test). |
| Context — on-chain | **blockchain.info** | hashrate, difficulty. Descriptive only, never a VaR feature. |
| Context — macro | **Yahoo** (`^GSPC`, `DX-Y.NYB`) | SPX, DXY. Fed funds / CPI come from FRED and stay `NULL` where FRED is unreachable (optional context). |
| Microstructure | **Binance** perp funding + open interest | Funding back to contract listing (~2019-09); OI ~30 days. `netflow` / `stbl_supply_chg` need a paid provider → `NULL` for now; the `-X` models degrade to "no exogenous block" (V2_PLAN §10). |

Store: DuckDB (`data/store/cryptorisk.duckdb`). `prices_daily` rows carry
`vintage_ts` for point-in-time reads. All writers are idempotent.

Row counts after the first full ingest (2018-01-01 → 2026-09):
`prices_daily` 12,698 · `returns_daily` 6,350 · `bars_5m` 1,819,828 ·
`realized_daily` 6,330 · `context_daily` 3,175 · `microstructure_daily` 6,350.

## Quality report

`python -m cryptorisk.study.run_ingest --quality-only` regenerates
`data/results/quality_report{,_explained,_unexplained}.csv`. The Phase 1
done-criterion is **`quality_report_unexplained.csv` empty**.

First full ingest: **71 flags, all allow-listed, 0 unexplained.**

### `source_divergence` (51) — all before 2019-07

Binance quotes Tether pairs (`BTCUSDT`). USDT traded at a discount to USD
through 2018 (down to ~0.92 in October 2018) and re-depegged around the NY
Attorney General's action against Bitfinex/Tether (2019-04-25). So
`Binance close (USDT) / CoinMetrics (USD)` diverged by the USDT/USD basis, up to
~6%. From mid-2019 the basis is negligible. The canonical return series uses the
CoinMetrics USD reference, so the study is unaffected — the flag confirms that
choice was the right one.

The check flags a day only when the two sources disagree by **more than 2% and
more than that day's own return**. On a big-move day an exchange close and a
reference-rate snapshot naturally differ by a few percent (few-minute timing
offset); that is not a data error and is not flagged.

### `few_intraday_bars` (18 = 9 days × 2 assets) — all 2018-2020

Binance maintenance / unscheduled outages: the multi-day February 2018 upgrade
(2018-02-08 has 6/288 bars), plus scheduled windows in 2018-2020. The daily
return series (CoinMetrics reference) does not depend on Binance uptime.
Realized measures on these days are computed on partial bars and stay flagged so
the realized-model evaluation can down-weight them.

### `extreme_return` (2) — 2020-03-12

COVID "Black Thursday": BTC −47%, ETH −57% close-to-close, confirmed across
venues. A genuine tail event — exactly what the models must handle — not a bad
print.

## Allow-list

`config/quality_allowlist.yaml`. Each rule matches on `kind` plus any of
`dates` / `before` / `after` / `asset`, and **must** carry a `reason`.
`quality.apply_allowlist` splits the report into explained / unexplained.
