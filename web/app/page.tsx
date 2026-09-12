"use client";

import { useMemo, useState } from "react";
import { Badge } from "@/components/Badge";
import { ConeChart, ConeSeries } from "@/components/ConeChart";
import { MetricCard } from "@/components/MetricCard";
import { PriceChart, PricePoint } from "@/components/PriceChart";
import { ForecastResponse } from "@/lib/api";
import { fmtConfidence, fmtDate, fmtPct, fmtUsd } from "@/lib/format";
import {
  useBacktests,
  useConfig,
  useForecast,
  useModels,
  useModelsComparison,
  usePrice,
  usePrices,
} from "@/lib/hooks";

function buildConeSeries(forecast: ForecastResponse | undefined): ConeSeries[] {
  const cone = forecast?.cone;
  if (!cone) return [];
  const series: ConeSeries[] = [
    { key: "jd", label: "Jump-Diffusion", color: "var(--amber)", points: cone.jump_diffusion ?? [] },
    { key: "evt", label: "GARCH-EVT", color: "var(--violet)", points: cone.garch_evt ?? [] },
  ];
  if (cone.crisis_scenario) {
    series.push({
      key: "crisis",
      label: "MS-GARCH crisis scenario",
      color: "var(--red)",
      dash: "4 3",
      points: cone.crisis_scenario,
    });
  }
  return series;
}

function buildBand(
  prices: { date: string; close: number }[],
  backtests: { date: string; var: number; es: number; violation: boolean }[],
) {
  const dates = prices.map((p) => p.date.slice(0, 10));
  const closeByDate = new Map(prices.map((p) => [p.date.slice(0, 10), p.close]));
  const varLine: PricePoint[] = [];
  const esLine: PricePoint[] = [];
  const breaches: PricePoint[] = [];
  const sorted = [...backtests].sort((a, b) => a.date.localeCompare(b.date));
  for (const row of sorted) {
    const d = row.date.slice(0, 10);
    const idx = dates.indexOf(d);
    const priorClose = idx > 0 ? prices[idx - 1].close : closeByDate.get(d);
    if (priorClose == null) continue;
    varLine.push({ time: d, value: priorClose * Math.exp(row.var) });
    esLine.push({ time: d, value: priorClose * Math.exp(row.es) });
    if (row.violation) {
      const close = closeByDate.get(d);
      if (close != null) breaches.push({ time: d, value: close });
    }
  }
  return { varLine, esLine, breaches };
}

export default function OverviewPage() {
  const { data: config } = useConfig();
  const assets = useMemo(() => config?.assets ?? [], [config]);
  const alphas = useMemo(() => config?.alphas ?? [], [config]);

  const [asset, setAsset] = useState<string | null>(null);
  const [alpha, setAlpha] = useState<number | null>(null);
  const [modelOverride, setModelOverride] = useState<string | null>(null);

  const effAsset = asset ?? assets[0] ?? null;
  const effAlpha = alpha ?? alphas[0] ?? 0.025;

  const { data: allModels } = useModels();
  const { data: comparison } = useModelsComparison(effAsset, effAlpha);
  const bestModel = comparison?.find((r) => r.is_best)?.model ?? null;
  const effModel = modelOverride ?? bestModel;

  const { data: price } = usePrice(effAsset);
  const { data: prices } = usePrices(effAsset, 200);
  const { data: forecast } = useForecast(effAsset, effAlpha, effModel);
  const { data: backtests } = useBacktests(effAsset, effModel, effAlpha, 180);

  const chartData = useMemo(() => {
    const priceLine: PricePoint[] = (prices ?? []).map((p) => ({
      time: p.date.slice(0, 10),
      value: p.close,
    }));
    const { varLine, esLine, breaches } = buildBand(prices ?? [], backtests ?? []);
    return { priceLine, varLine, esLine, breaches };
  }, [prices, backtests]);

  const coneSeries = useMemo(() => buildConeSeries(forecast), [forecast]);

  if (!config || !effAsset) {
    return <div className="text-muted">Loading config from the API...</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">&#9672; CUBO+ Risk Terminal</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Live spot price vs. the study&apos;s out-of-sample VaR/ES band. Backtest
          numbers are frozen at the last pipeline run; the band&apos;s model is
          re-fit on demand for today&apos;s forecast only.
        </p>
      </div>

      <div className="flex flex-wrap gap-4">
        <Field label="Asset">
          <select
            className="select"
            value={effAsset}
            onChange={(e) => setAsset(e.target.value)}
          >
            {assets.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Confidence">
          <select
            className="select"
            value={effAlpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
          >
            {alphas.map((a) => (
              <option key={a} value={a}>
                {fmtConfidence(a)}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Model (defaults to FZ0-best, in-MCS)">
          <select
            className="select min-w-[10rem]"
            value={effModel ?? ""}
            onChange={(e) => setModelOverride(e.target.value)}
          >
            {(allModels ?? []).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {price ? (
          <>
            <MetricCard
              label={`${effAsset}-USD (Binance spot)`}
              value={fmtUsd(price.price, 2)}
              delta={`${price.change_pct >= 0 ? "+" : ""}${price.change_pct.toFixed(2)}% 24h`}
              deltaColor={price.change_pct >= 0 ? "text-green" : "text-red"}
            />
            <MetricCard
              label="24h High / Low"
              value={`${fmtUsd(price.high)} / ${fmtUsd(price.low)}`}
            />
          </>
        ) : (
          <MetricCard label={`${effAsset}-USD`} value="loading..." />
        )}
        {forecast && (
          <>
            {forecast.upper_price != null && forecast.var_price != null ? (
              <MetricCard
                label={`Implied ${fmtConfidence(effAlpha)} range (next close)`}
                value={`${fmtUsd(forecast.var_price)} – ${fmtUsd(forecast.upper_price)}`}
              />
            ) : (
              <MetricCard
                label={`VaR floor (${fmtConfidence(effAlpha)})`}
                value={fmtUsd(forecast.var_price)}
                title="Model has no upper-tail quantile (e.g. CAViaR)."
              />
            )}
            <MetricCard
              label={`ES floor (${fmtConfidence(effAlpha)})`}
              value={fmtUsd(forecast.es_price)}
            />
          </>
        )}
      </div>

      {forecast && forecast.var_price != null && (
        <p className="rounded border border-grid bg-panel px-4 py-3 text-sm">
          Right now, {forecast.model} estimates a{" "}
          <span className="font-semibold text-amber">~{fmtPct(effAlpha, 1)} chance</span> that{" "}
          {effAsset} closes tomorrow below{" "}
          <span className="font-semibold">{fmtUsd(forecast.var_price)}</span>
          {forecast.es_price != null && (
            <>
              . If that happens, the expected loss beyond that point (ES) is
              around <span className="font-semibold">{fmtUsd(forecast.es_price)}</span>
            </>
          )}
          . The forward cone below extends this to 5/10/30 days using three
          different specialized models.
        </p>
      )}

      {forecast && (
        <div className="flex items-center gap-2 text-sm text-muted">
          {forecast.source === "live_refit" ? (
            <>
              <Badge kind="live">Live re-fit</Badge>
              <span>
                {forecast.model} refit on the latest 500 obs as of{" "}
                {fmtDate(forecast.asof)}, forecasting the next close. Not the
                study&apos;s frozen backtest number.
              </span>
            </>
          ) : (
            <>
              <Badge kind="frozen">Frozen backtest</Badge>
              <span>
                Live re-fit for {forecast.model} unavailable on the current
                cached window; showing the last computed out-of-sample row.
              </span>
            </>
          )}
        </div>
      )}

      <div className="rounded border border-grid bg-panel p-3">
        {chartData.priceLine.length > 0 ? (
          <PriceChart
            price={chartData.priceLine}
            varLine={chartData.varLine}
            esLine={chartData.esLine}
            breaches={chartData.breaches}
            livePrice={price?.price}
          />
        ) : (
          <div className="flex h-[420px] items-center justify-center text-muted">
            Loading chart data...
          </div>
        )}
      </div>
      <p className="text-xs text-muted">
        Amber/red lines are the {effModel ?? "selected model"} walk-forward
        VaR/ES for that day&apos;s close, plotted against the prior close.
        Markers = realized OOS violations (realized &lt; VaR).
      </p>

      {coneSeries.length > 0 && forecast?.last_close != null && (
        <ConeChart series={coneSeries} lastClose={forecast.last_close} />
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-muted">
      {label}
      {children}
    </label>
  );
}
