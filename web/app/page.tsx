"use client";

import { Suspense, useMemo, useState } from "react";
import { Badge } from "@/components/Badge";
import { ConeChart, ConeSeries } from "@/components/ConeChart";
import { Field } from "@/components/Field";
import { MarketWatch } from "@/components/MarketWatch";
import { MetricCard } from "@/components/MetricCard";
import { ModelSpread } from "@/components/ModelSpread";
import { PriceChart, PricePoint } from "@/components/PriceChart";
import { RegimeDistribution } from "@/components/RegimeDistribution";
import { RiskZoneCard } from "@/components/RiskZoneCard";
import { ChartSkeleton, MetricCardSkeleton, Skeleton } from "@/components/Skeleton";
import { WhyThisVar } from "@/components/WhyThisVar";
import { ForecastResponse } from "@/lib/api";
import { fmtConfidence, fmtDate, fmtPct, fmtUsd } from "@/lib/format";
import {
  useBacktests,
  useConfig,
  useCoverage,
  useForecast,
  useModels,
  useModelsComparison,
  useModelsLatest,
  usePrice,
  usePrices,
} from "@/lib/hooks";
import { useQueryParam } from "@/lib/useQueryParam";

function buildConeSeries(forecast: ForecastResponse | undefined): ConeSeries[] {
  const cone = forecast?.cone;
  if (!cone) return [];
  return [
    { key: "jd", label: "Jump-Diffusion", color: "var(--amber)", points: cone.jump_diffusion ?? [] },
    { key: "evt", label: "GARCH-EVT", color: "var(--violet)", points: cone.garch_evt ?? [] },
  ];
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

const DAY_MS = 86_400_000;
const addDays = (d: string, n: number) =>
  new Date(Date.parse(d + "T00:00:00Z") + n * DAY_MS).toISOString().slice(0, 10);

/** The walk-forward band stops at the last computed day. If that is the day
 * before today's forecast, extend the line with the live re-fit; otherwise
 * (band older than that) return the live re-fit as detached points -- a line
 * across the uncovered days would pass for a forecast. */
function liveBandPoints(
  band: { varLine: PricePoint[]; esLine: PricePoint[] },
  forecast: { source: string; asof?: string; var_price?: number | null; es_price?: number | null } | undefined,
): { liveVar: PricePoint | null; liveEs: PricePoint | null } {
  const none = { liveVar: null, liveEs: null };
  if (forecast?.source !== "live_refit" || !forecast.asof || forecast.var_price == null || forecast.es_price == null) return none;
  const liveDate = addDays(forecast.asof.slice(0, 10), 1);
  const last = band.varLine[band.varLine.length - 1]?.time as string | undefined;
  if (last && last >= liveDate) return none;
  const v = { time: liveDate, value: forecast.var_price };
  const e = { time: liveDate, value: forecast.es_price };
  if (last && addDays(last, 1) === liveDate) {
    band.varLine.push(v);
    band.esLine.push(e);
    return none;
  }
  return { liveVar: v, liveEs: e };
}

export default function OverviewPage() {
  return (
    <Suspense>
      <OverviewPageInner />
    </Suspense>
  );
}

function OverviewPageInner() {
  const { data: config } = useConfig();
  const assets = useMemo(() => config?.assets ?? [], [config]);
  const alphas = useMemo(() => config?.alphas ?? [], [config]);

  const [asset, setAsset] = useQueryParam("asset");
  const [alphaStr, setAlphaStr] = useQueryParam("alpha");
  const [modelOverride, setModelOverride] = useState<string | null>(null);

  const effAsset = asset ?? assets[0] ?? null;
  const effAlpha = alphaStr != null ? Number(alphaStr) : (alphas[0] ?? 0.025);

  const { data: allModels } = useModels();
  const { data: comparison } = useModelsComparison(effAsset, effAlpha);
  const bestModel = comparison?.find((r) => r.is_best)?.model ?? null;
  const effModel = modelOverride ?? bestModel;

  const { data: price } = usePrice(effAsset);
  const { data: prices } = usePrices(effAsset, 200);
  const { data: forecast } = useForecast(effAsset, effAlpha, effModel);
  const { data: backtests } = useBacktests(effAsset, effModel, effAlpha, 180, true);
  const { data: coverage } = useCoverage(effAsset, effAlpha);
  const { data: modelsLatest } = useModelsLatest(effAsset, effAlpha);
  const inMcsByModel = useMemo(
    () => Object.fromEntries((comparison ?? []).map((r) => [r.model, r.in_mcs])),
    [comparison],
  );
  const coverageRow = coverage?.find((r) => r.model === effModel) ?? null;

  const fcSource = forecast?.source;
  const fcAsof = forecast?.asof;
  const fcVarPrice = forecast?.var_price;
  const fcEsPrice = forecast?.es_price;
  const chartData = useMemo(() => {
    const priceLine: PricePoint[] = (prices ?? []).map((p) => ({
      time: p.date.slice(0, 10),
      value: p.close,
    }));
    const { varLine, esLine, breaches } = buildBand(prices ?? [], backtests ?? []);
    const { liveVar, liveEs } = liveBandPoints(
      { varLine, esLine },
      { source: fcSource ?? "", asof: fcAsof, var_price: fcVarPrice, es_price: fcEsPrice },
    );
    return { priceLine, varLine, esLine, breaches, liveVar, liveEs };
    // Primitives, not the `forecast` object: it carries a fetch timestamp that changes every
    // refresh, and a new chartData rebuilds the whole chart (and drops the user's zoom).
  }, [prices, backtests, fcSource, fcAsof, fcVarPrice, fcEsPrice]);

  const coneSeries = useMemo(() => buildConeSeries(forecast), [forecast]);

  if (!config || !effAsset) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-9 w-96" />
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <MetricCardSkeleton />
          <MetricCardSkeleton />
          <MetricCardSkeleton />
          <MetricCardSkeleton />
        </div>
        <ChartSkeleton height={420} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1
          className="bg-clip-text text-3xl font-semibold text-transparent"
          style={{ backgroundImage: "linear-gradient(90deg, var(--text), var(--amber))" }}
        >
          &#9672; Cryptorisk Terminal
        </h1>
        <p className="mt-1.5 max-w-3xl text-sm text-muted">
          Live spot price vs. the study&apos;s out-of-sample VaR/ES band. Backtest
          numbers are frozen at the last pipeline run; the band&apos;s model is
          re-fit on demand for today&apos;s forecast only.
        </p>
      </div>

      <MarketWatch assets={assets} alpha={effAlpha} selected={effAsset} onSelect={setAsset} />

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
            onChange={(e) => setAlphaStr(e.target.value)}
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

      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        {price ? (
          <>
            <MetricCard
              label={`${effAsset}-USD (Binance spot)`}
              value={fmtUsd(price.price, 2)}
              delta={`${price.change_pct >= 0 ? "+" : ""}${price.change_pct.toFixed(2)}% 24h`}
              deltaColor={price.change_pct >= 0 ? "text-green" : "text-red"}
              accent={price.change_pct >= 0 ? "var(--green)" : "var(--red)"}
            />
            <MetricCard
              label="24h High / Low"
              value={`${fmtUsd(price.high)} / ${fmtUsd(price.low)}`}
            />
          </>
        ) : (
          <>
            <MetricCardSkeleton />
            <MetricCardSkeleton />
          </>
        )}
        {forecast ? (
          <>
            {forecast.upper_price != null && forecast.var_price != null ? (
              <MetricCard
                label={`Implied ${fmtConfidence(effAlpha)} range (next close)`}
                value={`${fmtUsd(forecast.var_price)} – ${fmtUsd(forecast.upper_price)}`}
                accent="var(--violet)"
              />
            ) : (
              <MetricCard
                label={`VaR floor (${fmtConfidence(effAlpha)})`}
                value={fmtUsd(forecast.var_price)}
                title="Model has no upper-tail quantile (e.g. CAViaR)."
                accent="var(--amber)"
              />
            )}
            <MetricCard
              label={`ES floor (${fmtConfidence(effAlpha)})`}
              value={fmtUsd(forecast.es_price)}
              accent="var(--red)"
            />
            <RiskZoneCard
              label="Basel risk zone"
              zone={coverageRow?.basel_zone}
              caption={`${effModel ?? "model"} · 250d exceptions`}
            />
          </>
        ) : (
          <>
            <MetricCardSkeleton />
            <MetricCardSkeleton />
          </>
        )}
      </div>

      {forecast && forecast.var_price != null && (
        <p className="card border-l-2 border-l-amber px-4 py-3 text-sm">
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
          . The panels below extend this to 5/10/30 days (two specialized
          models) and compare what a normal vs. crisis day looks like.
        </p>
      )}

      <WhyThisVar asset={effAsset} alpha={effAlpha} model={effModel} />

      {modelsLatest && modelsLatest.length > 0 && (
        <ModelSpread
          rows={modelsLatest}
          alpha={effAlpha}
          inMcs={inMcsByModel}
          highlight={effModel}
          onSelect={setModelOverride}
        />
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

      {forecast?.intraday && forecast.var != null && (
        <div className="card flex flex-wrap items-center gap-x-6 gap-y-1 p-3 text-sm">
          <Badge kind={forecast.intraday.var_breached ? "frozen" : "live"}>
            {forecast.intraday.var_breached ? "VaR already hit today" : "Inside VaR today"}
          </Badge>
          <span className="text-muted">
            {fmtDate(forecast.intraday.date)} UTC so far ({Math.round(forecast.intraday.fraction_of_day * 100)}% of the
            day): {fmtPct(forecast.intraday.ret_so_far)} now, worst {fmtPct(forecast.intraday.low_ret)}
            {" "}vs VaR {fmtPct(forecast.var)}
            {forecast.es != null && <> / ES {fmtPct(forecast.es)}</>}
            {forecast.intraday.es_breached && " — ES level also crossed"}.
          </span>
        </div>
      )}

      {chartData.priceLine.length > 0 ? (
        <div className="card p-4">
          <PriceChart
            price={chartData.priceLine}
            varLine={chartData.varLine}
            esLine={chartData.esLine}
            breaches={chartData.breaches}
            livePrice={price?.price}
            liveVar={chartData.liveVar}
            liveEs={chartData.liveEs}
          />
        </div>
      ) : (
        <ChartSkeleton height={420} />
      )}
      <p className="text-xs text-muted">
        Amber/red lines are the {effModel ?? "selected model"} walk-forward
        VaR/ES for that day&apos;s close, plotted against the prior close.
        Markers = realized OOS violations (realized &lt; VaR). Days after
        the study&apos;s frozen sample end come from a daily incremental
        walk-forward (same model and window, not part of the evaluation
        tables); the final point is today&apos;s live re-fit.
      </p>

      {forecast === undefined ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <ChartSkeleton height={260} />
          <ChartSkeleton height={260} />
        </div>
      ) : (
        coneSeries.length > 0 &&
        forecast.last_close != null && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {coneSeries.map((s) => (
              <ConeChart key={s.key} series={[s]} lastClose={forecast.last_close as number} />
            ))}
          </div>
        )
      )}

      {forecast && forecast.regime_summary && (
        <RegimeDistribution
          normal={forecast.regime_summary.normal}
          crisis={forecast.regime_summary.crisis}
        />
      )}
    </div>
  );
}
