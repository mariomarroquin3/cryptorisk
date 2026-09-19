"use client";

import { Suspense, useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Column, DataTable } from "@/components/DataTable";
import { Field } from "@/components/Field";
import { MetricCard } from "@/components/MetricCard";
import { ChartSkeleton, MetricCardSkeleton } from "@/components/Skeleton";
import {
  CorrelationContrast,
  DurationHistogram,
  RegimeRow,
  RegimeTimeline,
  RegimeVolChart,
  TransitionMatrix,
} from "@/components/RegimeCharts";
import { PcrisisBandChart, RegimeStatsTable } from "@/components/RegimeStats";
import { PriceHistoryRow, RegimeCorrRow } from "@/lib/api";
import { useConfig, usePrices, useRegimes } from "@/lib/hooks";
import { useQueryParam } from "@/lib/useQueryParam";

function rollingAbsMean(prices: PriceHistoryRow[], window = 21): Map<string, number> {
  const map = new Map<string, number>();
  let sum = 0;
  for (let i = 0; i < prices.length; i++) {
    sum += Math.abs(prices[i].log_return);
    if (i >= window) sum -= Math.abs(prices[i - window].log_return);
    if (i >= window - 1) map.set(prices[i].date.slice(0, 10), sum / window);
  }
  return map;
}

/** Trailing realized volatility: sd-like sqrt(mean r^2) over `window` days. */
function rollingVol(prices: PriceHistoryRow[], window = 21): Map<string, number> {
  const map = new Map<string, number>();
  let sum = 0;
  for (let i = 0; i < prices.length; i++) {
    sum += prices[i].log_return ** 2;
    if (i >= window) sum -= prices[i - window].log_return ** 2;
    if (i >= window - 1) map.set(prices[i].date.slice(0, 10), Math.sqrt(sum / window));
  }
  return map;
}

export default function RegimesPage() {
  return (
    <Suspense>
      <RegimesPageInner />
    </Suspense>
  );
}

function RegimesPageInner() {
  const { data: config } = useConfig();
  const assets = config?.assets ?? [];
  const [asset, setAsset] = useQueryParam("asset");
  const effAsset = asset ?? assets[0] ?? null;

  const { data: regimes } = useRegimes(effAsset, 3000);
  const { data: prices } = usePrices(effAsset, 3000);

  const merged = useMemo(() => {
    if (!regimes?.series || !prices) return [];
    const absMean = rollingAbsMean(prices, 21);
    return regimes.series.map((s) => {
      const d = s.date.slice(0, 10);
      return {
        date: d,
        prob_crisis_insample: s.prob_crisis_insample,
        prob_crisis_pred: s.prob_crisis_pred,
        sigma2: s.sigma2,
        abs_ret_21d: absMean.get(d) ?? null,
      };
    });
  }, [regimes, prices]);

  const regimeRows = useMemo<RegimeRow[]>(() => {
    if (!regimes?.series || !prices) return [];
    const vol = rollingVol(prices, 21);
    return regimes.series.map((s) => {
      const d = s.date.slice(0, 10);
      return {
        date: d,
        insample: s.prob_crisis_insample,
        wf: s.prob_crisis_pred,
        sigma: s.sigma2 != null ? Math.sqrt(s.sigma2) : null,
        vol21: vol.get(d) ?? null,
      };
    });
  }, [regimes, prices]);

  const priceSeries = useMemo(
    () => (prices ?? []).map((p) => ({ date: p.date.slice(0, 10), close: p.close })),
    [prices],
  );

  const current = merged.length > 0 ? merged[merged.length - 1] : null;

  const corrCols: Column<RegimeCorrRow>[] = [
    { key: "series", label: "series" },
    { key: "kind", label: "kind" },
    {
      key: "corr_absret",
      label: "corr(|ret|)",
      help: "Correlation between this regime-probability series and same-day absolute return -- how well it tracks realized volatility.",
    },
    {
      key: "spearman_absret",
      label: "spearman(|ret|)",
      help: "Rank-correlation version of the same measure, more robust to outliers.",
    },
    {
      key: "corr_rv",
      label: "corr(RV)",
      help: "Correlation with same-day realized variance (from 5-minute bars) instead of the daily absolute return.",
    },
    {
      key: "spearman_rv21",
      label: "spearman(RV,21d)",
      help: "Rank correlation with a 21-day realized-variance average.",
    },
    {
      key: "mean_prob",
      label: "mean P(crisis)",
      help: "Average value of the crisis-probability series over the full sample.",
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Regimes</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          MS-GARCH 2-regime crisis probability (full-sample fit, in-sample) as
          a high-volatility-day detector — <strong>not</strong> a sustained
          walk-forward regime signal. The VaR/ES the decision layer uses is
          still walk-forward.
        </p>
      </div>

      <Field label="Asset">
        <select className="select" value={effAsset ?? ""} onChange={(e) => setAsset(e.target.value)}>
          {assets.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </select>
      </Field>

      {regimes === undefined || prices === undefined ? (
        <>
          <ChartSkeleton height={220} />
          <ChartSkeleton height={220} />
        </>
      ) : (
        <>
          <CorrelationContrast rows={regimes.correlations} />
          <RegimeTimeline rows={regimeRows} />
          <div className="grid gap-6 md:grid-cols-2">
            <TransitionMatrix summary={regimes.regime_summary} />
            <DurationHistogram rows={regimeRows} summary={regimes.regime_summary} />
          </div>
          <RegimeVolChart rows={regimeRows} />
          <RegimeStatsTable rows={regimes.stats ?? []} />
          <PcrisisBandChart rows={regimes.band ?? []} />
        </>
      )}

      {regimes === undefined || prices === undefined ? (
        <ChartSkeleton height={260} />
      ) : (
        priceSeries.length > 0 && (
          <div className="card p-4">
            <div className="mb-2 text-sm text-muted">{effAsset} price</div>
            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart data={priceSeries}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: "var(--muted)", fontSize: 10 }} minTickGap={40} />
                <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} width={70} domain={["auto", "auto"]} />
                <Tooltip contentStyle={{ background: "var(--bg)", border: "1px solid var(--grid)" }} />
                <Line type="monotone" dataKey="close" stroke="var(--text)" dot={false} strokeWidth={1.5} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )
      )}

      {regimes === undefined || prices === undefined ? (
        <ChartSkeleton height={300} />
      ) : (
        merged.length > 0 && (
          <div className="card p-4">
            <div className="mb-2 text-sm text-muted">
              In-sample crisis probability vs. realized vol proxy
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={merged}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis dataKey="date" tick={{ fill: "var(--muted)", fontSize: 10 }} minTickGap={40} />
                <YAxis
                  yAxisId="left"
                  domain={[0, 1]}
                  tick={{ fill: "var(--muted)", fontSize: 11 }}
                  width={40}
                />
                <YAxis yAxisId="right" orientation="right" tick={{ fill: "var(--muted)", fontSize: 11 }} width={50} />
                <Tooltip contentStyle={{ background: "var(--bg)", border: "1px solid var(--grid)" }} />
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="prob_crisis_insample"
                  stroke="var(--red)"
                  fill="var(--red)"
                  fillOpacity={0.25}
                  name="P(crisis) in-sample"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="abs_ret_21d"
                  stroke="var(--amber)"
                  strokeDasharray="4 3"
                  dot={false}
                  name="|return| 21d avg"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )
      )}

      {regimes === undefined ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <MetricCardSkeleton />
          <MetricCardSkeleton />
          <MetricCardSkeleton />
        </div>
      ) : (
        current && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <MetricCard
              label="Current P(crisis), in-sample"
              value={`${(current.prob_crisis_insample * 100).toFixed(0)}%`}
            />
            <MetricCard
              label="Current P(crisis), walk-forward"
              value={`${(current.prob_crisis_pred * 100).toFixed(0)}%`}
              title="Near-zero OOS correlation with vol -- shown for completeness only."
            />
            <MetricCard
              label="MS-GARCH sigma (next-day)"
              value={current.sigma2 != null ? Math.sqrt(current.sigma2).toFixed(4) : "n/a"}
            />
          </div>
        )
      )}

      <div>
        <h2 className="mb-2 text-lg font-medium">Regime-identification correlations</h2>
        <DataTable
          columns={corrCols}
          rows={regimes?.correlations ?? []}
          keyField={(row) => `${row.series}-${row.kind}`}
          loading={regimes === undefined}
        />
        <p className="mt-2 text-xs text-muted">
          &quot;insample&quot; tracks vol (~0.5-0.75); &quot;filt_wf&quot;/&quot;pred_wf&quot; (walk-forward)
          barely do (~0.0-0.1) — the MS-GARCH doesn&apos;t identify in 500-day
          rolling windows.
        </p>
      </div>
    </div>
  );
}
