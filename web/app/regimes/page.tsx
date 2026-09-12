"use client";

import { useMemo, useState } from "react";
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
import { MetricCard } from "@/components/MetricCard";
import { PriceHistoryRow, RegimeCorrRow } from "@/lib/api";
import { useConfig, usePrices, useRegimes } from "@/lib/hooks";

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

export default function RegimesPage() {
  const { data: config } = useConfig();
  const assets = config?.assets ?? [];
  const [asset, setAsset] = useState<string | null>(null);
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

  const priceSeries = useMemo(
    () => (prices ?? []).map((p) => ({ date: p.date.slice(0, 10), close: p.close })),
    [prices],
  );

  const current = merged.length > 0 ? merged[merged.length - 1] : null;

  const corrCols: Column<RegimeCorrRow>[] = [
    { key: "series", label: "series" },
    { key: "kind", label: "kind" },
    { key: "corr_absret", label: "corr(|ret|)" },
    { key: "spearman_absret", label: "spearman(|ret|)" },
    { key: "corr_rv", label: "corr(RV)" },
    { key: "spearman_rv21", label: "spearman(RV,21d)" },
    { key: "mean_prob", label: "mean P(crisis)" },
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

      {priceSeries.length > 0 && (
        <div className="rounded border border-grid bg-panel p-3">
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
      )}

      {merged.length > 0 && (
        <div className="rounded border border-grid bg-panel p-3">
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
      )}

      {current && (
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
      )}

      <div>
        <h2 className="mb-2 text-lg font-medium">Regime-identification correlations</h2>
        <DataTable
          columns={corrCols}
          rows={regimes?.correlations ?? []}
          keyField={(row) => `${row.series}-${row.kind}`}
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

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-muted">
      {label}
      {children}
    </label>
  );
}
