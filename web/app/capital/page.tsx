"use client";

import { Suspense } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Column, DataTable } from "@/components/DataTable";
import { Field } from "@/components/Field";
import { MetricCard } from "@/components/MetricCard";
import { ChartSkeleton, MetricCardSkeleton } from "@/components/Skeleton";
import { ZoneBadge } from "@/components/ZoneBadge";
import { fmtPct, fmtUsd } from "@/lib/format";
import { CapitalRow, LimitsRow } from "@/lib/api";
import { useCapital, useConfig, useEstimationRisk, useHedge, useLimits } from "@/lib/hooks";
import { useQueryParam } from "@/lib/useQueryParam";

export default function CapitalPage() {
  return (
    <Suspense>
      <CapitalPageInner />
    </Suspense>
  );
}

function CapitalPageInner() {
  const { data: config } = useConfig();
  const assets = config?.assets ?? [];
  const [asset, setAsset] = useQueryParam("asset");
  const effAsset = asset ?? assets[0] ?? null;

  const { data: capital } = useCapital(effAsset);
  const { data: estRisk } = useEstimationRisk(effAsset);
  const { data: limits } = useLimits(effAsset);
  const { data: hedge } = useHedge(effAsset);

  const inMcs = capital?.filter((r) => r.in_mcs) ?? [];
  const row =
    (inMcs.length > 0
      ? [...inMcs].sort((a, b) => a.capital_usd - b.capital_usd)[0]
      : capital?.[0]) ?? null;

  const pointCapital = row?.capital_usd ?? 0;
  const modelAddon = row?.model_risk_addon_usd ?? 0;
  const estAddonMedian =
    estRisk && estRisk.length > 0
      ? median(estRisk.map((r) => r.estimation_risk_addon_usd))
      : 0;
  const totalCapital = pointCapital + modelAddon;

  const stackData = [
    { name: "Point ES capital", value: pointCapital, fill: "#4dabf7" },
    { name: "+ model-risk add-on", value: modelAddon, fill: "#ffb020" },
    { name: "+ estimation-risk add-on (median)", value: estAddonMedian, fill: "#ff4d4f" },
  ];

  const capitalCols: Column<CapitalRow>[] = [
    { key: "model", label: "model" },
    {
      key: "in_mcs",
      label: "in MCS",
      help: "In the 90% Model Confidence Set: statistically indistinguishable from the best model.",
    },
    { key: "es_975_1d", label: "ES 97.5% (1d)", help: "1-day 97.5% Expected Shortfall from the frozen backtest -- this model's own point estimate." },
    {
      key: "es_10d_sqrt",
      label: "ES (10d, sqrt-t)",
      help: "The 1-day ES scaled to a 10-day liquidity horizon by square-root-of-time (the standard regulatory convention).",
    },
    {
      key: "es_10d_bootstrap",
      label: "ES (10d, bootstrap)",
      help: "10-day ES from a stationary block bootstrap of compounded returns -- an empirical check against the sqrt-time scaling.",
    },
    {
      key: "exceptions_250d",
      label: "exceptions",
      help: "Number of VaR violations in the last 250 trading days -- drives the Basel traffic-light zone.",
    },
    {
      key: "basel_zone",
      label: "zone",
      help: "Traffic-light zone from the 250-day exception count: green (<=4), amber (5-9), red (>=10).",
      render: (r) => <ZoneBadge zone={r.basel_zone} />,
    },
    {
      key: "m_c",
      label: "Basel m_c",
      help: "Capital multiplier: 1.5 base plus a traffic-light add-on (0 in the green zone, up to 1.0 in the red zone).",
    },
    {
      key: "capital_usd",
      label: "capital $",
      render: (r) => fmtUsd(r.capital_usd),
      help: "Point ES capital = m_c x notional x |ES, 10d|.",
    },
    {
      key: "model_risk_addon_usd",
      label: "model-risk add-on $",
      render: (r) => fmtUsd(r.model_risk_addon_usd),
      help: "Extra capital for not knowing which in-MCS model is right -- the spread between the highest- and lowest-capital in-MCS models.",
    },
  ];

  const limitsCols: Column<LimitsRow>[] = [
    { key: "model", label: "model" },
    {
      key: "in_mcs",
      label: "in MCS",
      help: "In the 90% Model Confidence Set: statistically indistinguishable from the best model.",
    },
    {
      key: "n_star_usd",
      label: "limit $",
      render: (r) => fmtUsd(r.n_star_usd),
      help: "Position size limit implied by the risk budget and this model's ES.",
    },
    {
      key: "bind_rate",
      label: "bind rate",
      help: "Fraction of days the position limit would have actually constrained trading (the limit binds).",
    },
    {
      key: "mean_utilisation",
      label: "avg util.",
      help: "Average utilization of the risk budget under this model's limit.",
    },
    {
      key: "budget_breach_rate",
      label: "budget breach rate",
      help: "Fraction of days the realized loss would have exceeded the risk budget despite the limit.",
    },
    {
      key: "worst_loss_usd",
      label: "worst loss $",
      render: (r) => fmtUsd(r.worst_loss_usd),
      help: "Worst single-day realized loss under this model's limit, in dollars.",
    },
  ];

  const h = hedge?.[0];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Capital &amp; Decision</h1>
        <p className="mt-1 text-sm text-muted">
          FRTB ES-IMA capital, position limits, estimation-risk add-on, perp
          hedge.
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

      {capital === undefined ? (
        <>
          <ChartSkeleton height={320} />
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <MetricCardSkeleton />
            <MetricCardSkeleton />
            <MetricCardSkeleton />
            <MetricCardSkeleton />
          </div>
        </>
      ) : (
        row && (
        <>
          <div className="card p-4">
            <div className="mb-2 flex items-center justify-between text-sm text-muted">
              <span>
                {effAsset} capital stack — {row.model}
              </span>
              <ZoneBadge zone={row.basel_zone} />
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={stackData} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "var(--muted)", fontSize: 11 }} />
                <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} width={80} />
                <Tooltip
                  contentStyle={{ background: "var(--bg)", border: "1px solid var(--grid)" }}
                  formatter={(v) => fmtUsd(Number(v))}
                />
                <Bar dataKey="value" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <MetricCard label="Model" value={row.model} />
            <MetricCard
              label="Point + model-risk capital"
              value={fmtUsd(totalCapital)}
              title="This model's own point ES capital plus the MCS spread add-on."
            />
            <MetricCard
              label="Model-risk add-on"
              value={fmtUsd(modelAddon)}
              title="Capital spread (max - min) across the in-MCS models for this asset."
            />
            <MetricCard
              label="Basel exceptions (250d)"
              value={`${row.exceptions_250d} (${row.m_c}x)`}
            />
          </div>

          <DataTable
            columns={capitalCols}
            rows={[...(capital ?? [])].sort((a, b) => a.capital_usd - b.capital_usd)}
            keyField="model"
          />
        </>
        )
      )}

      <div>
        <h2 className="mb-2 text-lg font-medium">Estimation-risk band (parameter-uncertainty bootstrap)</h2>
        {estRisk === undefined ? (
          <ChartSkeleton height={300} />
        ) : estRisk.length > 0 ? (
          <div className="card p-4">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={estRisk} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis dataKey="estimator" tick={{ fill: "var(--muted)", fontSize: 11 }} />
                <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} width={80} />
                <Tooltip
                  contentStyle={{ background: "var(--bg)", border: "1px solid var(--grid)" }}
                  formatter={(v) => fmtUsd(Number(v))}
                />
                <Bar dataKey="capital_point_usd" stackId="a" name="point capital" fill="#4dabf7" />
                <Bar
                  dataKey="estimation_risk_addon_usd"
                  stackId="a"
                  name="estimation-risk add-on"
                  fill="#ff4d4f"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-sm text-muted">No estimation-risk output for this asset.</div>
        )}
        <p className="mt-2 text-xs text-muted">
          Prudent ES = 5th-percentile of the bootstrap draws (HS: block
          bootstrap; GARCH-t / FHS: parameter draw from the fitted covariance,
          no refit). Add-on is second-order next to the model-risk add-on
          above.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <h2 className="mb-2 text-lg font-medium">Position limits</h2>
          <DataTable
            columns={limitsCols}
            rows={limits ?? []}
            keyField="model"
            loading={limits === undefined}
          />
        </div>
        <div>
          <h2 className="mb-2 text-lg font-medium">Perp hedge</h2>
          {hedge === undefined ? (
            <div className="space-y-3">
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </div>
          ) : h ? (
            <div className="space-y-3">
              <MetricCard label="Min-variance hedge ratio" value={h.ratio_min_var.toFixed(3)} />
              <MetricCard
                label="Funding carry (annualised)"
                value={fmtPct(h.funding_carry_annual_frac)}
                delta={fmtUsd(h.funding_carry_annual_usd) + "/yr"}
              />
              {h.es_reduction != null ? (
                <MetricCard label="ES reduction from hedge" value={fmtPct(h.es_reduction)} />
              ) : (
                <div className="text-xs text-muted">Note: {h.note ?? "n/a"}</div>
              )}
            </div>
          ) : (
            <div className="text-sm text-muted">No hedge output for this asset.</div>
          )}
        </div>
      </div>
    </div>
  );
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
