"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Column, DataTable } from "@/components/DataTable";
import { AXIS, ChartCard, pct, TIP } from "@/components/ExplainCharts";
import { ModelTip } from "@/components/ModelTip";
import { ChartSkeleton } from "@/components/Skeleton";
import { ZoneBadge } from "@/components/ZoneBadge";
import { BaselHeadroomRow, BreachDistance } from "@/lib/api";
import { fmtUsd } from "@/lib/format";
import { useBaselHeadroom, useBreachDistanceAll, useModelsInfo } from "@/lib/hooks";

type Row = BaselHeadroomRow & { dist: number | null; breachPrice: number | null };

const ZONE_FILL: Record<string, string> = { green: "var(--green)", amber: "var(--amber)", red: "var(--red)" };

/** How close each model is to a 99% VaR breach and to the next Basel zone. */
export function BreachHeadroom({ asset }: { asset: string | null }) {
  const { data: head } = useBaselHeadroom(asset);
  const { data: info } = useModelsInfo();
  const models = useMemo(() => (head ?? []).filter((r) => r.model !== "MS-GARCH").map((r) => r.model), [head]);
  const dist = useBreachDistanceAll(asset, models);

  const rows: Row[] = useMemo(
    () =>
      (head ?? []).map((r) => {
        const d = dist[r.model] as BreachDistance | null | undefined;
        return { ...r, dist: d?.dist_var ?? null, breachPrice: d?.var_price ?? null };
      }),
    [head, dist],
  );
  const pending = models.filter((m) => dist[m] === undefined).length;
  const newest = rows.reduce((m, r) => (r.as_of > m ? r.as_of : m), "");
  const named = (r: Row) => (r.as_of < newest ? `${r.model} (data to ${r.as_of})` : r.model);
  const nearAmber = rows.filter((r) => r.zone === "green" && r.to_next_zone === 1);
  const ageing = rows.filter((r) => r.zone !== "green" && r.ageing_out_30d > 0);

  const cols: Column<Row>[] = [
    { key: "model", label: "model", render: (r) => <ModelTip name={r.model} info={info?.[r.model]} /> },
    { key: "exceptions_250d", label: "exceptions (250d)", help: "Days in the last 250 where the 99% VaR was breached. Basel: green 0-4, amber 5-9, red 10 or more." },
    { key: "zone", label: "zone", render: (r) => <ZoneBadge zone={r.zone} /> },
    { key: "to_next_zone", label: "breaches to next zone", render: (r) => (r.to_next_zone == null ? "-" : r.to_next_zone), help: "How many more exceptions until the model changes zone (green to amber at 5, amber to red at 10)." },
    { key: "ageing_out_30d", label: "ageing out (30d)", help: "Exceptions in the oldest 30 days of the window. They leave the 250-day count within a month, so the zone can improve with no new good days." },
    { key: "m_c", label: "multiplier", render: (r) => `${r.m_c.toFixed(2)} → ${r.m_c_if_breached.toFixed(2)}`, help: "Basel multiplier now, and after one more exception." },
    { key: "extra_capital_if_breached_usd", label: "extra capital if breached", render: (r) => fmtUsd(r.extra_capital_if_breached_usd), help: "Capital added by one more exception: the multiplier step times the 1M notional times the 10-day 97.5% ES." },
    { key: "breachPrice", label: "99% VaR price", render: (r) => (r.breachPrice == null ? "…" : fmtUsd(r.breachPrice, 0)), help: "The price below which tomorrow's close would breach this model's 99% VaR (a live re-fit)." },
    { key: "as_of", label: "data to", help: "Last day of this model's walk-forward. Older than the rest means it is not rolled forward daily (MS-GARCH runs offline in R)." },
    { key: "dist", label: "distance to breach", render: (r) => (r.dist == null ? "…" : pct(r.dist, 1)), help: "How far the live price is above that breach price, as a % of the live price. Smaller = closer." },
  ];

  if (head === undefined) return <ChartSkeleton height={300} />;
  if (head.length === 0) return <div className="text-sm text-muted">No exception history for this asset.</div>;

  const bars = rows
    .filter((r) => r.dist != null)
    .map((r) => ({ model: r.model, dist: -(r.dist as number) * 100, zone: r.zone }))
    .sort((a, b) => a.dist - b.dist);

  return (
    <div className="space-y-4">
      <div className="card p-4 text-sm">
        {nearAmber.length > 0 && (
          <p>
            <span className="font-semibold">{nearAmber.map(named).join(", ")}</span>{" "}
            {nearAmber.length === 1 ? "is" : "are"} one exception from the amber zone.{" "}
          </p>
        )}
        {ageing.length > 0 && (
          <p className="text-muted">
            {ageing.length} amber/red model{ageing.length === 1 ? "" : "s"} will shed exceptions within 30 days as old
            ones leave the window (up to {Math.max(...ageing.map((r) => r.ageing_out_30d))}), even with no new good day.
          </p>
        )}
        {nearAmber.length === 0 && ageing.length === 0 && <p className="text-muted">No model is one exception from a zone change.</p>}
        <p className="mt-1 text-xs text-muted">
          {pending > 0 ? `Re-fitting ${pending} more models for the breach prices...` : "Breach prices are live re-fits."}
        </p>
      </div>
      {bars.length > 0 && (
        <ChartCard
          title="How far the price is from breaching each model's 99% VaR"
          caption="Distance from the live price down to the price that would breach the model's 99% VaR for tomorrow, coloured by Basel zone. A shorter bar means a model that would be breached by a smaller fall."
        >
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={bars} margin={{ top: 10, right: 10, left: 0, bottom: 60 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="model" angle={-40} textAnchor="end" interval={0} tick={AXIS} height={80} />
              <YAxis tick={AXIS} width={50} tickFormatter={(v) => `${Number(v).toFixed(0)}%`} />
              <Tooltip {...TIP} formatter={(v) => `${Number(v).toFixed(2)}% below the live price`} />
              <Bar dataKey="dist" name="distance to breach" isAnimationActive={false}>
                {bars.map((b) => (
                  <Cell key={b.model} fill={ZONE_FILL[b.zone] ?? "var(--muted)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      )}
      <DataTable columns={cols} rows={rows} keyField="model" filterKey="model" />
    </div>
  );
}
