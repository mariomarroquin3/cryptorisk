"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Column, DataTable } from "@/components/DataTable";
import { AXIS, ChartCard, TIP } from "@/components/ExplainCharts";
import { ModelTip } from "@/components/ModelTip";
import { ChartSkeleton } from "@/components/Skeleton";
import { fmtPct } from "@/lib/format";
import { ModelInfo, VarChangeRow } from "@/lib/api";
import { useModels, useModelsInfo, useVarChangeAll } from "@/lib/hooks";

interface Point {
  model: string;
  prev: number;
  now: number;
  /** Change in VaR magnitude, in percentage points: positive = the model sees more risk than yesterday. */
  total: number;
  fromNew: number;
  fromOld: number;
}

/** Magnitudes, not signed levels: VaR is negative, so "up" would otherwise read as less risk. */
function toPoints(rows: Record<string, VarChangeRow | null | undefined>): Point[] {
  const out: Point[] = [];
  for (const [model, r] of Object.entries(rows)) {
    if (!r || ![r.var.prev, r.var.now, r.var.new, r.var.old].every(Number.isFinite)) continue;
    out.push({
      model,
      prev: r.var.prev,
      now: r.var.now,
      total: -(r.var.now - r.var.prev) * 100,
      fromNew: -r.var.new * 100,
      fromOld: -r.var.old * 100,
    });
  }
  return out;
}

function ChangeBars({ points }: { points: Point[] }) {
  const data = [...points].sort((a, b) => Math.abs(b.total) - Math.abs(a.total));
  return (
    <ResponsiveContainer width="100%" height={340}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 60 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="model" angle={-40} textAnchor="end" interval={0} tick={AXIS} height={80} />
        <YAxis tick={AXIS} width={50} tickFormatter={(v) => `${Number(v).toFixed(1)}`} />
        <Tooltip {...TIP} formatter={(v) => `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(2)} pp`} />
        <Legend />
        <ReferenceLine y={0} stroke="var(--muted)" />
        <Bar dataKey="fromNew" name="new day entered" stackId="a" fill="#4dabf7" isAnimationActive={false} />
        <Bar dataKey="fromOld" name="oldest day left" stackId="a" fill="var(--amber)" isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Attribution of each model's day-over-day VaR change: the new observation vs the one that
 * fell out of the 500-day window. */
export function VarChangePanel({ asset, alpha }: { asset: string | null; alpha: number }) {
  const { data: allModels } = useModels();
  const { data: info } = useModelsInfo();
  const models = useMemo(() => (allModels ?? []).filter((m) => m !== "MS-GARCH"), [allModels]);
  const rows = useVarChangeAll(asset, models, alpha);
  const points = useMemo(() => toPoints(rows), [rows]);
  const ctx = Object.values(rows).find((r) => r) ?? null;
  const pending = models.filter((m) => rows[m] === undefined).length;
  const failed = models.filter((m) => rows[m] === null);

  const cols: Column<Point>[] = [
    { key: "model", label: "model", render: (r) => <ModelTip name={r.model} info={info?.[r.model] as ModelInfo | undefined} /> },
    { key: "prev", label: "VaR yesterday", render: (r) => fmtPct(r.prev, 2), help: "Forecast made from the window ending the day before." },
    { key: "now", label: "VaR today", render: (r) => fmtPct(r.now, 2), help: "Forecast made from today's window." },
    { key: "total", label: "change (pp)", render: (r) => `${r.total >= 0 ? "+" : ""}${r.total.toFixed(2)}`, help: "Change in VaR magnitude. Positive = the model now sees more risk." },
    { key: "fromNew", label: "from new day (pp)", render: (r) => `${r.fromNew >= 0 ? "+" : ""}${r.fromNew.toFixed(2)}`, help: "Part of the change caused by the latest day entering the window." },
    { key: "fromOld", label: "from oldest day (pp)", render: (r) => `${r.fromOld >= 0 ? "+" : ""}${r.fromOld.toFixed(2)}`, help: "Part of the change caused by the oldest day leaving the window." },
  ];

  if (points.length === 0) return <ChartSkeleton height={340} />;
  return (
    <div className="space-y-4">
      {ctx && (
        <div className="card p-4 text-sm">
          Between the forecast for {ctx.prev_asof} and the one for {ctx.asof}, the window gained{" "}
          <span className="font-semibold">{ctx.new_date}</span> ({fmtPct(Math.expm1(ctx.new_return), 2)}) and dropped{" "}
          <span className="font-semibold">{ctx.dropped_date}</span> ({fmtPct(Math.expm1(ctx.dropped_return), 2)}).{" "}
          <span className="text-muted">
            {pending > 0 ? `Re-fitting ${pending} more models...` : `${points.length} models re-fitted`}
            {failed.length > 0 && `; no live re-fit on this server for ${failed.join(", ")}`}.
          </span>
        </div>
      )}
      <ChartCard
        title="What moved each model's VaR since yesterday"
        caption="Change in VaR magnitude (percentage points; positive = more risk), split into the effect of the new day entering the 500-day window and of the oldest day leaving it. The two effects interact, so each is averaged over both orders and they add up exactly to the total change. A small dropped-day bar just means the day that left was unremarkable; it matters only when a large past shock ages out."
      >
        <ChangeBars points={points} />
      </ChartCard>
      <DataTable columns={cols} rows={points} keyField="model" filterKey="model" />
    </div>
  );
}
