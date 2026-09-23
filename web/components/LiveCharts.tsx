"use client";

import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, pct, TIP } from "@/components/ExplainCharts";
import { LiveDay, LiveModelRow, ModelInfo } from "@/lib/api";

/** Realized return each day against the range of every model's VaR: a
 * breach is the line dropping below the whole band's lower edge. */
export function RealizedVsBand({ days }: { days: LiveDay[] }) {
  const data = days.map((d) => ({ ...d, band: [d.var_min, d.var_max] as [number, number] }));
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="date" tick={AXIS} tickFormatter={(v: string) => v.slice(5)} />
        <YAxis tick={AXIS} width={50} tickFormatter={(v) => pct(v, 0)} />
        <Tooltip
          {...TIP}
          formatter={(v) =>
            Array.isArray(v) ? `${pct(Number(v[0]), 2)} to ${pct(Number(v[1]), 2)}` : pct(Number(v), 2)
          }
        />
        <ReferenceLine y={0} stroke="var(--grid)" />
        <Area
          dataKey="band"
          name="model VaR range"
          stroke="none"
          fill="var(--amber)"
          fillOpacity={0.25}
          isAnimationActive={false}
        />
        <Line dataKey="es_median" name="median ES" stroke="var(--red)" strokeDasharray="4 3" dot={false} isAnimationActive={false} />
        <Line dataKey="realized" name="realized return" stroke="var(--text)" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Mean FZ0 per model over the live days (lower is better), machine learning
 * highlighted. */
export function LiveFz0Bars({
  rows,
  info,
}: {
  rows: LiveModelRow[];
  info: Record<string, ModelInfo> | undefined;
}) {
  const data = [...rows].sort((a, b) => a.mean_fz0 - b.mean_fz0);
  const isMl = (m: string) => (info?.[m]?.family ?? "").toLowerCase().includes("machine");
  const lo = Math.min(...data.map((d) => d.mean_fz0));
  return (
    <ResponsiveContainer width="100%" height={340}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 60 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="model" angle={-40} textAnchor="end" interval={0} tick={AXIS} height={80} />
        <YAxis
          tick={AXIS}
          width={50}
          domain={[Math.floor(lo * 10) / 10, "auto"]}
          tickFormatter={(v) => Number(v).toFixed(1)}
        />
        <Tooltip {...TIP} formatter={(v) => Number(v).toFixed(3)} />
        <Bar dataKey="mean_fz0" name="mean FZ0" radius={[3, 3, 0, 0]}>
          {data.map((d) => (
            <Cell key={d.model} fill={isMl(d.model) ? "var(--amber)" : "#4dabf7"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
