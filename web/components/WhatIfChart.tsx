"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, TIP } from "@/components/ExplainCharts";
import { ModelInfo, WhatIfRow } from "@/lib/api";

export interface WhatIfPoint {
  model: string;
  base: number;
  shocked: number;
  multiple: number;
}

export function toPoints(rows: Record<string, WhatIfRow | null | undefined>): WhatIfPoint[] {
  const out: WhatIfPoint[] = [];
  for (const [model, r] of Object.entries(rows)) {
    if (!r || !Number.isFinite(r.baseline.var) || !Number.isFinite(r.shocked.var) || r.baseline.var === 0) continue;
    out.push({
      model,
      base: r.baseline.var,
      shocked: r.shocked.var,
      multiple: r.shocked.var / r.baseline.var,
    });
  }
  return out;
}

/** How much each model's VaR moves: shocked VaR as a multiple of today's. 1 =
 * no reaction; a slow or smoothing model sits near 1, a reactive one well above. */
export function ReactionBars({
  points,
  info,
}: {
  points: WhatIfPoint[];
  info: Record<string, ModelInfo> | undefined;
}) {
  const data = [...points].sort((a, b) => b.multiple - a.multiple);
  const isMl = (m: string) => (info?.[m]?.family ?? "").toLowerCase().includes("machine");
  return (
    <ResponsiveContainer width="100%" height={340}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 60 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="model" angle={-40} textAnchor="end" interval={0} tick={AXIS} height={80} />
        <YAxis tick={AXIS} width={45} tickFormatter={(v) => `${Number(v).toFixed(1)}x`} />
        <Tooltip {...TIP} formatter={(v) => `${Number(v).toFixed(2)}x today's VaR`} />
        <ReferenceLine y={1} stroke="var(--green)" strokeDasharray="4 3" />
        <Bar dataKey="multiple" name="VaR multiple" radius={[3, 3, 0, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.model} fill={isMl(d.model) ? "var(--amber)" : "#4dabf7"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
