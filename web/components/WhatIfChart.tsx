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
  flat: number;
  shocked: number;
  /** shocked VaR / flat-day VaR: the effect of the shock itself. */
  effect: number;
}

export function toPoints(rows: Record<string, WhatIfRow | null | undefined>): WhatIfPoint[] {
  const out: WhatIfPoint[] = [];
  for (const [model, r] of Object.entries(rows)) {
    if (!r || !Number.isFinite(r.flat.var) || !Number.isFinite(r.shocked.var) || r.flat.var === 0) continue;
    out.push({
      model,
      base: r.baseline.var,
      flat: r.flat.var,
      shocked: r.shocked.var,
      effect: r.shocked.var / r.flat.var,
    });
  }
  return out;
}

/** How much the shock itself moves each model's VaR: shocked VaR as a multiple of the
 * VaR after a flat day. 1 = no reaction; a slow or smoothing model sits near 1, a
 * reactive one well above. */
export function ReactionBars({
  points,
  info,
}: {
  points: WhatIfPoint[];
  info: Record<string, ModelInfo> | undefined;
}) {
  const data = [...points].sort((a, b) => b.effect - a.effect);
  const isMl = (m: string) => (info?.[m]?.family ?? "").toLowerCase().includes("machine");
  return (
    <ResponsiveContainer width="100%" height={340}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 60 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="model" angle={-40} textAnchor="end" interval={0} tick={AXIS} height={80} />
        <YAxis tick={AXIS} width={45} tickFormatter={(v) => `${Number(v).toFixed(1)}x`} />
        <Tooltip {...TIP} formatter={(v) => `${Number(v).toFixed(2)}x the flat-day VaR`} />
        <ReferenceLine y={1} stroke="var(--green)" strokeDasharray="4 3" />
        <Bar dataKey="effect" name="shock effect" radius={[3, 3, 0, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.model} fill={isMl(d.model) ? "var(--amber)" : "#4dabf7"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
