"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { scaledStudentTDensity } from "@/lib/studentT";
import { fmtPct } from "@/lib/format";

export interface RegimeInfo {
  vol: number;
  nu: number;
  p_stay: number | null;
}

/** What a "normal" day vs. a "crisis" day looks like, statistically -- two
 * Student-t densities (mean 0, each regime's own fitted std and tail
 * shape), not a time-indexed forecast. Deliberately a separate, single-
 * purpose chart instead of folding MS-GARCH into the forward cone's time
 * axis, since the walk-forward regime signal can't honestly support a
 * "regime at day H" prediction (see api/cone.py's docstring). */
export function RegimeDistribution({
  normal,
  crisis,
}: {
  normal: RegimeInfo;
  crisis: RegimeInfo;
}) {
  const span = Math.max(normal.vol, crisis.vol) * 4;
  const n = 121;
  const data = Array.from({ length: n }, (_, i) => {
    const r = -span + (2 * span * i) / (n - 1);
    return {
      r,
      normal: scaledStudentTDensity(r, normal.vol, normal.nu),
      crisis: scaledStudentTDensity(r, crisis.vol, crisis.nu),
    };
  });

  return (
    <div className="card p-4">
      <div className="mb-2 text-sm text-muted">
        Normal vs. crisis regime -- what a day&apos;s return looks like statistically
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis
            dataKey="r"
            tickFormatter={(v) => fmtPct(v, 0)}
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            axisLine={{ stroke: "var(--grid)" }}
          />
          <YAxis tick={false} axisLine={false} width={4} />
          <Tooltip
            contentStyle={{ background: "var(--bg)", border: "1px solid var(--grid)", fontSize: 12 }}
            labelFormatter={(v) => `return ${fmtPct(Number(v), 1)}`}
            formatter={(value) => Number(value).toFixed(3)}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Area
            dataKey="normal"
            name={`Normal (σ=${fmtPct(normal.vol, 2)})`}
            stroke="var(--green)"
            fill="var(--green)"
            fillOpacity={0.15}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
          <Area
            dataKey="crisis"
            name={`Crisis (σ=${fmtPct(crisis.vol, 2)})`}
            stroke="var(--red)"
            fill="var(--red)"
            fillOpacity={0.15}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-muted">
        Two Student-t densities (mean 0, each regime&apos;s own fitted daily
        std and tail shape from the full-sample MS-GARCH fit) -- not a
        forecast of which regime holds at any future date, just what a
        typical day in each regime looks like. Regime persistence (once
        entered, P(stay)): normal {fmtPct(normal.p_stay, 0)}, crisis{" "}
        {fmtPct(crisis.p_stay, 0)}.
      </p>
    </div>
  );
}
