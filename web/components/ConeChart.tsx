"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ConePoint } from "@/lib/api";
import { fmtUsd } from "@/lib/format";

export interface ConeSeries {
  key: string;
  label: string;
  color: string;
  dash?: string;
  points: ConePoint[];
}

/** Three specialized models' forward VaR/upper bounds, each its own
 * line-pair converging on today's price -- deliberately lines, not filled
 * areas, since three overlapping fills would blend into an unreadable
 * color mush. Showing them side by side instead of picking one "the" cone
 * makes model disagreement itself visible -- the same spirit as the
 * study's own model-risk add-on (decision/capital.py). */
export function ConeChart({ series, lastClose }: { series: ConeSeries[]; lastClose: number }) {
  const withData = series.filter((s) => s.points.length > 0);
  const days = Array.from(
    new Set(withData.flatMap((s) => s.points.map((p) => p.days))),
  ).sort((a, b) => a - b);

  const data = [0, ...days].map((d) => {
    const row: Record<string, number | string | null> = { days: d, label: d === 0 ? "today" : `+${d}d` };
    for (const s of withData) {
      const point = s.points.find((p) => p.days === d);
      row[`${s.key}_var`] = d === 0 ? lastClose : (point?.var_price ?? null);
      row[`${s.key}_upper`] = d === 0 ? lastClose : (point?.upper_price ?? null);
    }
    return row;
  });

  const allPrices = withData.flatMap((s) => s.points.flatMap((p) => [p.var_price, p.upper_price]));
  const finite = [lastClose, ...allPrices.filter((v): v is number => v != null && Number.isFinite(v))];
  const yDomain: [number, number] = [Math.min(...finite) * 0.96, Math.max(...finite) * 1.04];

  return (
    <div className="rounded border border-grid bg-panel p-3">
      <div className="mb-2 text-sm text-muted">
        Forward cone -- three models&apos; VaR/upper bound, today out to{" "}
        {days[days.length - 1] ?? "?"} days
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="label" tick={{ fill: "var(--muted)", fontSize: 11 }} axisLine={{ stroke: "var(--grid)" }} />
          <YAxis
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            tickFormatter={(v) => fmtUsd(v)}
            width={80}
            domain={yDomain}
            allowDataOverflow
          />
          <ReferenceLine y={lastClose} stroke="var(--green)" strokeDasharray="3 3" />
          <Tooltip
            contentStyle={{ background: "var(--bg)", border: "1px solid var(--grid)", fontSize: 12 }}
            labelStyle={{ color: "var(--text)" }}
            formatter={(value) => (value == null ? "n/a" : fmtUsd(Number(value)))}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {withData.map((s) => (
            <Line
              key={`${s.key}_var`}
              dataKey={`${s.key}_var`}
              name={`${s.label} (VaR)`}
              stroke={s.color}
              strokeDasharray={s.dash}
              strokeWidth={1.5}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          ))}
          {withData.map((s) => (
            <Line
              key={`${s.key}_upper`}
              dataKey={`${s.key}_upper`}
              name={`${s.label} (upper)`}
              stroke={s.color}
              strokeDasharray={s.dash}
              strokeWidth={1.5}
              dot={false}
              connectNulls
              isAnimationActive={false}
              legendType="none"
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-muted">
        Amber = Jump-Diffusion (Poisson jumps + diffusion, compounded exactly
        to each horizon). Violet = GARCH-EVT (mean-reverting GARCH variance
        term structure + the fitted extreme-value tail). Red dashed = a
        labeled <em>scenario</em>, not a forecast -- &quot;if the MS-GARCH
        crisis regime&apos;s own stationary volatility applied and
        persisted&quot;. None of these are Monte Carlo path simulations of a
        single future; each line is that model&apos;s own estimate of the
        return distribution at that horizon.
      </p>
    </div>
  );
}
