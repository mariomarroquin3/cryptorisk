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

const NOTES: Record<string, string> = {
  jd: "Poisson jumps + diffusion, compounded exactly to each horizon (jump count and variance both scale with days, not sqrt(days)).",
  evt: "Mean-reverting GARCH variance term structure (the sum of each step's forecast, not a flat scaling) plus the fitted extreme-value tail.",
};

/** One specialized model's forward VaR/upper bound, converging on today's
 * price -- deliberately its own small chart per model (call this once per
 * series) rather than overlaying several models' lines on one chart, which
 * gets busy fast. Passing >1 series still works (lines, not filled areas,
 * so they stay legible layered), for callers that want that instead. */
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

  const title =
    withData.length === 1
      ? `${withData[0].label} -- forward VaR/upper, today out to ${days[days.length - 1] ?? "?"} days`
      : `Forward cone -- VaR/upper bound, today out to ${days[days.length - 1] ?? "?"} days`;

  return (
    <div className="rounded border border-grid bg-panel p-3">
      <div className="mb-2 text-sm text-muted">{title}</div>
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
        {withData.length === 1
          ? (NOTES[withData[0].key] ?? "")
          : withData.map((s) => `${s.label}: ${NOTES[s.key] ?? ""}`).join(" ")}{" "}
        Not a Monte Carlo path simulation of a single future -- this is the
        model&apos;s own estimate of the return distribution at each horizon.
      </p>
    </div>
  );
}
