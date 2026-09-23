"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  BacktestRow,
  CoverageRow,
  Fz0Row,
  ModelInfo,
  RfDiagnosticsRow,
  RfImportanceRow,
  RfInputRow,
} from "@/lib/api";

export const AXIS = { fill: "var(--muted)", fontSize: 11 };
export const TIP = {
  contentStyle: { background: "var(--bg)", border: "1px solid var(--grid)", fontSize: 12 },
  labelStyle: { color: "var(--text)" },
};
const PALETTE = ["#ffb020", "#4dabf7", "#3ddc84", "#b083f0", "#ff4d4f", "#20c9c9", "#f78fb3", "#a0a8b5"];

export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
const shortDate = (d: string) => d.slice(0, 10);

/** Prefix sums of violations: cum[k] = violations among the first k rows. */
function cumulative(rows: BacktestRow[]): number[] {
  const out = [0];
  rows.forEach((r) => out.push(out[out.length - 1] + (r.violation ? 1 : 0)));
  return out;
}

export function ChartCard({
  title,
  caption,
  children,
}: {
  title: string;
  caption?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="card p-4">
      <div className="mb-1 text-sm font-medium text-text">{title}</div>
      {caption && <p className="mb-3 text-xs leading-snug text-muted">{caption}</p>}
      {children}
    </div>
  );
}

/** Realized return vs the model's VaR / ES, violations marked. */
export function BacktestBand({ rows }: { rows: BacktestRow[] }) {
  const data = rows.map((r) => ({
    date: shortDate(r.date),
    realized: r.realized,
    var: r.var,
    es: r.es,
  }));
  const breaches = rows
    .filter((r) => r.violation)
    .map((r) => ({ date: shortDate(r.date), breach: r.realized }));
  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="date" tick={AXIS} minTickGap={50} />
        <YAxis tick={AXIS} width={55} tickFormatter={(v) => pct(v, 0)} />
        <Tooltip {...TIP} formatter={(v) => pct(Number(v), 2)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <ReferenceLine y={0} stroke="var(--grid)" />
        <Line dataKey="realized" name="realized return" stroke="var(--muted)" dot={false} strokeWidth={1} />
        <Line dataKey="var" name="VaR" stroke="var(--amber)" dot={false} strokeWidth={1.6} />
        <Line dataKey="es" name="ES" stroke="var(--violet)" dot={false} strokeWidth={1.2} strokeDasharray="4 3" />
        <Scatter data={breaches} dataKey="breach" name="violation" fill="var(--red)" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Cumulative violations vs the expected count, with a 95% binomial band. */
export function CumulativeViolations({ rows, alpha }: { rows: BacktestRow[]; alpha: number }) {
  const cum = cumulative(rows);
  const data = rows.map((r, i) => {
    const c = cum[i + 1];
    const n = i + 1;
    const se = 1.96 * Math.sqrt(n * alpha * (1 - alpha));
    return {
      date: shortDate(r.date),
      actual: c,
      expected: n * alpha,
      lo: Math.max(0, n * alpha - se),
      hi: n * alpha + se,
    };
  });
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="date" tick={AXIS} minTickGap={50} />
        <YAxis tick={AXIS} width={40} />
        <Tooltip {...TIP} formatter={(v) => Number(v).toFixed(1)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line dataKey="hi" name="95% band" stroke="var(--grid)" dot={false} strokeDasharray="3 3" />
        <Line dataKey="lo" name="lower band" stroke="var(--grid)" dot={false} strokeDasharray="3 3" legendType="none" />
        <Line dataKey="expected" name="expected" stroke="var(--muted)" dot={false} />
        <Line dataKey="actual" name="actual violations" stroke="var(--red)" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Rolling hit rate against the nominal rate. */
export function RollingHitRate({
  rows,
  alpha,
  win = 250,
}: {
  rows: BacktestRow[];
  alpha: number;
  win?: number;
}) {
  const cum = cumulative(rows);
  const data = rows
    .map((r, i) => ({ date: shortDate(r.date), hit: (cum[i + 1] - cum[Math.max(0, i + 1 - win)]) / win, i }))
    .filter((d) => d.i >= win - 1);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="date" tick={AXIS} minTickGap={50} />
        <YAxis tick={AXIS} width={45} tickFormatter={(v) => pct(v, 1)} domain={[0, "auto"]} />
        <Tooltip {...TIP} formatter={(v) => pct(Number(v), 2)} />
        <ReferenceLine y={alpha} stroke="var(--green)" strokeDasharray="4 3" />
        <Line dataKey="hit" name={`${win}d hit rate`} stroke="var(--amber)" dot={false} strokeWidth={1.8} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** PIT histogram: flat at 10% per bin = perfectly calibrated density. */
export function PitHistogram({ rows }: { rows: BacktestRow[] }) {
  const pits = rows.map((r) => r.pit).filter((p): p is number => p != null && Number.isFinite(p));
  const bins = 10;
  const counts = Array.from({ length: bins }, () => 0);
  pits.forEach((p) => {
    counts[Math.min(bins - 1, Math.floor(p * bins))] += 1;
  });
  const n = pits.length || 1;
  const band = 1.96 * Math.sqrt((0.1 * 0.9) / n);
  const data = counts.map((c, i) => ({ bin: (i / bins).toFixed(1), share: c / n }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="bin" tick={AXIS} />
        <YAxis tick={AXIS} width={45} tickFormatter={(v) => pct(v, 0)} domain={[0, "auto"]} />
        <Tooltip {...TIP} formatter={(v) => pct(Number(v), 1)} />
        <ReferenceLine y={0.1} stroke="var(--green)" strokeDasharray="4 3" />
        <ReferenceLine y={0.1 + band} stroke="var(--grid)" strokeDasharray="2 3" />
        <ReferenceLine y={Math.max(0, 0.1 - band)} stroke="var(--grid)" strokeDasharray="2 3" />
        <Bar dataKey="share" radius={[3, 3, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={Math.abs(d.share - 0.1) > band ? "var(--red)" : "var(--blue)"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Hit rate per model vs the nominal rate; red = fails a coverage test. */
export function HitRateBars({
  rows,
  alpha,
  highlight,
}: {
  rows: CoverageRow[];
  alpha: number;
  highlight: string | null;
}) {
  const data = [...rows].sort((a, b) => a.hit_rate - b.hit_rate);
  return (
    <ResponsiveContainer width="100%" height={340}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 60 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="model" angle={-40} textAnchor="end" interval={0} tick={AXIS} height={80} />
        <YAxis tick={AXIS} width={45} tickFormatter={(v) => pct(v, 1)} />
        <Tooltip {...TIP} formatter={(v) => pct(Number(v), 2)} />
        <ReferenceLine y={alpha} stroke="var(--green)" strokeDasharray="4 3" />
        <Bar dataKey="hit_rate" radius={[3, 3, 0, 0]}>
          {data.map((d) => (
            <Cell
              key={d.model}
              fill={d.passes_all ? "var(--green)" : "var(--red)"}
              stroke={d.model === highlight ? "var(--amber)" : undefined}
              strokeWidth={d.model === highlight ? 2 : 0}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Mean FZ0 rank per method family -- does machine learning beat structure? */
export function FamilyRankChart({
  rows,
  info,
}: {
  rows: Fz0Row[];
  info: Record<string, ModelInfo> | undefined;
}) {
  const acc = new Map<string, { sum: number; n: number; best: number }>();
  rows.forEach((r) => {
    const fam = info?.[r.model]?.family || "Other";
    const a = acc.get(fam) ?? { sum: 0, n: 0, best: Infinity };
    a.sum += r.fz0_rank;
    a.n += 1;
    a.best = Math.min(a.best, r.fz0_rank);
    acc.set(fam, a);
  });
  const data = [...acc.entries()]
    .map(([family, a]) => ({ family, mean_rank: a.sum / a.n, best_rank: a.best }))
    .sort((x, y) => x.mean_rank - y.mean_rank);
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data} layout="vertical" margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" horizontal={false} />
        <XAxis type="number" tick={AXIS} domain={[0, rows.length || "auto"]} />
        <YAxis type="category" dataKey="family" tick={AXIS} width={150} />
        <Tooltip {...TIP} formatter={(v) => Number(v).toFixed(1)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar dataKey="mean_rank" name="mean FZ0 rank (lower = better)" radius={[0, 3, 3, 0]}>
          {data.map((d) => (
            <Cell key={d.family} fill={d.family === "Machine learning" ? "var(--amber)" : "var(--blue)"} />
          ))}
        </Bar>
        <Bar dataKey="best_rank" name="best model's rank" fill="var(--green)" radius={[0, 3, 3, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function pivotImportance(rows: RfImportanceRow[]) {
  const feats = [...new Set(rows.map((r) => r.feature))];
  const byDate = new Map<string, Record<string, number | string>>();
  rows.forEach((r) => {
    const d = shortDate(r.date);
    const o = byDate.get(d) ?? { date: d };
    o[r.feature] = r.importance;
    byDate.set(d, o);
  });
  const data = [...byDate.values()].sort((a, b) => String(a.date).localeCompare(String(b.date)));
  return { feats, data };
}

/** Mean impurity importance per feature across the OOS refits. */
export function RfImportanceBars({ rows }: { rows: RfImportanceRow[] }) {
  const acc = new Map<string, number[]>();
  rows.forEach((r) => acc.set(r.feature, [...(acc.get(r.feature) ?? []), r.importance]));
  const data = [...acc.entries()]
    .map(([feature, v]) => ({ feature, importance: v.reduce((a, b) => a + b, 0) / v.length }))
    .sort((a, b) => b.importance - a.importance);
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} layout="vertical" margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" horizontal={false} />
        <XAxis type="number" tick={AXIS} tickFormatter={(v) => pct(v, 0)} />
        <YAxis type="category" dataKey="feature" tick={AXIS} width={90} />
        <Tooltip {...TIP} formatter={(v) => pct(Number(v), 1)} />
        <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
          {data.map((d, i) => (
            <Cell key={d.feature} fill={PALETTE[i % PALETTE.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** How the forest's feature reliance drifts over time (stacked shares). */
export function RfImportanceOverTime({ rows }: { rows: RfImportanceRow[] }) {
  const { feats, data } = pivotImportance(rows);
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data} stackOffset="expand" margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="date" tick={AXIS} minTickGap={50} />
        <YAxis tick={AXIS} width={45} tickFormatter={(v) => pct(v, 0)} />
        <Tooltip {...TIP} formatter={(v) => pct(Number(v), 1)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {feats.map((f, i) => (
          <Area
            key={f}
            dataKey={f}
            stackId="1"
            stroke={PALETTE[i % PALETTE.length]}
            fill={PALETTE[i % PALETTE.length]}
            fillOpacity={0.55}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

/** Effective number of training days behind each forecast. */
export function RfEssChart({ rows }: { rows: RfDiagnosticsRow[] }) {
  const data = rows.map((r) => ({ date: shortDate(r.date), ess: r.ess, n: r.n_train }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="date" tick={AXIS} minTickGap={50} />
        <YAxis tick={AXIS} width={45} />
        <Tooltip {...TIP} formatter={(v) => Number(v).toFixed(0)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line dataKey="n" name="training days in window" stroke="var(--muted)" dot={false} strokeDasharray="4 3" />
        <Line dataKey="ess" name="effective sample size" stroke="var(--amber)" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Conditional VaR vs flat-weight (HS) VaR: what the features add. */
export function RfConditioningShift({ rows }: { rows: RfDiagnosticsRow[] }) {
  const data = rows.map((r) => ({ date: shortDate(r.date), cond: r.var_cond, hs: r.var_hs }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="date" tick={AXIS} minTickGap={50} />
        <YAxis tick={AXIS} width={55} tickFormatter={(v) => pct(v, 0)} />
        <Tooltip {...TIP} formatter={(v) => pct(Number(v), 2)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line dataKey="hs" name="flat-weight VaR (HS)" stroke="var(--muted)" dot={false} strokeDasharray="4 3" />
        <Line dataKey="cond" name="forest-conditioned VaR" stroke="var(--amber)" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Today's forecast inputs as z-scores of their own window. */
export function RfInputsChart({ rows }: { rows: RfInputRow[] }) {
  const data = rows.map((r) => ({ feature: r.feature, zscore: r.zscore }));
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} layout="vertical" margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" horizontal={false} />
        <XAxis type="number" tick={AXIS} domain={[-3, 3]} />
        <YAxis type="category" dataKey="feature" tick={AXIS} width={90} />
        <Tooltip {...TIP} formatter={(v) => `${Number(v).toFixed(2)} sd`} />
        <ReferenceLine x={0} stroke="var(--muted)" />
        <Bar dataKey="zscore" radius={[0, 3, 3, 0]}>
          {data.map((d) => (
            <Cell key={d.feature} fill={d.zscore >= 0 ? "var(--red)" : "var(--blue)"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
