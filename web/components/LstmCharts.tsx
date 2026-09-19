"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { LstmImportanceRow } from "@/lib/api";

const AXIS = { fill: "var(--muted)", fontSize: 11 };
const TIP = {
  contentStyle: { background: "var(--bg)", border: "1px solid var(--grid)", fontSize: 12 },
  labelStyle: { color: "var(--text)" },
};
const COLORS = ["var(--amber)", "var(--blue)", "var(--violet)"];

const FEATURES = ["return", "squared return", "squared down-return"];

function grid(rows: LstmImportanceRow[]) {
  const lags = [...new Set(rows.map((r) => r.lag))].sort((a, b) => a - b);
  const at = new Map(rows.map((r) => [`${r.lag}|${r.feature}`, r.importance]));
  return { lags, at };
}

/** Which (lag, feature) cells of the input window the fitted network needs:
 * rise in the training loss when that one cell is shuffled across sequences. */
export function LstmImportanceHeatmap({ rows }: { rows: LstmImportanceRow[] }) {
  const { lags, at } = grid(rows);
  const max = Math.max(1e-12, ...rows.map((r) => Math.max(0, r.importance)));
  return (
    <div className="overflow-x-auto">
      <div className="min-w-[34rem] space-y-1">
        {FEATURES.map((f) => (
          <div key={f} className="flex items-center gap-1">
            <div className="w-36 shrink-0 text-xs text-muted">{f}</div>
            {lags.map((l) => {
              const v = at.get(`${l}|${f}`) ?? 0;
              const t = Math.max(0, v) / max;
              return (
                <div
                  key={l}
                  title={`${f}, ${l} day${l > 1 ? "s" : ""} ago: +${v.toExponential(2)} loss`}
                  className="h-8 flex-1 rounded-sm border border-grid/50"
                  style={{ background: `rgba(255, 176, 32, ${0.06 + 0.94 * Math.sqrt(t)})` }}
                />
              );
            })}
          </div>
        ))}
        <div className="flex items-center gap-1 pt-1">
          <div className="w-36 shrink-0" />
          {lags.map((l) => (
            <div key={l} className="flex-1 text-center text-[0.6rem] text-muted">
              {l === 1 || l % 5 === 0 ? l : ""}
            </div>
          ))}
        </div>
        <div className="pl-[9.25rem] text-[0.65rem] text-muted">days ago (1 = yesterday) &rarr;</div>
      </div>
    </div>
  );
}

/** Share of total importance by input channel. */
export function LstmFeatureShare({ rows }: { rows: LstmImportanceRow[] }) {
  const total = rows.reduce((a, r) => a + Math.max(0, r.importance), 0) || 1;
  const data = FEATURES.map((f) => ({
    feature: f,
    share: rows.filter((r) => r.feature === f).reduce((a, r) => a + Math.max(0, r.importance), 0) / total,
  }));
  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" horizontal={false} />
        <XAxis type="number" domain={[0, 1]} tick={AXIS} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
        <YAxis type="category" dataKey="feature" tick={AXIS} width={130} />
        <Tooltip {...TIP} formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
        <Bar dataKey="share" radius={[0, 3, 3, 0]}>
          {data.map((d, i) => (
            <Cell key={d.feature} fill={COLORS[i % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Total importance by lag: how far back the network actually looks. */
export function LstmLagProfile({ rows }: { rows: LstmImportanceRow[] }) {
  const { lags } = grid(rows);
  const data = lags.map((l) => ({
    lag: String(l),
    importance: rows.filter((r) => r.lag === l).reduce((a, r) => a + Math.max(0, r.importance), 0),
  }));
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="lag" tick={AXIS} />
        <YAxis tick={AXIS} width={55} tickFormatter={(v) => Number(v).toExponential(0)} />
        <Tooltip {...TIP} formatter={(v) => Number(v).toExponential(2)} />
        <Bar dataKey="importance" name="loss increase" fill="var(--amber)" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
