"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PcrisisBandRow, RegimeStatRow } from "@/lib/api";

const AXIS = { fill: "var(--muted)", fontSize: 11 };
const TIP = {
  contentStyle: { background: "var(--bg)", border: "1px solid var(--grid)", fontSize: 12 },
  labelStyle: { color: "var(--text)" },
};

const pct = (v: number, d = 2) => `${(v * 100).toFixed(d)}%`;
const num = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "n/a");

/** Return moments by regime, for both ways of labelling the regime. */
export function RegimeStatsTable({ rows }: { rows: RegimeStatRow[] }) {
  if (rows.length === 0) return null;
  const sources = [...new Set(rows.map((r) => r.source))];
  return (
    <div className="card p-4">
      <div className="mb-1 text-sm font-medium text-text">Daily-return statistics by regime</div>
      <p className="mb-3 text-xs leading-snug text-muted">
        A day is Crisis when P(crisis) &gt; 50%. If the two regimes are real, Crisis days must be visibly more
        volatile. Under the full-sample fit they are; under the walk-forward label the &quot;Crisis&quot;
        days are <em>not</em> more volatile (ratio below 1, or indistinguishable), so the two regimes are
        not distinct in real time.
      </p>
      <div className="space-y-4 overflow-x-auto">
        {sources.map((src) => {
          const g = rows.filter((r) => r.source === src);
          const ratio = g[0].vol_ratio;
          const p = g[0].levene_p;
          const separated = ratio > 1.2 && p < 0.05;
          return (
            <div key={src}>
              <div className="mb-1 flex flex-wrap items-baseline gap-x-3 text-xs">
                <span className="font-medium uppercase tracking-wide text-text">{src}</span>
                <span className={separated ? "text-green" : "text-red"}>
                  sd(Crisis)/sd(Normal) = {num(ratio)}
                </span>
                <span className="text-muted">Levene p = {num(p, 4)}</span>
              </div>
              <table className="w-full min-w-[34rem] text-xs tabular-nums">
                <thead>
                  <tr className="text-left text-muted">
                    {["regime", "days", "share", "mean", "sd", "mean |r|", "skew", "ex. kurtosis"].map((h) => (
                      <th key={h} className="py-1 pr-3 font-normal uppercase tracking-wide">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {g.map((r) => (
                    <tr key={r.regime} className="border-t border-grid">
                      <td className="py-1.5 pr-3" style={{ color: r.regime === "Crisis" ? "var(--red)" : "var(--green)" }}>
                        {r.regime}
                      </td>
                      <td className="pr-3">{r.n}</td>
                      <td className="pr-3">{pct(r.share, 0)}</td>
                      <td className="pr-3">{pct(r.mean)}</td>
                      <td className="pr-3">{pct(r.sd)}</td>
                      <td className="pr-3">{pct(r.mean_abs)}</td>
                      <td className="pr-3">{num(r.skew)}</td>
                      <td className="pr-3">{num(r.kurt)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Bootstrap parameter-uncertainty band around the walk-forward P(crisis). */
export function PcrisisBandChart({ rows }: { rows: PcrisisBandRow[] }) {
  if (rows.length === 0) {
    return (
      <div className="card p-4 text-sm text-muted">
        No bootstrap band for this asset yet (<code>msgarch/bootstrap_pcrisis.R</code>).
      </div>
    );
  }
  const data = rows.map((r) => ({
    date: r.date.slice(0, 10),
    point: r.p_point,
    range: [r.p_lo, r.p_hi] as [number, number],
    lo: r.p_lo,
    hi: r.p_hi,
  }));
  const width = rows.map((r) => r.p_hi - r.p_lo);
  const meanWidth = width.reduce((a, b) => a + b, 0) / width.length;
  const straddle = rows.filter((r) => r.p_lo < 0.5 && r.p_hi > 0.5).length;
  const pointOutside = rows.filter((r) => r.p_point < r.p_lo || r.p_point > r.p_hi).length;
  return (
    <div className="card p-4">
      <div className="mb-1 text-sm font-medium text-text">
        P(crisis) with a 90% bootstrap band (parameter uncertainty)
      </div>
      <p className="mb-3 text-xs leading-snug text-muted">
        At {rows.length} dates the same 500-day window is block-bootstrapped and the MS-GARCH refit each time; the band is the 5th-95th percentile of P(crisis) on the original window. Mean
        band width is {pct(meanWidth, 0)} of the whole 0-100% range, and in {straddle} of {rows.length} dates
        the band contains 50%, so whether the day is called Crisis or Normal depends on estimation noise rather
        than on the data. In {pointOutside} dates the headline estimate sits outside its own band.
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="date" tick={AXIS} minTickGap={40} />
          <YAxis domain={[0, 1]} tick={AXIS} width={40} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip
            {...TIP}
            formatter={(v) => (Array.isArray(v) ? `${(v[0] * 100).toFixed(0)}% - ${(v[1] * 100).toFixed(0)}%` : `${(Number(v) * 100).toFixed(0)}%`)}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <ReferenceLine y={0.5} stroke="var(--muted)" strokeDasharray="4 3" />
          <Area dataKey="range" name="90% bootstrap band" stroke="none" fill="var(--amber)" fillOpacity={0.3} isAnimationActive={false} />
          <Line dataKey="point" name="point estimate" stroke="var(--red)" strokeWidth={1.8} dot={{ r: 3 }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
