"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { RegimeCorrRow, RegimeSummary } from "@/lib/api";

const AXIS = { fill: "var(--muted)", fontSize: 11 };
const TIP = {
  contentStyle: { background: "var(--bg)", border: "1px solid var(--grid)", fontSize: 12 },
  labelStyle: { color: "var(--text)" },
};

export interface RegimeRow {
  date: string;
  insample: number; // full-sample fit P(crisis)
  wf: number; // walk-forward one-step-ahead P(crisis)
  sigma: number | null; // MS-GARCH next-day sigma
  vol21: number | null; // realized 21d volatility (daily sd of returns)
}

const CRISIS = "var(--red)";
const NORMAL = "var(--green)";

/** Lengths of consecutive same-state runs, split by state (true = crisis). */
function runLengths(states: boolean[]) {
  const out = { normal: [] as number[], crisis: [] as number[] };
  let i = 0;
  while (i < states.length) {
    let j = i;
    while (j < states.length && states[j] === states[i]) j++;
    (states[i] ? out.crisis : out.normal).push(j - i);
    i = j;
  }
  return out;
}

const mean = (x: number[]) => (x.length ? x.reduce((a, b) => a + b, 0) / x.length : NaN);

/** The thesis in one picture: the regime probability tracks volatility only
 * when it is fitted on the whole sample. */
export function CorrelationContrast({ rows }: { rows: RegimeCorrRow[] }) {
  const pick = (s: string) => rows.find((r) => r.series === s)?.corr_absret;
  const ins = pick("insample");
  const wf = pick("filt_wf");
  const data = [
    { name: "full-sample fit (in-sample)", corr: ins, fill: "var(--blue)" },
    { name: "walk-forward, filtered", corr: wf, fill: CRISIS },
    { name: "walk-forward, one-step-ahead", corr: pick("pred_wf"), fill: CRISIS },
  ].filter((d) => d.corr != null);
  if (ins == null || wf == null) return null;
  return (
    <div className="card p-4">
      <div className="mb-1 text-sm font-medium text-text">
        Does P(crisis) track volatility? corr(P(crisis), |return|)
      </div>
      <p className="mb-3 text-xs leading-snug text-muted">
        Same model, same data. Fitted once on the whole sample it follows volatility; refit on the 500-day
        windows a real forecaster has, the correlation vanishes. The regime signal has no real-time content.
      </p>
      <div className="grid gap-4 md:grid-cols-[minmax(0,14rem)_1fr]">
        <div className="flex flex-col justify-center gap-3">
          <div>
            <div className="text-[0.65rem] uppercase tracking-wide text-muted">in-sample</div>
            <div className="font-display text-4xl font-semibold" style={{ color: "var(--blue)" }}>
              {ins.toFixed(3)}
            </div>
          </div>
          <div>
            <div className="text-[0.65rem] uppercase tracking-wide text-muted">walk-forward</div>
            <div className="font-display text-4xl font-semibold" style={{ color: CRISIS }}>
              {wf.toFixed(3)}
            </div>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 0 }}>
            <CartesianGrid stroke="var(--grid)" horizontal={false} />
            <XAxis type="number" domain={[-0.2, 1]} tick={AXIS} />
            <YAxis type="category" dataKey="name" tick={AXIS} width={170} />
            <Tooltip {...TIP} formatter={(v) => Number(v).toFixed(3)} />
            <ReferenceLine x={0} stroke="var(--muted)" />
            <Bar dataKey="corr" radius={[0, 3, 3, 0]}>
              {data.map((d) => (
                <Cell key={d.name} fill={d.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** 2x2 transition matrix from the fitted persistence of each regime. */
export function TransitionMatrix({ summary }: { summary: RegimeSummary | null | undefined }) {
  const pn = summary?.normal.p_stay;
  const pc = summary?.crisis.p_stay;
  if (pn == null || pc == null) {
    return <div className="text-sm text-muted">No fitted transition probabilities for this asset.</div>;
  }
  const cells = [
    { row: "from Normal", col: "to Normal", p: pn, stay: true },
    { row: "from Normal", col: "to Crisis", p: 1 - pn, stay: false },
    { row: "from Crisis", col: "to Normal", p: 1 - pc, stay: false },
    { row: "from Crisis", col: "to Crisis", p: pc, stay: true },
  ];
  const dur = (p: number) => (p < 1 ? 1 / (1 - p) : Infinity);
  return (
    <div className="card p-4">
      <div className="mb-1 text-sm font-medium text-text">Transition matrix P (fitted, last window)</div>
      <p className="mb-3 text-xs leading-snug text-muted">
        Diagonal = persistence. A regime that stays put only {(pn * 100).toFixed(0)}% / {(pc * 100).toFixed(0)}% of
        the time lasts on average {dur(pn).toFixed(1)} / {dur(pc).toFixed(1)} days: closer to noise than to a
        market regime.
      </p>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5 text-center text-xs">
        <div />
        <div className="text-muted">to Normal</div>
        <div className="text-muted">to Crisis</div>
        {[0, 1].map((r) => (
          <div key={r} className="contents">
            <div className="flex items-center pr-2 text-muted">{r === 0 ? "from Normal" : "from Crisis"}</div>
            {cells.slice(r * 2, r * 2 + 2).map((c) => {
              const tone = c.stay ? "34, 197, 94" : "255, 77, 79";
              return (
                <div
                  key={c.col + c.row}
                  className="rounded-md border border-grid py-5 font-display text-2xl font-semibold"
                  style={{ background: `rgba(${tone}, ${0.12 + 0.6 * c.p})` }}
                >
                  {(c.p * 100).toFixed(0)}%
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

/** One coloured cell per day, three stacked rows, with a hover readout. */
export function RegimeTimeline({ rows }: { rows: RegimeRow[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const n = rows.length;
  if (n === 0) return null;
  const vols = rows.map((r) => r.vol21).filter((v): v is number => v != null).sort((a, b) => a - b);
  const q80 = vols.length ? vols[Math.floor(vols.length * 0.8)] : Infinity;
  const strips = [
    { label: "In-sample fit: P(crisis) > 50%", on: (r: RegimeRow) => r.insample > 0.5, color: CRISIS },
    { label: "Walk-forward: P(crisis) > 50%", on: (r: RegimeRow) => r.wf > 0.5, color: CRISIS },
    {
      label: "Reality: realized vol in the top 20% of days",
      on: (r: RegimeRow) => r.vol21 != null && r.vol21 >= q80,
      color: "var(--amber)",
    },
  ];
  const H = 22;
  const h = hover != null ? rows[hover] : null;
  return (
    <div className="card p-4">
      <div className="mb-1 text-sm font-medium text-text">Regime timeline (every out-of-sample day)</div>
      <p className="mb-3 text-xs leading-snug text-muted">
        Coloured = flagged high-volatility. The in-sample row lines up with the real high-vol stretches; the
        walk-forward row flips almost at random. Same model, same days.
      </p>
      <svg
        viewBox={`0 0 ${n} ${strips.length * (H + 4)}`}
        preserveAspectRatio="none"
        className="h-28 w-full cursor-crosshair"
        onMouseMove={(e) => {
          const b = e.currentTarget.getBoundingClientRect();
          setHover(Math.min(n - 1, Math.max(0, Math.floor(((e.clientX - b.left) / b.width) * n))));
        }}
        onMouseLeave={() => setHover(null)}
      >
        {strips.map((s, i) =>
          rows.map((r, k) => (
            <rect
              key={`${i}-${k}`}
              x={k}
              y={i * (H + 4)}
              width={1.05}
              height={H}
              fill={s.on(r) ? s.color : "var(--grid-soft)"}
            />
          )),
        )}
        {hover != null && (
          <rect x={hover - 1} y={0} width={3} height={strips.length * (H + 4)} fill="var(--text)" opacity={0.7} />
        )}
      </svg>
      <div className="mt-2 space-y-0.5 text-[0.7rem] text-muted">
        {strips.map((s) => (
          <div key={s.label} className="flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: s.color }} />
            {s.label}
          </div>
        ))}
      </div>
      <div className="mt-2 h-4 text-xs text-text">
        {h &&
          `${h.date}  ·  in-sample P ${(h.insample * 100).toFixed(0)}%  ·  walk-forward P ${(h.wf * 100).toFixed(0)}%  ·  realized 21d vol ${
            h.vol21 != null ? (h.vol21 * 100).toFixed(2) + "%" : "n/a"
          }`}
      </div>
    </div>
  );
}

const BINS: [string, number, number][] = [
  ["1", 1, 1],
  ["2", 2, 2],
  ["3", 3, 3],
  ["4-5", 4, 5],
  ["6-10", 6, 10],
  ["11-20", 11, 20],
  ["21+", 21, Infinity],
];

/** How long do the regimes last? Run-length distribution by state. */
export function DurationHistogram({
  rows,
  summary,
}: {
  rows: RegimeRow[];
  summary: RegimeSummary | null | undefined;
}) {
  const [src, setSrc] = useState<"insample" | "wf">("wf");
  const runs = runLengths(rows.map((r) => (src === "insample" ? r.insample : r.wf) > 0.5));
  const data = BINS.map(([label, lo, hi]) => ({
    bin: label,
    normal: runs.normal.filter((d) => d >= lo && d <= hi).length,
    crisis: runs.crisis.filter((d) => d >= lo && d <= hi).length,
  }));
  const dur = (p: number | null | undefined) => (p != null && p < 1 ? (1 / (1 - p)).toFixed(1) : "n/a");
  return (
    <div className="card p-4">
      <div className="mb-1 flex items-center justify-between gap-2">
        <div className="text-sm font-medium text-text">Regime duration (days per episode)</div>
        <div className="flex overflow-hidden rounded-md border border-grid text-xs">
          {(["wf", "insample"] as const).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setSrc(k)}
              className={`px-2.5 py-1 ${src === k ? "bg-panel-2 text-text" : "text-muted hover:text-text"}`}
            >
              {k === "wf" ? "walk-forward" : "in-sample"}
            </button>
          ))}
        </div>
      </div>
      <p className="mb-3 text-xs leading-snug text-muted">
        Mean run: Normal {mean(runs.normal).toFixed(1)}d, Crisis {mean(runs.crisis).toFixed(1)}d over{" "}
        {runs.normal.length + runs.crisis.length} episodes (the fitted P implies {dur(summary?.normal.p_stay)}d /{" "}
        {dur(summary?.crisis.p_stay)}d). Durable regimes would pile up on the right; a spike at 1-3 days means
        the labels flicker.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="bin" tick={AXIS} />
          <YAxis tick={AXIS} width={40} allowDecimals={false} />
          <Tooltip {...TIP} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="normal" name="Normal episodes" fill={NORMAL} radius={[3, 3, 0, 0]} />
          <Bar dataKey="crisis" name="Crisis episodes" fill={CRISIS} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** MS-GARCH next-day sigma split by its own dominant regime, against realized vol. */
export function RegimeVolChart({ rows }: { rows: RegimeRow[] }) {
  const data = rows.map((r, i) => {
    const crisis = r.wf > 0.5;
    const prevCrisis = i > 0 ? rows[i - 1].wf > 0.5 : crisis;
    const nextCrisis = i < rows.length - 1 ? rows[i + 1].wf > 0.5 : crisis;
    // A point next to a regime change belongs to both lines, so each
    // coloured segment spans the change with no visual gap.
    return {
      date: r.date,
      sigma_normal: !crisis || !prevCrisis || !nextCrisis ? r.sigma : null,
      sigma_crisis: crisis || prevCrisis || nextCrisis ? r.sigma : null,
      realized: r.vol21,
    };
  });
  return (
    <div className="card p-4">
      <div className="mb-1 text-sm font-medium text-text">
        MS-GARCH conditional volatility, coloured by its own regime
      </div>
      <p className="mb-3 text-xs leading-snug text-muted">
        The model&apos;s next-day sigma (walk-forward) in green while it believes it is in the Normal regime, red
        while in Crisis, against realized 21-day volatility. If the model saw crises, red would sit on the
        realized-vol spikes.
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="date" tick={AXIS} minTickGap={50} />
          <YAxis tick={AXIS} width={50} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip {...TIP} formatter={(v) => `${(Number(v) * 100).toFixed(2)}%`} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line dataKey="realized" name="realized 21d vol" stroke="var(--amber)" dot={false} strokeDasharray="4 3" strokeWidth={1.2} isAnimationActive={false} />
          <Line dataKey="sigma_normal" name="MS-GARCH sigma (Normal)" stroke={NORMAL} dot={false} strokeWidth={1.6} connectNulls={false} isAnimationActive={false} />
          <Line dataKey="sigma_crisis" name="MS-GARCH sigma (Crisis)" stroke={CRISIS} dot={false} strokeWidth={1.6} connectNulls={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
