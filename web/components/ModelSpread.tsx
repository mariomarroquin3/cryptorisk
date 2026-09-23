"use client";

import { useMemo, useState } from "react";
import { ModelLatestRow } from "@/lib/api";
import { fmtConfidence, fmtPct } from "@/lib/format";

const W = 720;
const H = 92;
const PAD = 36;
const R = 5;

interface Placed {
  model: string;
  var: number;
  x: number;
  y: number;
  inMcs: boolean;
}

/** Every model's latest frozen VaR for one asset/alpha, as dots along one
 * axis -- the same "how much do the 20 models disagree today" spread that
 * funds the capital page's model-risk add-on, made visible at a glance.
 * Dots that would overlap are stacked instead of hidden (simple greedy
 * collision avoidance: same lane if far enough apart on x, next lane down
 * otherwise), so a tight cluster still shows every model. */
export function ModelSpread({
  rows,
  alpha,
  inMcs,
  highlight,
  onSelect,
}: {
  rows: ModelLatestRow[];
  alpha: number;
  /** model -> whether it's in the 90% MCS today, from /models/comparison. */
  inMcs: Record<string, boolean> | undefined;
  highlight: string | null;
  onSelect?: (model: string) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);

  const { placed, lo, hi, mean } = useMemo(() => {
    if (rows.length === 0) return { placed: [] as Placed[], lo: -1, hi: 0, mean: 0 };
    const vals = rows.map((r) => r.var);
    const lo = Math.min(...vals);
    const hi = Math.max(0, ...vals);
    const span = hi - lo || 1;
    const x = (v: number) => PAD + ((v - lo) / span) * (W - 2 * PAD);
    const sorted = [...rows].sort((a, b) => a.var - b.var);
    const lanes: number[] = []; // last-placed x per lane
    const minGap = R * 2.4;
    const placed: Placed[] = sorted.map((r) => {
      const px = x(r.var);
      let lane = 0;
      while (lanes[lane] != null && px - lanes[lane] < minGap) lane++;
      lanes[lane] = px;
      return { model: r.model, var: r.var, x: px, y: lane, inMcs: inMcs?.[r.model] ?? false };
    });
    return { placed, lo, hi, mean: vals.reduce((a, b) => a + b, 0) / vals.length };
  }, [rows, inMcs]);

  if (rows.length === 0) return null;
  const maxLane = Math.max(0, ...placed.map((p) => p.y));
  const height = Math.max(H, 40 + maxLane * (R * 2.4));
  const zeroX = PAD + ((0 - lo) / ((hi - lo) || 1)) * (W - 2 * PAD);
  const meanX = PAD + ((mean - lo) / ((hi - lo) || 1)) * (W - 2 * PAD);
  const active = placed.find((p) => p.model === (hover ?? highlight));

  return (
    <div className="card p-4">
      <div className="mb-1 text-sm font-medium text-text">
        Model agreement today &middot; {fmtConfidence(1 - alpha)} VaR, all 20 models
      </div>
      <p className="mb-2 text-xs leading-snug text-muted">
        Each dot is one model&apos;s latest frozen VaR. Green = inside the 90% Model Confidence Set, grey =
        statistically dominated. A tight cluster means the models agree on the day&apos;s risk; a wide spread
        is exactly what the capital page&apos;s model-risk add-on charges for.
      </p>
      <svg viewBox={`0 0 ${W} ${height}`} className="w-full" style={{ height: Math.min(140, height) }}>
        <line x1={zeroX} y1={0} x2={zeroX} y2={height} stroke="var(--grid)" strokeDasharray="2 3" />
        <line x1={meanX} y1={0} x2={meanX} y2={height} stroke="var(--muted)" strokeDasharray="4 2" />
        {placed.map((p) => {
          const isHi = p.model === highlight;
          const isHover = p.model === hover;
          return (
            <g
              key={p.model}
              onMouseEnter={() => setHover(p.model)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelect?.(p.model)}
              className={onSelect ? "cursor-pointer" : undefined}
            >
              <circle
                cx={p.x}
                cy={18 + p.y * (R * 2.4)}
                r={isHi || isHover ? R + 1.5 : R}
                fill={p.inMcs ? "var(--green)" : "var(--muted)"}
                fillOpacity={p.inMcs ? 0.85 : 0.55}
                stroke={isHi ? "var(--amber)" : "none"}
                strokeWidth={2}
              />
            </g>
          );
        })}
      </svg>
      <div className="flex min-h-[1.25rem] items-center justify-between text-xs">
        <span className="text-muted">
          {fmtPct(lo, 2)} &middot; dashed grey = mean {fmtPct(mean, 2)} &middot; dotted = 0%
        </span>
        <span className="text-text">
          {active ? `${active.model}: ${fmtPct(active.var, 2)}` : "hover a dot for its model"}
        </span>
      </div>
    </div>
  );
}
