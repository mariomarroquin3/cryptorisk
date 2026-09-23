"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { RfPdp } from "@/lib/api";

const AXIS = { fill: "var(--muted)", fontSize: 11 };
const TIP = {
  contentStyle: { background: "var(--bg)", border: "1px solid var(--grid)", fontSize: 12 },
  labelStyle: { color: "var(--text)" },
};
const pct = (v: number, d = 2) => `${(v * 100).toFixed(d)}%`;

/** RF-QR partial dependence: sweep one feature across its historical range,
 * every other input pinned at today's actual value, and read off the
 * resulting VaR at each point -- a live "what if this input were different"
 * curve for the forest's current forecast, not an average over history. */
export function RfPdpChart({ pdp }: { pdp: RfPdp }) {
  const data = pdp.grid.map((g, i) => ({ x: g, var: pdp.var[i] }));
  return (
    <>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis
            dataKey="x"
            tick={AXIS}
            tickFormatter={(v) => Number(v).toExponential(1)}
            type="number"
            domain={["dataMin", "dataMax"]}
          />
          <YAxis tick={AXIS} width={55} tickFormatter={(v) => pct(v, 1)} />
          <Tooltip {...TIP} formatter={(v) => pct(Number(v))} labelFormatter={(v) => `${pdp.feature} = ${Number(v).toExponential(2)}`} />
          <ReferenceLine
            x={pdp.actual}
            stroke="var(--amber)"
            strokeDasharray="4 3"
            label={{ value: "today", fill: "var(--amber)", fontSize: 10, position: "insideTopRight" }}
          />
          <Line dataKey="var" stroke="var(--blue)" dot={{ r: 2 }} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <p className="mt-1 text-xs text-muted">
        Amber line is today&apos;s actual value of {pdp.feature} ({pdp.actual.toExponential(2)}); the curve is
        VaR if only that input had been different, holding everything else at today&apos;s window.
      </p>
    </>
  );
}
