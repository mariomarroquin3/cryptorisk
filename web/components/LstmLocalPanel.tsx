"use client";

import { useMemo } from "react";
import {
  Bar,
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
import { AXIS, ChartCard, pct, TIP } from "@/components/ExplainCharts";
import { ChartSkeleton } from "@/components/Skeleton";
import { fmtConfidence } from "@/lib/format";
import { useLstmLocal } from "@/lib/hooks";

/** Which of the last 20 days drive today's LSTM-Vol VaR: occlusion, one day at a time. */
export function LstmLocalPanel({ asset, alpha }: { asset: string | null; alpha: number }) {
  const { data } = useLstmLocal(asset, alpha);

  const rows = useMemo(
    () =>
      (data?.days ?? []).map((d) => ({
        date: d.date.slice(5),
        contribution: d.per_day * 100,
        ret: d.ret * 100,
      })),
    [data],
  );
  const top = useMemo(() => [...(data?.days ?? [])].sort((a, b) => Math.abs(b.per_day) - Math.abs(a.per_day))[0], [data]);

  if (data === undefined) return <ChartSkeleton height={300} />;
  if (data.days.length === 0 || data.base_var == null || data.flat_var == null) {
    return (
      <div className="text-sm text-muted">
        No local LSTM explanation yet (run <code>python -m cryptorisk.study.run_lstm_local</code>).
      </div>
    );
  }
  const total = data.base_var - data.flat_var;
  return (
    <div className="space-y-3">
      <div className="card p-4 text-sm">
        As of {data.asof}, the network puts the {fmtConfidence(alpha)} VaR at{" "}
        <span className="font-semibold">{pct(-data.base_var, 2)}</span>. If the last 20 days had all been flat it would say{" "}
        <span className="font-semibold">{pct(-data.flat_var, 2)}</span>, so the recent history moves it by{" "}
        <span className="font-semibold">
          {total >= 0 ? "+" : ""}
          {(total * 100).toFixed(2)} pp
        </span>
        {top && (
          <>
            ; the single biggest driver is {top.date} ({pct(top.ret, 2)}), worth {top.per_day >= 0 ? "+" : ""}
            {(top.per_day * 100).toFixed(2)} pp.
          </>
        )}
      </div>
      <ChartCard
        title="What drives today's LSTM VaR, day by day (occlusion)"
        caption="Each bar replaces one of the last 20 days with a flat day and reports how much shallower the VaR becomes: positive = that day pushes today's VaR deeper. The line is that day's return. Compare the bars with the VaR itself (stated above): the smaller they are, the closer the forecast is to a constant that ignores recent history."
      >
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={rows} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="date" tick={AXIS} interval={1} />
            <YAxis yAxisId="c" tick={AXIS} width={50} tickFormatter={(v) => `${Number(v).toFixed(2)}`} />
            <YAxis yAxisId="r" orientation="right" tick={AXIS} width={45} tickFormatter={(v) => `${Number(v).toFixed(0)}%`} />
            <Tooltip {...TIP} formatter={(v, name) => (name === "return" ? `${Number(v).toFixed(2)}%` : `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(3)} pp`)} />
            <Legend />
            <ReferenceLine yAxisId="c" y={0} stroke="var(--muted)" />
            <Bar yAxisId="c" dataKey="contribution" name="effect on VaR (pp)" fill="#4dabf7" isAnimationActive={false} />
            <Line yAxisId="r" dataKey="ret" name="return" stroke="var(--amber)" dot={{ r: 2 }} strokeWidth={1.5} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}
