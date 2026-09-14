"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface Fz0Bar {
  model: string;
  fz0_mean: number;
  fz0_rank: number;
  in_mcs: boolean;
}

export function Fz0BarChart({ data, title }: { data: Fz0Bar[]; title?: string }) {
  return (
    <div className="card p-4">
      {title && <div className="mb-2 text-sm text-muted">{title}</div>}
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data} margin={{ top: 20, right: 10, left: 0, bottom: 60 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis
            dataKey="model"
            angle={-40}
            textAnchor="end"
            interval={0}
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            height={80}
          />
          <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} width={70} />
          <Tooltip
            contentStyle={{
              background: "var(--bg)",
              border: "1px solid var(--grid)",
              fontSize: 12,
            }}
            labelStyle={{ color: "var(--text)" }}
            formatter={(value) => Number(value).toFixed(4)}
          />
          <Bar dataKey="fz0_mean" radius={[3, 3, 0, 0]}>
            {data.map((d) => (
              <Cell key={d.model} fill={d.in_mcs ? "var(--green)" : "var(--muted)"} />
            ))}
            <LabelList dataKey="fz0_rank" position="top" fill="var(--muted)" fontSize={11} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
