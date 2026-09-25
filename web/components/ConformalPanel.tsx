"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
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
import { Column, DataTable } from "@/components/DataTable";
import { AXIS, ChartCard, pct, TIP } from "@/components/ExplainCharts";
import { ModelTip } from "@/components/ModelTip";
import { ChartSkeleton } from "@/components/Skeleton";
import { ConformalRow } from "@/lib/api";
import { fmtConfidence } from "@/lib/format";
import { useConformalPath, useConformalSummary, useModelsInfo } from "@/lib/hooks";

const COLORS = ["#4dabf7", "#ffb020", "#ff4d4f", "#3ddc84", "#b083f0", "#40c4ff"];

const fmtP = (p: number | null | undefined) => (p == null || !Number.isFinite(p) ? "n/a" : p.toFixed(3));

/** One sentence per wrapped model, built from the numbers rather than written in advance. */
function verdicts(rows: ConformalRow[], alpha: number): string[] {
  const out: string[] = [];
  for (const base of [...new Set(rows.map((r) => r.base_model))]) {
    const raw = rows.find((r) => r.base_model === base && r.variant === "raw");
    const aci = rows.find((r) => r.base_model === base && r.variant === "ACI");
    if (!raw || !aci) continue;
    const rawOk = raw.kupiec_p >= 0.05;
    const aciOk = aci.kupiec_p >= 0.05;
    const coverage = `${pct(raw.hit_rate, 2)} to ${pct(aci.hit_rate, 2)} (target ${pct(alpha, 1)}; Kupiec p ${fmtP(raw.kupiec_p)} to ${fmtP(aci.kupiec_p)})`;
    const dm = aci.dm_fz0_diff;
    const score =
      dm == null || !Number.isFinite(dm)
        ? ""
        : ` FZ0 ${dm < 0 ? "improves" : "worsens"} by ${Math.abs(dm).toFixed(3)}${
            aci.dm_p != null && aci.dm_p < 0.05 ? " (significant)" : " (not significant)"
          }.`;
    const closer = Math.abs(aci.hit_rate - alpha) < Math.abs(raw.hit_rate - alpha);
    const status =
      !rawOk && aciOk
        ? "repairs coverage"
        : rawOk && aciOk
          ? closer ? "keeps coverage, closer to the target" : "keeps coverage"
          : rawOk ? "breaks coverage" : "does not fix coverage";
    out.push(`${base}: ACI ${status}: hit rate ${coverage}.${score}`);
  }
  return out;
}

/** Did adaptive conformal recalibration repair the models' calibration, and at what cost? */
export function ConformalPanel({ asset, alpha }: { asset: string | null; alpha: number }) {
  const { data: rows } = useConformalSummary(asset, alpha);
  const { data: path } = useConformalPath(asset, alpha);
  const { data: info } = useModelsInfo();

  const bars = useMemo(() => {
    const bases = [...new Set((rows ?? []).map((r) => r.base_model))];
    return bases.map((b) => ({
      model: b,
      raw: rows?.find((r) => r.base_model === b && r.variant === "raw")?.hit_rate ?? null,
      aci: rows?.find((r) => r.base_model === b && r.variant === "ACI")?.hit_rate ?? null,
    }));
  }, [rows]);

  const lines = useMemo(() => {
    const byDate = new Map<string, Record<string, number | string>>();
    for (const p of path ?? []) {
      const d = p.date.slice(0, 10);
      const row = byDate.get(d) ?? { date: d };
      row[p.model.replace("+ACI", "")] = p.level;
      byDate.set(d, row);
    }
    return [...byDate.values()].sort((a, b) => String(a.date).localeCompare(String(b.date)));
  }, [path]);
  const models = useMemo(() => [...new Set((path ?? []).map((p) => p.model.replace("+ACI", "")))], [path]);

  const cols: Column<ConformalRow>[] = [
    { key: "base_model", label: "model", render: (r) => <ModelTip name={r.base_model} info={info?.[r.base_model]} /> },
    { key: "variant", label: "variant", help: "raw = the model as in the study; ACI = the same model with adaptive conformal recalibration." },
    { key: "n", label: "days" },
    { key: "hit_rate", label: "hit rate", render: (r) => pct(r.hit_rate, 2), help: "Share of days the return fell below the VaR. The target is the tail probability." },
    { key: "kupiec_p", label: "Kupiec p", pShade: true, render: (r) => fmtP(r.kupiec_p), help: "Tests that the hit rate equals the target. Below 0.05 = miscalibrated." },
    { key: "chr_cc_p", label: "CC p", pShade: true, render: (r) => fmtP(r.chr_cc_p), help: "Christoffersen conditional coverage: right hit rate AND violations not clustered." },
    { key: "dq_p", label: "DQ p", pShade: true, render: (r) => fmtP(r.dq_p), help: "Engle-Manganelli dynamic quantile test: violations should be unpredictable from the past." },
    { key: "fz0_mean", label: "mean FZ0", render: (r) => r.fz0_mean.toFixed(4), help: "Joint VaR/ES score; lower is better." },
    { key: "dm_p", label: "DM p (ACI vs raw)", render: (r) => (r.variant === "ACI" ? fmtP(r.dm_p) : ""), help: "Diebold-Mariano test of the FZ0 difference between the ACI version and the raw one." },
  ];

  if (rows === undefined) return <ChartSkeleton height={320} />;
  if (rows.length === 0) {
    return (
      <div className="text-sm text-muted">
        No conformal results yet (run <code>make conformal</code>).
      </div>
    );
  }
  return (
    <div className="space-y-4">
      <ul className="card list-disc space-y-1 p-4 pl-8 text-sm">
        {verdicts(rows, alpha).map((v) => (
          <li key={v}>{v}</li>
        ))}
      </ul>
      <div className="grid gap-6 md:grid-cols-2">
        <ChartCard
          title={`Hit rate: raw vs ACI @ ${fmtConfidence(alpha)}`}
          caption="Dashed green line = the target rate. ACI shifts the tail level the model is asked for so that, over time, breaches occur at the target rate whatever the model does."
        >
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={bars} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="model" tick={AXIS} />
              <YAxis tick={AXIS} width={45} tickFormatter={(v) => pct(v, 1)} />
              <Tooltip {...TIP} formatter={(v) => pct(Number(v), 2)} />
              <Legend />
              <ReferenceLine y={alpha} stroke="var(--green)" strokeDasharray="4 3" />
              <Bar dataKey="raw" name="raw" fill="#4dabf7" isAnimationActive={false} />
              <Bar dataKey="aci" name="ACI" fill="var(--amber)" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard
          title="The level ACI asked each model for"
          caption="Starts at the target tail probability. It drifts below it when a model keeps being breached (ask for a deeper quantile) and above it when a model is too conservative. The size of the drift is how far the raw model was off."
        >
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lines} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="date" tick={AXIS} minTickGap={50} tickFormatter={(v: string) => v.slice(0, 7)} />
              <YAxis tick={AXIS} width={50} tickFormatter={(v) => pct(v, 2)} />
              <Tooltip {...TIP} formatter={(v) => pct(Number(v), 3)} />
              <Legend />
              <ReferenceLine y={alpha} stroke="var(--green)" strokeDasharray="4 3" />
              {models.map((m, i) => (
                <Line key={m} dataKey={m} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={1.5} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
      <DataTable columns={cols} rows={rows} keyField={(r) => `${r.base_model}-${r.variant}`} filterKey="base_model" />
    </div>
  );
}
