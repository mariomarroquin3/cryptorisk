"use client";

import { Suspense } from "react";
import { ChartCard } from "@/components/ExplainCharts";
import { Column, DataTable } from "@/components/DataTable";
import { Field } from "@/components/Field";
import { LiveFz0Bars, RealizedVsBand } from "@/components/LiveCharts";
import { ModelTip } from "@/components/ModelTip";
import { ChartSkeleton } from "@/components/Skeleton";
import { fmtConfidence, fmtDate, fmtPct } from "@/lib/format";
import { LiveModelRow } from "@/lib/api";
import { useConfig, useLiveTrackRecord, useModelsInfo } from "@/lib/hooks";
import { useQueryParam } from "@/lib/useQueryParam";

export default function LivePage() {
  return (
    <Suspense>
      <LivePageInner />
    </Suspense>
  );
}

function LivePageInner() {
  const { data: config } = useConfig();
  const { data: info } = useModelsInfo();
  const assets = config?.assets ?? [];
  const alphas = config?.alphas ?? [];
  const [asset, setAsset] = useQueryParam("asset");
  const [alphaStr, setAlphaStr] = useQueryParam("alpha");
  const effAsset = asset ?? assets[0] ?? null;
  const effAlpha = alphaStr != null ? Number(alphaStr) : (alphas[0] ?? 0.025);
  const { data } = useLiveTrackRecord(effAsset, effAlpha);

  const days = data?.days ?? [];
  const rows = data?.models ?? [];
  const n = rows[0]?.n ?? 0;
  const breached = rows.filter((r) => r.violations > 0);

  const cols: Column<LiveModelRow>[] = [
    { key: "fz0_rank", label: "rank", help: "Rank by mean FZ0 over the live days (1 = best). With so few days it is noisy." },
    { key: "model", label: "model", render: (r) => <ModelTip name={r.model} info={info?.[r.model]} /> },
    { key: "violations", label: "violations", help: "Days the realized return fell below the model VaR." },
    { key: "expected", label: "expected", render: (r) => r.expected.toFixed(2), help: "Days scored times the tail probability." },
    {
      key: "p_at_least",
      label: "P(at least this many)",
      pShade: true,
      render: (r) => r.p_at_least.toFixed(3),
      help: "Binomial probability of seeing this many violations or more if the model were calibrated. Small = suspicious; with a few weeks of data it is rarely small.",
    },
    { key: "mean_fz0", label: "mean FZ0", render: (r) => r.mean_fz0.toFixed(3), help: "Joint VaR/ES scoring loss over the live days. Lower is better." },
    { key: "mean_var", label: "avg VaR", render: (r) => fmtPct(r.mean_var, 2) },
    { key: "mean_es", label: "avg ES", render: (r) => fmtPct(r.mean_es, 2) },
    {
      key: "min_margin",
      label: "tightest day",
      render: (r) => fmtPct(r.min_margin, 2),
      help: "Smallest gap between the realized return and the VaR over the live days. Negative = breached; near zero = a close call.",
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Live track record</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          How each model has done on the days after the study&apos;s frozen sample end. None of these
          days entered the study, so this is genuinely out-of-sample. Each model is re-run day by day
          with the same window and code as the backtest.
        </p>
      </div>

      <div className="flex flex-wrap gap-4">
        <Field label="Asset">
          <select className="select" value={effAsset ?? ""} onChange={(e) => setAsset(e.target.value)}>
            {assets.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Confidence">
          <select className="select" value={effAlpha} onChange={(e) => setAlphaStr(e.target.value)}>
            {alphas.map((a) => (
              <option key={a} value={a}>
                {fmtConfidence(a)}
              </option>
            ))}
          </select>
        </Field>
      </div>

      {data === undefined ? (
        <ChartSkeleton height={320} />
      ) : days.length === 0 ? (
        <div className="text-sm text-muted">No live days yet for this selection.</div>
      ) : (
        <>
          <div className="card p-4 text-sm">
            <span className="font-semibold">{n} days</span> scored ({fmtDate(days[0].date)} to{" "}
            {fmtDate(days[days.length - 1].date)}). At {fmtConfidence(effAlpha)} a calibrated model is
            expected to break about {(n * effAlpha).toFixed(1)} times.{" "}
            {breached.length === 0
              ? "No model has been breached."
              : `${breached.length} of ${rows.length} models breached at least once.`}{" "}
            <span className="text-muted">
              That is far too short to tell a good model from a lucky one: read it as a monitor, not a
              verdict, and expect the picture to sharpen as days accumulate.
            </span>
          </div>

          <ChartCard
            title="Realized return vs the range of model VaRs"
            caption="The shaded band spans the lowest to the highest VaR among all models each day; the dashed line is the median ES. A breach is the realized return dropping below the band's lower edge."
          >
            <RealizedVsBand days={days} />
          </ChartCard>

          <ChartCard
            title="Mean FZ0 by model (lower is better)"
            caption="The joint VaR/ES score over the live days. Amber = machine learning, blue = everything else. When no model has been breached, this mostly rewards VaR and ES levels that stayed tight without being crossed."
          >
            <LiveFz0Bars rows={rows} info={info} />
          </ChartCard>

          <DataTable columns={cols} rows={rows} keyField="model" filterKey="model" />
          <p className="text-xs text-muted">
            MS-GARCH is not shown: its walk-forward runs in R and is not extended live. The frozen
            evaluation tables (Model Comparison) are unchanged.
          </p>
        </>
      )}
    </div>
  );
}
