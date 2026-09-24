"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { ChartCard } from "@/components/ExplainCharts";
import { Column, DataTable } from "@/components/DataTable";
import { Field } from "@/components/Field";
import { ModelTip } from "@/components/ModelTip";
import { ChartSkeleton } from "@/components/Skeleton";
import { ReactionBars, toPoints, WhatIfPoint } from "@/components/WhatIfChart";
import { fmtConfidence, fmtPct, fmtUsd } from "@/lib/format";
import { useConfig, useModels, useModelsInfo, useWhatIfAll } from "@/lib/hooks";
import { useQueryParam } from "@/lib/useQueryParam";

const PRESETS = [-0.2, -0.1, -0.05, 0.05, 0.1];

export default function WhatIfPage() {
  return (
    <Suspense>
      <WhatIfInner />
    </Suspense>
  );
}

function WhatIfInner() {
  const { data: config } = useConfig();
  const { data: allModels } = useModels();
  const { data: info } = useModelsInfo();
  const assets = config?.assets ?? [];
  const alphas = config?.alphas ?? [];
  const [asset, setAsset] = useQueryParam("asset");
  const [alphaStr, setAlphaStr] = useQueryParam("alpha");
  const effAsset = asset ?? assets[0] ?? null;
  const effAlpha = alphaStr != null ? Number(alphaStr) : (alphas[0] ?? 0.025);

  // Slider moves freely; the (expensive) re-fits run once it settles.
  const [slider, setSlider] = useState(-0.08);
  const [shock, setShock] = useState(-0.08);
  useEffect(() => {
    const t = setTimeout(() => setShock(slider), 500);
    return () => clearTimeout(t);
  }, [slider]);

  // MS-GARCH is served from an offline R run, so it cannot be re-fitted live.
  const models = useMemo(() => (allModels ?? []).filter((m) => m !== "MS-GARCH"), [allModels]);
  const rows = useWhatIfAll(effAsset, models, shock, effAlpha);
  const points = useMemo(() => toPoints(rows), [rows]);
  const pending = models.filter((m) => rows[m] === undefined).length;
  const sample = Object.values(rows).find((r) => r) ?? null;

  const cols: Column<WhatIfPoint & { dpp: number }>[] = [
    { key: "model", label: "model", render: (r) => <ModelTip name={r.model} info={info?.[r.model]} /> },
    { key: "base", label: "VaR today", render: (r) => fmtPct(r.base, 2), help: "One-day VaR from the model re-fitted on the latest 500 days, before any hypothetical move." },
    { key: "shocked", label: "VaR after shock", render: (r) => fmtPct(r.shocked, 2), help: "One-day VaR for the day after, with the hypothetical return appended to the window." },
    { key: "dpp", label: "change (pp)", render: (r) => `${r.dpp >= 0 ? "+" : ""}${(r.dpp * 100).toFixed(2)}`, help: "Shocked minus baseline VaR, in percentage points. Negative = the model now sees more downside risk." },
    { key: "multiple", label: "multiple", render: (r) => `${r.multiple.toFixed(2)}x`, help: "Shocked VaR divided by today's VaR. 1.00x = no reaction." },
  ];
  const tableRows = points.map((p) => ({ ...p, dpp: p.shocked - p.base }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">What-if shock</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Pick a hypothetical return for the next day and see how each model would change its VaR for
          the day after. Every model is re-fitted live on the latest 500 days with that day appended,
          so the differences are pure model behaviour: how fast each one reacts to news.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-4">
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
        <Field label={`Next-day return: ${slider >= 0 ? "+" : ""}${(slider * 100).toFixed(0)}%`}>
          <input
            type="range"
            min={-0.25}
            max={0.25}
            step={0.01}
            value={slider}
            onChange={(e) => setSlider(Number(e.target.value))}
            className="w-64"
            aria-label="Hypothetical next-day return"
          />
        </Field>
        <div className="flex gap-2">
          {PRESETS.map((p) => (
            <button
              key={p}
              type="button"
              className="rounded border border-grid px-2 py-1 text-xs text-muted hover:border-amber hover:text-amber"
              onClick={() => setSlider(p)}
            >
              {p > 0 ? "+" : ""}
              {(p * 100).toFixed(0)}%
            </button>
          ))}
        </div>
      </div>

      {sample && (
        <div className="card p-4 text-sm">
          {effAsset} at {fmtUsd(sample.last_close, 0)} moving {(shock * 100).toFixed(0)}% would close at{" "}
          <span className="font-semibold">{fmtUsd(sample.shocked_close, 0)}</span>.{" "}
          {pending > 0 ? (
            <span className="text-muted">Re-fitting {pending} more models (the slow ones come last)...</span>
          ) : (
            <span className="text-muted">All {points.length} models re-fitted.</span>
          )}
        </div>
      )}

      {points.length === 0 ? (
        <ChartSkeleton height={340} />
      ) : (
        <>
          <ChartCard
            title="How much each model's VaR moves"
            caption="Shocked VaR as a multiple of today's VaR (dashed line = no reaction). Amber = machine learning, blue = the rest. Realized-measure models (HAR-RV, HARQ, GARCH-X) treat the shock day as a smooth move with no jump; the model that barely moves is the one that would be slowest to warn you."
          >
            <ReactionBars points={points} info={info} />
          </ChartCard>
          <DataTable columns={cols} rows={tableRows} keyField="model" filterKey="model" />
          <p className="text-xs text-muted">
            Assumption for the realized-measure models (HAR-RV, HARQ, Realized-GARCH/SV, GARCH-X,
            RF-QR): the intraday path of the hypothetical day is unknown, so its realized variance is
            taken as the squared shock, with no jump. Their reaction depends on that choice. MS-GARCH
            is not shown: its forecasts come from an offline R run. The hypothetical day is not
            added to any stored data, and none of this is part of the study&apos;s results.
          </p>
        </>
      )}
    </div>
  );
}
