"use client";

import { useState } from "react";
import { Column, DataTable } from "@/components/DataTable";
import { Fz0BarChart } from "@/components/Fz0BarChart";
import { fmtConfidence } from "@/lib/format";
import { CoverageRow, EsTestRow, Fz0Row, GwCpaRow } from "@/lib/api";
import { useConfig, useCoverage, useEsTests, useGwCpa, useModelsComparison } from "@/lib/hooks";

export default function ModelComparisonPage() {
  const { data: config } = useConfig();
  const assets = config?.assets ?? [];
  const alphas = config?.alphas ?? [];

  const [asset, setAsset] = useState<string | null>(null);
  const [alpha, setAlpha] = useState<number | null>(null);
  const effAsset = asset ?? assets[0] ?? null;
  const effAlpha = alpha ?? alphas[0] ?? 0.025;

  const { data: comparison } = useModelsComparison(effAsset, effAlpha);
  const { data: coverage } = useCoverage(effAsset, effAlpha);
  const { data: esTests } = useEsTests(effAsset, effAlpha);
  const { data: gwCpa } = useGwCpa(effAsset, effAlpha);

  const fz0Cols: Column<Fz0Row>[] = [
    { key: "fz0_rank", label: "rank" },
    { key: "model", label: "model" },
    { key: "fz0_mean", label: "FZ0" },
    { key: "in_mcs", label: "in MCS" },
    { key: "mcs_p", label: "MCS p" },
    { key: "dm_vs_best_p", label: "DM p (vs best)" },
    { key: "n_degenerate", label: "degenerate days" },
  ];

  const coverageCols: Column<CoverageRow>[] = [
    { key: "model", label: "model" },
    { key: "hit_rate", label: "hit rate" },
    { key: "kupiec_p", label: "Kupiec p", pShade: true },
    { key: "chr_cc_p", label: "Christoffersen CC p", pShade: true },
    { key: "dq_p", label: "DQ p", pShade: true },
    { key: "basel_zone", label: "Basel zone" },
    { key: "passes_all", label: "passes all" },
  ];

  const esCols: Column<EsTestRow>[] = [
    { key: "model", label: "model" },
    { key: "n_breach", label: "breaches" },
    { key: "z1", label: "Z1" },
    { key: "z1_p_approx", label: "Z1 p", pShade: true },
    { key: "z2", label: "Z2" },
    { key: "z2_p_approx", label: "Z2 p", pShade: true },
    { key: "es_reject_approx", label: "ES rejected" },
  ];

  const gwCols: Column<GwCpaRow>[] = [
    { key: "model_a", label: "model A" },
    { key: "model_b", label: "model B" },
    { key: "mean_fz0_gap", label: "mean FZ0 gap" },
    { key: "gw_stat", label: "GW stat" },
    { key: "gw_p", label: "GW p", pShade: true },
    { key: "best_edge_vs_rv", label: "edge vs RV" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Model Comparison</h1>
        <p className="mt-1 text-sm text-muted">
          FZ0 loss ranking + Model Confidence Set (90%), coverage tests,
          Acerbi-Szekely ES tests.
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
          <select
            className="select"
            value={effAlpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
          >
            {alphas.map((a) => (
              <option key={a} value={a}>
                {fmtConfidence(a)}
              </option>
            ))}
          </select>
        </Field>
      </div>

      {comparison && comparison.length > 0 && (
        <Fz0BarChart
          data={[...comparison].sort((a, b) => a.fz0_rank - b.fz0_rank)}
          title={`${effAsset} @ ${fmtConfidence(effAlpha)} — green = in the 90% MCS`}
        />
      )}
      <DataTable columns={fz0Cols} rows={comparison ?? []} keyField="model" />

      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <h2 className="mb-2 text-lg font-medium">Coverage tests</h2>
          <DataTable columns={coverageCols} rows={coverage ?? []} keyField="model" />
          <p className="mt-2 text-xs text-muted">
            Nominal miss rate at this alpha: {fmtConfidence(1 - effAlpha)}. p &lt;
            0.05 rejects correct coverage.
          </p>
        </div>
        <div>
          <h2 className="mb-2 text-lg font-medium">Expected Shortfall tests (Acerbi-Szekely)</h2>
          <DataTable columns={esCols} rows={esTests ?? []} keyField="model" />
          <p className="mt-2 text-xs text-muted">
            Asymptotic-normal p-values (no per-day predictive draws available
            for a simulated null).
          </p>
        </div>
      </div>

      <div>
        <h2 className="mb-2 text-lg font-medium">
          Giacomini-White conditional predictive ability (regime-conditioned)
        </h2>
        <DataTable
          columns={gwCols}
          rows={gwCpa ?? []}
          keyField={(row) => `${row.model_a}-${row.model_b}`}
        />
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-muted">
      {label}
      {children}
    </label>
  );
}
