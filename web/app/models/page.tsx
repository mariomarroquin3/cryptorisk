"use client";

import { Suspense } from "react";
import { Column, DataTable } from "@/components/DataTable";
import { Field } from "@/components/Field";
import { Fz0BarChart } from "@/components/Fz0BarChart";
import { ChartSkeleton } from "@/components/Skeleton";
import { ZoneBadge } from "@/components/ZoneBadge";
import { fmtConfidence } from "@/lib/format";
import { CoverageRow, EsTestRow, Fz0Row, GwCpaRow } from "@/lib/api";
import { useConfig, useCoverage, useEsTests, useGwCpa, useModelsComparison } from "@/lib/hooks";
import { useQueryParam } from "@/lib/useQueryParam";

export default function ModelComparisonPage() {
  return (
    <Suspense>
      <ModelComparisonPageInner />
    </Suspense>
  );
}

function ModelComparisonPageInner() {
  const { data: config } = useConfig();
  const assets = config?.assets ?? [];
  const alphas = config?.alphas ?? [];

  const [asset, setAsset] = useQueryParam("asset");
  const [alphaStr, setAlphaStr] = useQueryParam("alpha");
  const effAsset = asset ?? assets[0] ?? null;
  const effAlpha = alphaStr != null ? Number(alphaStr) : (alphas[0] ?? 0.025);

  const { data: comparison } = useModelsComparison(effAsset, effAlpha);
  const { data: coverage } = useCoverage(effAsset, effAlpha);
  const { data: esTests } = useEsTests(effAsset, effAlpha);
  const { data: gwCpa } = useGwCpa(effAsset, effAlpha);

  const fz0Cols: Column<Fz0Row>[] = [
    { key: "fz0_rank", label: "rank" },
    { key: "model", label: "model" },
    {
      key: "fz0_mean",
      label: "FZ0",
      help: "Consistent joint VaR/ES scoring function (Fissler-Ziegel). Lower is better -- this is what ranks the models.",
    },
    {
      key: "in_mcs",
      label: "in MCS",
      help: "In the 90% Model Confidence Set: statistically indistinguishable from the best model at this confidence level.",
    },
    {
      key: "mcs_p",
      label: "MCS p",
      help: "Model Confidence Set p-value. Higher means more confidently in the top set; models below the threshold get eliminated.",
    },
    {
      key: "dm_vs_best_p",
      label: "DM p (vs best)",
      help: "Diebold-Mariano test p-value against the top-ranked model. p < 0.05 means this model is significantly worse.",
    },
    {
      key: "n_degenerate",
      label: "degenerate days",
      help: "Days this model produced an unusable forecast (ES >= 0 or VaR >= 0) and was excluded from scoring.",
    },
  ];

  const coverageCols: Column<CoverageRow>[] = [
    { key: "model", label: "model" },
    {
      key: "hit_rate",
      label: "hit rate",
      help: "Fraction of days the realized loss exceeded VaR. Should sit close to the nominal miss rate (e.g. 2.5%) for correct coverage.",
    },
    {
      key: "kupiec_p",
      label: "Kupiec p",
      pShade: true,
      help: "Unconditional coverage test. p < 0.05 rejects: the violation rate is significantly off target.",
    },
    {
      key: "chr_cc_p",
      label: "Christoffersen CC p",
      pShade: true,
      help: "Conditional coverage test: checks violations are the right size AND not clustered in time. p < 0.05 rejects.",
    },
    {
      key: "dq_p",
      label: "DQ p",
      pShade: true,
      help: "Dynamic Quantile test (Engle-Manganelli): checks violations aren't predictable from recent history. p < 0.05 rejects.",
    },
    {
      key: "basel_zone",
      label: "Basel zone",
      help: "Traffic-light zone from the 250-day exception count: green (<=4), amber (5-9), red (>=10) -- drives the capital multiplier.",
      render: (r) => <ZoneBadge zone={r.basel_zone} />,
    },
    {
      key: "passes_all",
      label: "passes all",
      help: "Passes every coverage test in this row at the 5% significance level.",
    },
  ];

  const esCols: Column<EsTestRow>[] = [
    { key: "model", label: "model" },
    { key: "n_breach", label: "breaches" },
    {
      key: "z1",
      label: "Z1",
      help: "Acerbi-Szekely ES test statistic (magnitude form). Large negative values mean ES understates the tail (too optimistic).",
    },
    { key: "z1_p_approx", label: "Z1 p", pShade: true, help: "p < 0.05 rejects: ES is significantly miscalibrated (Z1 statistic)." },
    {
      key: "z2",
      label: "Z2",
      help: "Acerbi-Szekely ES test statistic (ratio form). Large negative values mean ES understates the tail.",
    },
    { key: "z2_p_approx", label: "Z2 p", pShade: true, help: "p < 0.05 rejects: ES is significantly miscalibrated (Z2 statistic)." },
    {
      key: "es_reject_approx",
      label: "ES rejected",
      help: "Whether the Z1/Z2 tests reject correct ES calibration at the 5% level.",
    },
  ];

  const gwCols: Column<GwCpaRow>[] = [
    { key: "model_a", label: "model A" },
    { key: "model_b", label: "model B" },
    {
      key: "mean_fz0_gap",
      label: "mean FZ0 gap",
      help: "Average FZ0 loss difference between the two models (positive = model B scores better on average).",
    },
    {
      key: "gw_stat",
      label: "GW stat",
      help: "Giacomini-White conditional predictive ability test statistic.",
    },
    {
      key: "gw_p",
      label: "GW p",
      pShade: true,
      help: "p < 0.05 rejects equal predictive ability, conditioning on the recent state (not just an unconditional average).",
    },
    {
      key: "best_edge_vs_rv",
      label: "edge vs RV",
      help: "Which model wins head-to-head against the HAR-RV baseline in this test.",
    },
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
            onChange={(e) => setAlphaStr(e.target.value)}
          >
            {alphas.map((a) => (
              <option key={a} value={a}>
                {fmtConfidence(a)}
              </option>
            ))}
          </select>
        </Field>
      </div>

      {comparison === undefined ? (
        <ChartSkeleton height={220} />
      ) : (
        comparison.length > 0 && (
          <Fz0BarChart
            data={[...comparison].sort((a, b) => a.fz0_rank - b.fz0_rank)}
            title={`${effAsset} @ ${fmtConfidence(effAlpha)} — green = in the 90% MCS`}
          />
        )
      )}
      <DataTable
        columns={fz0Cols}
        rows={comparison ?? []}
        keyField="model"
        loading={comparison === undefined}
      />

      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <h2 className="mb-2 text-lg font-medium">Coverage tests</h2>
          <DataTable
            columns={coverageCols}
            rows={coverage ?? []}
            keyField="model"
            loading={coverage === undefined}
          />
          <p className="mt-2 text-xs text-muted">
            Nominal miss rate at this alpha: {fmtConfidence(1 - effAlpha)}. p &lt;
            0.05 rejects correct coverage.
          </p>
        </div>
        <div>
          <h2 className="mb-2 text-lg font-medium">Expected Shortfall tests (Acerbi-Szekely)</h2>
          <DataTable
            columns={esCols}
            rows={esTests ?? []}
            keyField="model"
            loading={esTests === undefined}
          />
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
          loading={gwCpa === undefined}
        />
      </div>
    </div>
  );
}
