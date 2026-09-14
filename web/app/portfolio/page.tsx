"use client";

import { Suspense } from "react";
import { Column, DataTable } from "@/components/DataTable";
import { Field } from "@/components/Field";
import { Fz0BarChart } from "@/components/Fz0BarChart";
import { MetricCard } from "@/components/MetricCard";
import { fmtConfidence, fmtUsd } from "@/lib/format";
import { PortfolioEvalRow } from "@/lib/api";
import { useConfig, usePortfolioComposition, usePortfolioEval } from "@/lib/hooks";
import { useQueryParam } from "@/lib/useQueryParam";

export default function PortfolioPage() {
  return (
    <Suspense>
      <PortfolioPageInner />
    </Suspense>
  );
}

function PortfolioPageInner() {
  const { data: config } = useConfig();
  const alphas = config?.alphas ?? [];
  const port = config?.portfolio;

  const [alphaStr, setAlphaStr] = useQueryParam("alpha");
  const effAlpha = alphaStr != null ? Number(alphaStr) : (alphas[0] ?? 0.025);

  const { data: evalRows } = usePortfolioEval(effAlpha);
  const { data: composition } = usePortfolioComposition();

  const sorted = [...(evalRows ?? [])].sort((a, b) => a.fz0_rank - b.fz0_rank);
  const copulas = sorted.filter((r) => r.model.toLowerCase().includes("copula"));
  const directs = sorted.filter((r) => r.model.startsWith("Direct-"));

  const cols: Column<PortfolioEvalRow>[] = [
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
      key: "hit_rate",
      label: "hit rate",
      help: "Fraction of days the realized basket loss exceeded VaR. Should sit close to the nominal miss rate for correct coverage.",
    },
    {
      key: "kupiec_p",
      label: "Kupiec p",
      pShade: true,
      help: "Unconditional coverage test. p < 0.05 rejects: the violation rate is significantly off target.",
    },
    {
      key: "z2",
      label: "Z2",
      help: "Acerbi-Szekely ES test statistic. Large negative values mean ES understates the tail (too optimistic).",
    },
    {
      key: "basel_zone",
      label: "Basel zone",
      help: "Traffic-light zone from the 250-day exception count: green (<=4), amber (5-9), red (>=10).",
    },
    {
      key: "passes_all",
      label: "passes all",
      help: "Passes every coverage test in this row at the 5% significance level.",
    },
  ];

  const smallCols: Column<PortfolioEvalRow>[] = [
    { key: "fz0_rank", label: "rank" },
    { key: "model", label: "model" },
    { key: "fz0_mean", label: "FZ0" },
    { key: "hit_rate", label: "hit rate" },
  ];

  const basket = port?.assets ?? [];
  const weights = port?.weights ?? {};
  const prices = composition?.prices ?? {};
  const allPricesLoaded = basket.length > 0 && basket.every((a) => prices[a]);
  const basketChange = allPricesLoaded
    ? basket.reduce((acc, a) => acc + (weights[a] ?? 0) * (prices[a]?.change_pct ?? 0), 0)
    : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Portfolio</h1>
        <p className="mt-1 text-sm text-muted">
          Equal-weight basket ({basket.map((a) => `${a} ${((weights[a] ?? 0) * 100).toFixed(0)}%`).join(", ")}),
          copula tail vs. independence, evaluated with the same FZ0/MCS
          battery.
        </p>
      </div>

      <Field label="Confidence">
        <select className="select" value={effAlpha} onChange={(e) => setAlphaStr(e.target.value)}>
          {alphas.map((a) => (
            <option key={a} value={a}>
              {fmtConfidence(a)}
            </option>
          ))}
        </select>
      </Field>

      {sorted.length > 0 && (
        <Fz0BarChart data={sorted} title={`Basket @ ${fmtConfidence(effAlpha)}`} />
      )}
      <DataTable columns={cols} rows={sorted} keyField="model" />

      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <h2 className="mb-2 text-lg font-medium">Copula family comparison</h2>
          <DataTable columns={smallCols} rows={copulas} keyField="model" />
          <p className="mt-2 text-xs text-muted">
            Copula-independence ignoring tail dependence typically
            over-breaches badly at 99%.
          </p>
        </div>
        <div>
          <h2 className="mb-2 text-lg font-medium">Direct univariate models on the basket return</h2>
          <DataTable columns={smallCols} rows={directs} keyField="model" />
        </div>
      </div>

      <div>
        <h2 className="mb-2 text-lg font-medium">Live basket composition</h2>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {basket.map((a) => {
            const p = prices[a];
            return (
              <MetricCard
                key={a}
                label={`${a} (${((weights[a] ?? 0) * 100).toFixed(0)}%)`}
                value={p ? fmtUsd(p.price, 2) : "n/a"}
                delta={p ? `${p.change_pct >= 0 ? "+" : ""}${p.change_pct.toFixed(2)}%` : undefined}
                deltaColor={p && p.change_pct >= 0 ? "text-green" : "text-red"}
              />
            );
          })}
        </div>
        {basketChange != null && (
          <div className="mt-4 max-w-xs">
            <MetricCard
              label="Basket 24h change (weighted)"
              value={`${basketChange >= 0 ? "+" : ""}${basketChange.toFixed(2)}%`}
              deltaColor={basketChange >= 0 ? "text-green" : "text-red"}
            />
          </div>
        )}
      </div>
    </div>
  );
}
