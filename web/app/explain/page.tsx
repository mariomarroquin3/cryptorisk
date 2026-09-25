"use client";

import { Suspense } from "react";
import {
  BacktestBand,
  ChartCard,
  CumulativeViolations,
  FamilyRankChart,
  HitRateBars,
  PitHistogram,
  RfConditioningShift,
  RfEssChart,
  RfImportanceBars,
  RfImportanceOverTime,
  RfInputsChart,
  RollingHitRate,
} from "@/components/ExplainCharts";
import { Field } from "@/components/Field";
import { LstmFeatureShare, LstmImportanceHeatmap, LstmLagProfile } from "@/components/LstmCharts";
import { RfPdpChart } from "@/components/RfPdpChart";
import { VarChangePanel } from "@/components/VarChangePanel";
import { ChartSkeleton } from "@/components/Skeleton";
import { fmtConfidence } from "@/lib/format";
import {
  useBacktests,
  useConfig,
  useCoverage,
  useLstmExplain,
  useModels,
  useModelsComparison,
  useModelsInfo,
  useRfExplain,
  useRfPdp,
} from "@/lib/hooks";
import { useQueryParam } from "@/lib/useQueryParam";

const HORIZONS = [250, 500, 1000, 2700];

export default function ExplainPage() {
  return (
    <Suspense>
      <ExplainPageInner />
    </Suspense>
  );
}

function ExplainPageInner() {
  const { data: config } = useConfig();
  const { data: models } = useModels();
  const { data: modelsInfo } = useModelsInfo();
  const assets = config?.assets ?? [];
  const alphas = config?.alphas ?? [];

  const [asset, setAsset] = useQueryParam("asset");
  const [alphaStr, setAlphaStr] = useQueryParam("alpha");
  const [modelQ, setModel] = useQueryParam("model");
  const [daysQ, setDays] = useQueryParam("days");
  const effAsset = asset ?? assets[0] ?? null;
  const effAlpha = alphaStr != null ? Number(alphaStr) : (alphas[0] ?? 0.025);
  const effModel = modelQ ?? (models?.includes("RF-QR") ? "RF-QR" : (models?.[0] ?? null));
  const days = daysQ != null ? Number(daysQ) : 1000;

  const { data: bt } = useBacktests(effAsset, effModel, effAlpha, days);
  const { data: coverage } = useCoverage(effAsset, effAlpha);
  const { data: comparison } = useModelsComparison(effAsset, effAlpha);
  const { data: rf } = useRfExplain(effAsset);
  const { data: lstm } = useLstmExplain(effAsset);

  const rfFeatures = [...new Set((rf?.importance ?? []).map((r) => r.feature))];
  const [pdpFeatureQ, setPdpFeature] = useQueryParam("pdpFeature");
  const pdpFeature = pdpFeatureQ ?? (rfFeatures.includes("r2_w") ? "r2_w" : (rfFeatures[0] ?? null));
  const { data: pdp, error: pdpError } = useRfPdp(effAsset, pdpFeature, effAlpha);

  const ready = bt !== undefined;
  const hasBt = ready && bt.length > 0;
  const nViol = hasBt ? bt.filter((r) => r.violation).length : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Explainability</h1>
        <p className="mt-1 text-sm text-muted">
          Why a model&apos;s numbers look the way they do: how its VaR tracks realized returns, where
          it is calibrated or not, how machine learning compares with structure, and what the
          random forest actually relies on.
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
        <Field label="Model">
          <select className="select" value={effModel ?? ""} onChange={(e) => setModel(e.target.value)}>
            {(models ?? []).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Out-of-sample days shown">
          <select className="select" value={days} onChange={(e) => setDays(e.target.value)}>
            {HORIZONS.map((d) => (
              <option key={d} value={d}>
                last {d >= 2700 ? "all" : d}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <h2 className="text-lg font-medium">1. Is {effModel} calibrated?</h2>
      {!ready ? (
        <ChartSkeleton height={320} />
      ) : !hasBt ? (
        <div className="text-sm text-muted">No backtest rows for this selection.</div>
      ) : (
        <>
          <ChartCard
            title={`Realized return vs ${effModel} VaR / ES  ·  ${nViol} violations in ${bt.length} days (expected ${(bt.length * effAlpha).toFixed(1)})`}
            caption="Each red dot is a day the loss exceeded VaR. Violations should be rare, unclustered, and roughly alpha of all days; ES sits below VaR as the average loss given a breach."
          >
            <BacktestBand rows={bt} />
          </ChartCard>
          <div className="grid gap-6 md:grid-cols-2">
            <ChartCard
              title="Cumulative violations vs expected"
              caption="A line hugging the expected path (inside the 95% band) means correct unconditional coverage; drifting above means the model is too optimistic."
            >
              <CumulativeViolations rows={bt} alpha={effAlpha} />
            </ChartCard>
            <ChartCard
              title="Rolling 250-day hit rate"
              caption="Calibration over time: spikes are stress episodes where the model under-reacted, dips are stretches where it was too conservative."
            >
              <RollingHitRate rows={bt} alpha={effAlpha} />
            </ChartCard>
          </div>
          <ChartCard
            title="PIT histogram (density calibration)"
            caption="The model's predicted CDF evaluated at each realized return should be uniform. Bars outside the band (red) show where the predicted distribution is wrong: U-shape = tails too thin, hump = tails too fat, slope = biased."
          >
            <PitHistogram rows={bt} />
          </ChartCard>
        </>
      )}

      <h2 className="text-lg font-medium">2. How does it compare?</h2>
      <div className="grid gap-6 md:grid-cols-2">
        <ChartCard
          title={`Hit rate by model @ ${fmtConfidence(effAlpha)}`}
          caption="Dashed green line is the nominal rate. Green bars pass every coverage test, red bars fail at least one; the chosen model is outlined."
        >
          {coverage === undefined ? (
            <ChartSkeleton height={340} />
          ) : (
            <HitRateBars rows={coverage} alpha={effAlpha} highlight={effModel} />
          )}
        </ChartCard>
        <ChartCard
          title="Method family: does structure beat machine learning?"
          caption="Mean and best FZ0 rank per family (lower is better). Machine learning is highlighted; the question is whether learned models close the gap to hand-built volatility structure."
        >
          {comparison === undefined ? (
            <ChartSkeleton height={320} />
          ) : (
            <FamilyRankChart rows={comparison} info={modelsInfo} />
          )}
        </ChartCard>
      </div>

      <h2 className="text-lg font-medium">3. What does the random forest (RF-QR) use?</h2>
      {rf === undefined ? (
        <ChartSkeleton height={300} />
      ) : rf.importance.length === 0 ? (
        <div className="text-sm text-muted">
          No RF-QR explainability output yet (run <code>make explain</code>).
        </div>
      ) : (
        <>
          <div className="grid gap-6 md:grid-cols-2">
            <ChartCard
              title="Average feature importance"
              caption="Share of the forest's split gain attributed to each input, averaged over the out-of-sample refits. Importance is spread fairly evenly (roughly 10-16% per input); only the down-day leverage term is clearly lower (about 4%). No single input dominates, which is what a weak signal in noisy returns looks like."
            >
              <RfImportanceBars rows={rf.importance} />
            </ChartCard>
            <ChartCard
              title="Today's forecast inputs (z-scores)"
              caption="The latest forecast row against its own 500-day window. Red = unusually high, blue = unusually low; extreme values are where the forest extrapolates least reliably."
            >
              <RfInputsChart rows={rf.inputs} />
            </ChartCard>
          </div>
          <ChartCard
            title="Feature reliance over time"
            caption="How the importance mix shifts across refits. A stable mix means a stable relationship; sharp shifts flag regime changes in what predicts tail risk."
          >
            <RfImportanceOverTime rows={rf.importance} />
          </ChartCard>
          <div className="grid gap-6 md:grid-cols-2">
            <ChartCard
              title="Effective sample size behind each forecast"
              caption="The forest reweights past days by similarity to today; 1/sum(w^2) counts how many days effectively support the tail quantile. Low values mean a noisy, thinly-supported estimate."
            >
              <RfEssChart rows={rf.diagnostics} />
            </ChartCard>
            <ChartCard
              title="What conditioning adds: forest VaR vs plain window VaR (97.5%, fixed)"
              caption="Gap between the two lines is the value the features add over simply taking the window's empirical quantile (Historical Simulation)."
            >
              <RfConditioningShift rows={rf.diagnostics} />
            </ChartCard>
          </div>
          <ChartCard
            title={`Partial dependence: how today's VaR moves with one feature`}
            caption="Sweeps the selected input across its historical range, everything else pinned at today's actual values, and re-reads the forest's VaR at each point -- a live 'what if' for today's forecast, not a historical average. Not necessarily monotone: the forest routes through discrete leaves, so the curve can step rather than glide."
          >
            <div className="mb-3">
              <Field label="Feature">
                <select
                  className="select"
                  value={pdpFeature ?? ""}
                  onChange={(e) => setPdpFeature(e.target.value)}
                >
                  {rfFeatures.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            {pdpError ? (
              <div className="text-sm text-muted">No partial dependence available for this selection.</div>
            ) : pdp === undefined ? (
              <ChartSkeleton height={240} />
            ) : (
              <RfPdpChart pdp={pdp} />
            )}
          </ChartCard>
        </>
      )}

      <h2 className="text-lg font-medium">4. What does the LSTM (LSTM-Vol) use?</h2>
      {lstm === undefined ? (
        <ChartSkeleton height={200} />
      ) : lstm.length === 0 ? (
        <div className="text-sm text-muted">
          No LSTM-Vol explainability output yet (run <code>make explain</code>).
        </div>
      ) : (
        <>
          <ChartCard
            title="Permutation importance by input cell"
            caption="Each cell is one number in the 20-day input window (a feature on a given day). Its brightness is how much the fitted network's training loss rises when that column is shuffled across sequences, averaged over the out-of-sample refits. Bright = the network relies on it."
          >
            <LstmImportanceHeatmap rows={lstm} />
          </ChartCard>
          <div className="grid gap-6 md:grid-cols-2">
            <ChartCard
              title="Share by input channel"
              caption="The raw signed return carries almost all of the importance. The squared-return channels sit near 1e-4 in raw units, so a network trained for 80 steps barely uses them: the LSTM is rebuilding a volatility signal from signed returns rather than reading it directly."
            >
              <LstmFeatureShare rows={lstm} />
            </ChartCard>
            <ChartCard
              title="How far back it looks"
              caption="Total importance by lag. It concentrates in the most recent days and fades within a week or so, so the 20-day window is longer than the memory the network actually uses."
            >
              <LstmLagProfile rows={lstm} />
            </ChartCard>
          </div>
        </>
      )}

      <h2 className="text-lg font-medium">5. Why did each model&apos;s VaR change since yesterday?</h2>
      <VarChangePanel asset={effAsset} alpha={effAlpha} />
    </div>
  );
}
