"use client";

import { fmtConfidence, fmtPct } from "@/lib/format";
import { useBacktests, useForecast, useModelsComparison } from "@/lib/hooks";

/** Decomposes the current live VaR into three pieces a reader can check
 * independently: the window's own flat quantile (Historical Simulation,
 * no model), how much the chosen model shifts that, and where the model
 * lands in the FZ0 tournament against the other 19. Every number here comes
 * from endpoints the rest of the app already calls -- this just narrates
 * the gap between "the window was this risky" and "the model says this". */
export function WhyThisVar({
  asset,
  alpha,
  model,
}: {
  asset: string | null;
  alpha: number;
  model: string | null;
}) {
  const { data: forecast } = useForecast(asset, alpha, model);
  const { data: hs } = useBacktests(asset, "HS", alpha, 1);
  const { data: comparison } = useModelsComparison(asset, alpha);

  const modelVar = forecast?.var ?? null;
  const baseVar = hs && hs.length > 0 ? hs[hs.length - 1].var : null;
  const shift = modelVar != null && baseVar != null ? modelVar - baseVar : null;

  const row = comparison?.find((r) => r.model === model) ?? null;
  const best = comparison?.find((r) => r.is_best) ?? null;
  const gap = row && best ? row.fz0_mean - best.fz0_mean : null;
  const nModels = comparison?.length ?? null;

  if (!asset || !model || modelVar == null) return null;

  const maxAbs = Math.max(Math.abs(baseVar ?? modelVar), Math.abs(modelVar), 1e-6);
  const barPct = (v: number) => `${Math.min(100, (Math.abs(v) / maxAbs) * 100)}%`;

  return (
    <div className="card p-4">
      <div className="mb-1 text-sm font-medium text-text">
        Why {fmtPct(Math.abs(modelVar), 2)}? Decomposing {model}&apos;s {fmtConfidence(1 - alpha)} VaR for{" "}
        {asset}
      </div>
      <p className="mb-3 text-xs leading-snug text-muted">
        Three independent checks: what the raw window alone implies with no model at all, how much{" "}
        {model} moves that, and whether the move is one a 90% Model Confidence Set would actually back.
        {forecast?.source === "live_refit" && (
          <> Line 1 is HS&apos;s last frozen backtest row; line 2 is today&apos;s live re-fit, so a small gap
          can just be one extra day of data.</>
        )}
      </p>

      <div className="space-y-2.5">
        <div>
          <div className="mb-1 flex items-baseline justify-between text-xs">
            <span className="text-muted">1. Window base level (Historical Simulation, no model)</span>
            <span className="tabular-nums text-text">{baseVar != null ? fmtPct(baseVar, 2) : "n/a"}</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-panel-2">
            {baseVar != null && (
              <div className="h-full rounded-full bg-muted" style={{ width: barPct(baseVar) }} />
            )}
          </div>
        </div>

        <div>
          <div className="mb-1 flex items-baseline justify-between text-xs">
            <span className="text-muted">
              2. {model}&apos;s conditioning ({shift != null && shift < 0 ? "more" : "less"} severe than the
              flat window)
            </span>
            <span className={`tabular-nums ${shift != null && shift < 0 ? "text-red" : "text-green"}`}>
              {shift != null ? `${shift < 0 ? "" : "+"}${fmtPct(shift, 2)}` : "n/a"}
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-panel-2">
            <div
              className={`h-full rounded-full ${shift != null && shift < 0 ? "bg-red" : "bg-green"}`}
              style={{ width: barPct(modelVar) }}
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-grid pt-2.5 text-xs">
          <span className="text-muted">3. Tournament standing:</span>
          <span className="text-text">
            rank {row ? `${row.fz0_rank}/${nModels}` : "n/a"} on FZ0
          </span>
          <span className={row?.in_mcs ? "text-green" : "text-red"}>
            {row ? (row.in_mcs ? "inside the 90% MCS" : "outside the 90% MCS") : "n/a"}
          </span>
          {gap != null && (
            <span className="text-muted">
              {gap <= 1e-9 ? "is the top-ranked model" : `${gap.toFixed(4)} worse than the best (lower is better)`}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
