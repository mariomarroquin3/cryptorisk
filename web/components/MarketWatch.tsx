"use client";

import { Sparkline } from "@/components/Sparkline";
import { fmtConfidence, fmtPct, fmtUsd } from "@/lib/format";
import { useBacktests, useModelsComparison, usePrice, usePrices } from "@/lib/hooks";

function Row({
  asset,
  alpha,
  selected,
  onSelect,
}: {
  asset: string;
  alpha: number;
  selected: boolean;
  onSelect: (a: string) => void;
}) {
  const { data: price } = usePrice(asset);
  const { data: hist } = usePrices(asset, 30);
  const { data: comparison } = useModelsComparison(asset, alpha);
  const best = comparison?.find((r) => r.is_best)?.model ?? null;
  const { data: bt } = useBacktests(asset, best, alpha, 1, true);
  const last = bt && bt.length > 0 ? bt[bt.length - 1] : null;
  const change = price?.change_pct;
  const spot = price?.price ?? (hist && hist.length > 0 ? hist[hist.length - 1].close : null);
  // The frozen forecast for day D is made from D-1's close, so anchor the
  // price level there. Applying it to today's live spot would mix two dates.
  const anchor = (() => {
    if (!hist || !last) return null;
    const i = hist.findIndex((h) => h.date.slice(0, 10) === last.date.slice(0, 10));
    return i > 0 ? hist[i - 1].close : null;
  })();
  const varPrice = anchor != null && last ? anchor * Math.exp(last.var) : null;
  return (
    <tr
      onClick={() => onSelect(asset)}
      className={`cursor-pointer border-b border-grid/60 transition-colors last:border-0 hover:bg-panel-2/60 ${
        selected ? "bg-panel-2" : ""
      }`}
    >
      <td className="px-3 py-2 font-semibold">{asset}</td>
      <td className="px-3 py-2 text-right tabular-nums">{spot != null ? fmtUsd(spot, 2) : "…"}</td>
      <td className={`px-3 py-2 text-right tabular-nums ${(change ?? 0) >= 0 ? "text-green" : "text-red"}`}>
        {change != null ? `${change >= 0 ? "+" : ""}${change.toFixed(2)}%` : "n/a"}
      </td>
      <td className="px-3 py-2 text-center">{hist ? <Sparkline values={hist.map((h) => h.close)} /> : "…"}</td>
      <td className="px-3 py-2 text-right tabular-nums text-red">{last ? fmtPct(last.var, 2) : "n/a"}</td>
      <td className="px-3 py-2 text-right tabular-nums text-red">{last ? fmtPct(last.es, 2) : "n/a"}</td>
      <td className="px-3 py-2 text-right tabular-nums text-muted">{varPrice != null ? fmtUsd(varPrice, 0) : "n/a"}</td>
      <td className="px-3 py-2 text-xs text-muted">{best ?? "n/a"}</td>
    </tr>
  );
}

/** First thing on the Overview: every study asset at a glance, with the
 * best model's latest one-day VaR / ES. Click a row to load it below. */
export function MarketWatch({
  assets,
  alpha,
  selected,
  onSelect,
}: {
  assets: string[];
  alpha: number;
  selected: string | null;
  onSelect: (a: string) => void;
}) {
  const head = ["asset", "price", "24h", "30d", `VaR ${fmtConfidence(alpha)}`, `ES ${fmtConfidence(alpha)}`, "VaR price", "best model"];
  return (
    <div className="card overflow-hidden">
      <div className="border-b border-grid px-4 py-2.5 text-sm font-medium text-text">
        Market watch
        <span className="ml-2 text-xs font-normal text-muted">
          latest frozen out-of-sample forecast of the best-ranked model, per asset
        </span>
      </div>
      <div className="scrollbar-thin overflow-x-auto">
        <table className="w-full min-w-max text-left text-sm">
          <thead>
            <tr className="border-b border-grid bg-panel-2 text-muted">
              {head.map((h, i) => (
                <th
                  key={h}
                  className={`px-3 py-2 text-xs font-semibold uppercase tracking-wide ${i === 0 || i === 7 ? "text-left" : i === 3 ? "text-center" : "text-right"}`}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {assets.map((a) => (
              <Row key={a} asset={a} alpha={alpha} selected={selected === a} onSelect={onSelect} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
