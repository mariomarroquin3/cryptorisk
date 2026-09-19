"use client";

import { Sparkline } from "@/components/Sparkline";
import { fmtUsd } from "@/lib/format";
import { useConfig, usePrice, usePrices } from "@/lib/hooks";

function TickerItem({ asset, hasHistory }: { asset: string; hasHistory: boolean }) {
  const { data: price } = usePrice(asset);
  // Price history only exists for the study's assets (BTC/ETH), not the
  // portfolio basket's SOL/BNB.
  const { data: hist } = usePrices(hasHistory ? asset : null, 30);
  const change = price?.change_pct;
  const up = (change ?? 0) >= 0;
  return (
    <span className="inline-flex items-center gap-2 whitespace-nowrap px-5 text-xs tabular-nums">
      <span className="font-semibold text-text">{asset}</span>
      <span className="text-text">{price ? fmtUsd(price.price, price.price < 10 ? 3 : 2) : "…"}</span>
      {change != null && (
        <span className={up ? "text-green" : "text-red"}>
          {up ? "▲" : "▼"} {Math.abs(change).toFixed(2)}%
        </span>
      )}
      {hist && hist.length > 1 && <Sparkline values={hist.map((h) => h.close)} width={56} height={16} />}
    </span>
  );
}

/** Scrolling strip of live prices for the study assets and the portfolio
 * basket. The list is rendered twice so the CSS marquee loops seamlessly;
 * it pauses on hover and stops moving for users who prefer reduced motion. */
export function TickerTape() {
  const { data: config } = useConfig();
  if (!config) return null;
  const study = new Set(config.assets);
  const assets = [...new Set([...config.assets, ...(config.portfolio?.assets ?? [])])];
  if (assets.length === 0) return null;
  const items = assets.map((a) => <TickerItem key={a} asset={a} hasHistory={study.has(a)} />);
  return (
    <div className="ticker overflow-hidden border-b border-grid bg-panel py-1.5" aria-label="Live prices">
      <div className="ticker-track flex w-max">
        <div className="flex">{items}</div>
        <div className="flex" aria-hidden>
          {items}
        </div>
      </div>
    </div>
  );
}
