"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import {
  apiGet,
  BacktestRow,
  CapitalRow,
  Config,
  CoverageRow,
  EsTestRow,
  EstimationRiskRow,
  ForecastResponse,
  Fz0Row,
  GwCpaRow,
  HedgeRow,
  LimitsRow,
  LiveTrackRecord,
  LstmImportanceRow,
  ModelInfo,
  ModelLatestRow,
  PortfolioComposition,
  PortfolioEvalRow,
  PriceHistoryRow,
  PriceResponse,
  RegimesResponse,
  RfExplain,
  RfPdp,
  WhatIfRow,
} from "./api";

const fetcher = <T,>(path: string) => apiGet<T>(path);

export function useConfig() {
  return useSWR<Config>("/config", fetcher);
}

export function useModels() {
  return useSWR<string[]>("/models", fetcher);
}

export function useModelsInfo() {
  return useSWR<Record<string, ModelInfo>>("/models/info", fetcher);
}

export function useModelsLatest(asset: string | null, alpha: number) {
  return useSWR<ModelLatestRow[]>(
    asset ? `/models/latest?asset=${asset}&alpha=${alpha}` : null,
    fetcher,
  );
}

export function usePrice(asset: string | null) {
  return useSWR<PriceResponse>(asset ? `/price/${asset}` : null, fetcher, {
    refreshInterval: 15_000,
    revalidateOnFocus: true,
  });
}

export function usePrices(asset: string | null, limit = 200) {
  return useSWR<PriceHistoryRow[]>(
    asset ? `/prices/${asset}?limit=${limit}` : null,
    fetcher,
  );
}

export function useForecast(asset: string | null, alpha: number, model?: string | null) {
  const q = new URLSearchParams({ alpha: String(alpha) });
  if (model) q.set("model", model);
  return useSWR<ForecastResponse>(
    asset ? `/forecast/${asset}?${q.toString()}` : null,
    fetcher,
    { refreshInterval: 60_000 },
  );
}

export function useModelsComparison(asset: string | null, alpha: number) {
  return useSWR<Fz0Row[]>(
    asset ? `/models/comparison?asset=${asset}&alpha=${alpha}` : null,
    fetcher,
  );
}

export function useCoverage(asset: string | null, alpha: number) {
  return useSWR<CoverageRow[]>(
    asset ? `/coverage?asset=${asset}&alpha=${alpha}` : null,
    fetcher,
  );
}

export function useEsTests(asset: string | null, alpha: number) {
  return useSWR<EsTestRow[]>(
    asset ? `/es-tests?asset=${asset}&alpha=${alpha}` : null,
    fetcher,
  );
}

export function useGwCpa(asset: string | null, alpha: number) {
  return useSWR<GwCpaRow[]>(
    asset ? `/gw-cpa?asset=${asset}&alpha=${alpha}` : null,
    fetcher,
  );
}

export function useBacktests(
  asset: string | null,
  model: string | null,
  alpha: number,
  limit = 180,
  live = false,
) {
  return useSWR<BacktestRow[]>(
    asset && model
      ? `/backtests?asset=${asset}&model=${encodeURIComponent(model)}&alpha=${alpha}&limit=${limit}${live ? "&live=true" : ""}`
      : null,
    fetcher,
  );
}

export function usePortfolioEval(alpha: number) {
  return useSWR<PortfolioEvalRow[]>(`/portfolio/eval?alpha=${alpha}`, fetcher);
}

export function usePortfolioNarrative(alpha: number) {
  return useSWR<string[]>(`/portfolio/narrative?alpha=${alpha}`, fetcher);
}

export function usePortfolioComposition() {
  return useSWR<PortfolioComposition>("/portfolio/composition", fetcher, {
    refreshInterval: 15_000,
  });
}

export function useCapital(asset: string | null) {
  return useSWR<CapitalRow[]>(asset ? `/capital?asset=${asset}` : null, fetcher);
}

export function useEstimationRisk(asset: string | null) {
  return useSWR<EstimationRiskRow[]>(
    asset ? `/estimation-risk?asset=${asset}` : null,
    fetcher,
  );
}

export function useLimits(asset: string | null) {
  return useSWR<LimitsRow[]>(asset ? `/limits?asset=${asset}` : null, fetcher);
}

export function useHedge(asset: string | null) {
  return useSWR<HedgeRow[]>(asset ? `/hedge?asset=${asset}` : null, fetcher);
}

export function useLstmExplain(asset: string | null) {
  return useSWR<LstmImportanceRow[]>(asset ? `/explain/lstm?asset=${asset}` : null, fetcher);
}

export function useRfExplain(asset: string | null) {
  return useSWR<RfExplain>(asset ? `/explain/rf?asset=${asset}` : null, fetcher);
}

export function useRfPdp(asset: string | null, feature: string | null, alpha: number) {
  return useSWR<RfPdp>(
    asset && feature ? `/explain/rf/pdp?asset=${asset}&feature=${feature}&alpha=${alpha}` : null,
    fetcher,
  );
}

export function useRegimes(asset: string | null, limit = 2000) {
  return useSWR<RegimesResponse>(
    asset ? `/regimes/${asset}?limit=${limit}` : null,
    fetcher,
  );
}

export function useLiveTrackRecord(asset: string | null, alpha: number) {
  return useSWR<LiveTrackRecord>(
    asset ? `/live/track-record?asset=${asset}&alpha=${alpha}` : null,
    fetcher,
  );
}

/** Slow re-fits go last so the fast models fill in first. */
const SLOW_LAST = ["RF-QR", "Realized-SV", "LSTM-Vol"];

/** What-if VaR/ES for every model under one hypothetical next-day return.
 * One request per model, at most four in flight, results appear as they
 * arrive (`undefined` = pending, `null` = failed). */
export function useWhatIfAll(
  asset: string | null,
  models: string[],
  shock: number,
  alpha: number,
): Record<string, WhatIfRow | null | undefined> {
  type Rows = Record<string, WhatIfRow | null | undefined>;
  const key = models.join(",");
  const sig = `${asset}|${key}|${shock}|${alpha}`;
  // Results are tagged with the parameters they answer, so stale ones are
  // ignored when the inputs change (no reset-in-effect needed).
  const [state, setState] = useState<{ sig: string; rows: Rows }>({ sig: "", rows: {} });
  useEffect(() => {
    if (!asset || !key) return;
    let cancelled = false;
    const queue = key
      .split(",")
      .sort((a, b) => SLOW_LAST.indexOf(a) - SLOW_LAST.indexOf(b));
    const put = (m: string, r: WhatIfRow | null) =>
      setState((p) => ({ sig, rows: { ...(p.sig === sig ? p.rows : {}), [m]: r } }));
    const worker = async () => {
      while (queue.length > 0 && !cancelled) {
        const m = queue.shift() as string;
        try {
          const r = await apiGet<WhatIfRow>(
            `/whatif/${asset}?model=${encodeURIComponent(m)}&shock=${shock}&alpha=${alpha}`,
          );
          if (!cancelled) put(m, r);
        } catch {
          if (!cancelled) put(m, null);
        }
      }
    };
    void Promise.all(Array.from({ length: 4 }, worker));
    return () => {
      cancelled = true;
    };
  }, [asset, key, shock, alpha, sig]);
  return state.sig === sig ? state.rows : {};
}
