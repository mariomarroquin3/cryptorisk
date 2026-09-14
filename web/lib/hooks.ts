"use client";

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
  PortfolioComposition,
  PortfolioEvalRow,
  PriceHistoryRow,
  PriceResponse,
  RegimesResponse,
} from "./api";

const fetcher = <T,>(path: string) => apiGet<T>(path);

export function useConfig() {
  return useSWR<Config>("/config", fetcher);
}

export function useModels() {
  return useSWR<string[]>("/models", fetcher);
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
) {
  return useSWR<BacktestRow[]>(
    asset && model
      ? `/backtests?asset=${asset}&model=${encodeURIComponent(model)}&alpha=${alpha}&limit=${limit}`
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

export function useRegimes(asset: string | null, limit = 2000) {
  return useSWR<RegimesResponse>(
    asset ? `/regimes/${asset}?limit=${limit}` : null,
    fetcher,
  );
}
