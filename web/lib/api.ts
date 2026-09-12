// Thin typed client for the cryptorisk FastAPI backend (src/cryptorisk/api).
// Every shape here is permissive (extra fields ignored) since the backend is
// a thin pass-through over data/results/*.csv -- see that package for the
// authoritative schema.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, body || res.statusText);
  }
  return res.json() as Promise<T>;
}

export interface Config {
  assets: string[];
  alphas: number[];
  oos_start: string;
  portfolio: {
    assets: string[];
    weights: Record<string, number>;
    copulas: string[];
  };
}

export interface PriceResponse {
  asset: string;
  price: number;
  change_pct: number;
  high: number;
  low: number;
  volume: number;
  fetched_at: number;
}

export interface ForecastResponse {
  asset: string;
  model: string;
  alpha: number;
  source: "live_refit" | "frozen_backtest";
  asof?: string;
  date?: string;
  last_close?: number;
  var: number | null;
  es: number | null;
  upper?: number | null;
  var_price?: number | null;
  es_price?: number | null;
  upper_price?: number | null;
}

export interface Fz0Row {
  asset: string;
  window: number;
  alpha: number;
  model: string;
  n: number;
  n_degenerate: number;
  fz0_mean: number;
  fz0_rank: number;
  is_best: boolean;
  dm_vs_best_p: number | null;
  in_mcs: boolean;
  mcs_p: number;
}

export interface CoverageRow {
  asset: string;
  model: string;
  hit_rate: number;
  kupiec_p: number;
  chr_cc_p: number;
  dq_p: number;
  basel_zone: string | null;
  passes_all: boolean;
}

export interface EsTestRow {
  asset: string;
  model: string;
  n_breach: number;
  z1: number;
  z1_p_approx: number;
  z2: number;
  z2_p_approx: number;
  es_reject_approx: boolean;
}

export interface GwCpaRow {
  asset: string;
  alpha: number;
  model_a: string;
  model_b: string;
  mean_fz0_gap: number;
  gw_stat: number;
  gw_p: number;
  gw_reject: boolean;
  best_edge_vs_rv: string;
}

export interface BacktestRow {
  date: string;
  asset: string;
  model: string;
  window: number;
  alpha: number;
  var: number;
  es: number;
  sigma2: number | null;
  realized: number;
  violation: boolean;
  pit: number | null;
}

export interface CapitalRow {
  asset: string;
  model: string;
  in_mcs: boolean;
  es_975_1d: number;
  es_10d_sqrt: number;
  es_10d_bootstrap: number;
  exceptions_250d: number;
  m_c: number;
  capital_usd: number;
  model_risk_addon_usd: number;
}

export interface EstimationRiskRow {
  asset: string;
  estimator: string;
  alpha: number;
  capital_point_usd: number;
  estimation_risk_addon_usd: number;
}

export interface LimitsRow {
  asset: string;
  model: string;
  in_mcs: boolean;
  n_star_usd: number;
  bind_rate: number;
  mean_utilisation: number;
  budget_breach_rate: number;
  worst_loss_usd: number;
}

export interface HedgeRow {
  asset: string;
  ratio_min_var: number;
  ratio_es_min: number | null;
  es_unhedged: number;
  es_hedged: number | null;
  es_reduction: number | null;
  funding_carry_annual_frac: number;
  funding_carry_annual_usd: number;
  note: string | null;
}

export interface RegimePoint {
  date: string;
  sigma2: number | null;
  prob_crisis_insample: number;
  prob_crisis_pred: number;
}

export interface RegimeCorrRow {
  asset: string;
  series: string;
  kind: string;
  corr_absret: number;
  spearman_absret: number;
  corr_rv: number;
  spearman_rv21: number;
  mean_prob: number;
}

export interface RegimesResponse {
  series: RegimePoint[];
  correlations: RegimeCorrRow[];
}

export interface PriceHistoryRow {
  date: string;
  close: number;
  log_return: number;
}

export interface PortfolioEvalRow {
  asset: string;
  model: string;
  fz0_rank: number;
  fz0_mean: number;
  in_mcs: boolean;
  hit_rate: number;
  kupiec_p: number;
  z2: number;
  basel_zone: string | null;
  passes_all: boolean;
}

export interface PortfolioComposition {
  weights: Record<string, number>;
  prices: Record<string, PriceResponse | null>;
}
