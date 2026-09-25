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

export interface ModelInfo {
  family: string;
  idea: string;
  limitations: string[];
}

export interface RfImportanceRow {
  date: string;
  asset: string;
  feature: string;
  importance: number;
}

export interface RfDiagnosticsRow {
  date: string;
  asset: string;
  ess: number;
  n_train: number;
  var_cond: number;
  var_hs: number;
}

export interface RfInputRow {
  asset: string;
  date: string;
  feature: string;
  value: number;
  zscore: number;
}

export interface LstmImportanceRow {
  lag: number;
  feature: string;
  importance: number;
  sd: number | null;
  n_refits: number;
}

export interface RfPdp {
  feature: string;
  grid: number[];
  var: number[];
  actual: number;
}

export interface RfExplain {
  importance: RfImportanceRow[];
  diagnostics: RfDiagnosticsRow[];
  inputs: RfInputRow[];
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
  cone?: ForecastCone;
  /** Normal vs. crisis regime, as a distribution comparison (not a
   * horizon-indexed forecast) -- see RegimeDistribution.tsx. Null if
   * data/results/msgarch_regime_params.csv doesn't cover this asset. */
  regime_summary?: RegimeSummary | null;
  /** The current, incomplete UTC day measured from `last_close`. */
  intraday?: IntradayStatus | null;
}

export interface IntradayStatus {
  date: string;
  n_bars: number;
  fraction_of_day: number;
  price: number;
  ret_so_far: number;
  low_ret: number;
  high_ret: number;
  var_breached: boolean;
  es_breached: boolean;
}

export interface RegimeSummary {
  normal: RegimeParams;
  crisis: RegimeParams;
}

export interface RegimeParams {
  vol: number;
  nu: number;
  p_stay: number | null;
}

export interface ConePoint {
  days: number;
  var_price: number | null;
  es_price: number | null;
  upper_price: number | null;
}

export interface ForecastCone {
  horizons_days: number[];
  /** Merton jump-diffusion, compounded exactly to each horizon (Poisson jump
   * count scales with days, diffusion variance scales with days). */
  jump_diffusion: ConePoint[];
  /** GARCH-EVT: cumulative variance is the sum of arch's per-step forecasts
   * (mean-reverting), same fitted GPD tail rescaled per horizon. */
  garch_evt: ConePoint[];
}

export interface ModelLatestRow {
  model: string;
  date: string;
  var: number;
  es: number;
  violation: boolean;
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
  basel_zone: string;
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

export interface RegimeStatRow {
  source: string;
  regime: "Normal" | "Crisis";
  n: number;
  share: number;
  mean: number;
  sd: number;
  skew: number;
  kurt: number;
  mean_abs: number;
  vol_ratio: number;
  levene_p: number;
}

export interface PcrisisBandRow {
  asset: string;
  date: string;
  p_point: number;
  p_lo: number;
  p_med: number;
  p_hi: number;
  p_sd: number;
  n_ok: number;
}

export interface RegimesResponse {
  series: RegimePoint[];
  correlations: RegimeCorrRow[];
  regime_summary?: RegimeSummary | null;
  stats?: RegimeStatRow[];
  band?: PcrisisBandRow[];
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

export interface LiveDay {
  date: string;
  realized: number;
  var_min: number;
  var_max: number;
  var_median: number;
  es_median: number;
}

export interface LiveModelRow {
  model: string;
  n: number;
  violations: number;
  expected: number;
  hit_rate: number;
  p_at_least: number;
  mean_fz0: number;
  fz0_rank: number;
  mean_var: number;
  mean_es: number;
  min_margin: number;
}

export interface LiveTrackRecord {
  days: LiveDay[];
  models: LiveModelRow[];
}

export interface WhatIfRow {
  asset: string;
  model: string;
  alpha: number;
  shock: number;
  last_close: number;
  shocked_close: number;
  baseline: { var: number; es: number };
  /** The same model after a flat (0%) day: what the shock is measured against. */
  flat: { var: number; es: number };
  shocked: { var: number; es: number };
  baseline_var_price: number | null;
  shocked_var_price: number | null;
}

export interface VarChangeRow {
  asset: string;
  model: string;
  alpha: number;
  asof: string;
  prev_asof: string;
  new_date: string;
  new_return: number;
  dropped_date: string;
  dropped_return: number;
  var: { prev: number; now: number; new: number; old: number };
  es: { prev: number; now: number; new: number; old: number };
}

export interface ConformalRow {
  asset: string;
  alpha: number;
  base_model: string;
  variant: "raw" | "ACI";
  n: number;
  hit_rate: number;
  kupiec_p: number;
  chr_cc_p: number;
  dq_p: number;
  fz0_mean: number;
  mean_var: number;
  mean_es: number;
  /** ACI rows only: mean FZ0(ACI) - FZ0(raw); negative = ACI scores better. */
  dm_fz0_diff?: number | null;
  dm_p?: number | null;
}

export interface ConformalPathRow {
  date: string;
  model: string;
  level: number;
}

export interface LstmLocalDay {
  date: string;
  lag: number;
  ret: number;
  per_day: number;
  c_return: number;
  c_squared: number;
  c_down_squared: number;
}

export interface LstmLocal {
  asof: string | null;
  base_var: number | null;
  flat_var: number | null;
  days: LstmLocalDay[];
}

export interface BaselHeadroomRow {
  model: string;
  as_of: string;
  exceptions_250d: number;
  zone: string;
  zone_if_breached: string;
  /** Exceptions until the next zone; null in the red zone. */
  to_next_zone: number | null;
  /** Exceptions in the oldest 30 days of the 250-day window: they leave it within a month. */
  ageing_out_30d: number;
  m_c: number;
  m_c_if_breached: number;
  capital_usd: number;
  extra_capital_if_breached_usd: number;
}

export interface BreachDistance {
  asset: string;
  model: string;
  alpha: number;
  asof: string;
  spot: number;
  last_close: number;
  var_price: number | null;
  es_price: number | null;
  dist_var: number | null;
  dist_es: number | null;
}
