"""Volatility-forecast evaluation against realized variance (V2_PLAN §5.4).

The realized measures unlock a second, VaR-independent comparison: how well does
each model forecast *variance*?  Truth proxy is the 5-minute realized variance
``RV_t`` (sub-sampled); the forecast is the model's ``sigma2`` for the same day.
We report

* per-model mean **QLIKE** and **MSE** (QLIKE is robust to the noisy proxy,
  Patton 2011);
* a **Model Confidence Set** over the QLIKE loss series -> which models forecast
  variance best, at 90% confidence;
* the **Mincer-Zarnowitz** regression ``RV = a + b * sigma2`` with the joint test
  ``a = 0, b = 1`` (a calibrated forecast) and its R-squared.

Quantile-only models (CAViaR) have no ``sigma2`` and are skipped.

    python -m cryptorisk.study.vol_forecast_eval
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from cryptorisk.backtest.scoring import model_confidence_set, qlike_loss
from cryptorisk.config import load_config, repo_root


def _mincer_zarnowitz(rv: np.ndarray, s2: np.ndarray) -> dict[str, float]:
    """OLS ``rv = a + b*s2``; Wald test of (a, b) = (0, 1) ~ chi2(2)."""
    x = np.column_stack([np.ones_like(s2), s2])
    n = rv.size
    xtx = x.T @ x
    try:
        beta = np.linalg.solve(xtx, x.T @ rv)
    except np.linalg.LinAlgError:
        return dict(mz_a=np.nan, mz_b=np.nan, mz_joint_p=np.nan, mz_r2=np.nan)
    resid = rv - x @ beta
    rss = float(resid @ resid)
    dof = max(n - 2, 1)
    cov = (rss / dof) * np.linalg.inv(xtx)
    diff = beta - np.array([0.0, 1.0])
    try:
        wald = float(diff @ np.linalg.solve(cov, diff))
    except np.linalg.LinAlgError:
        wald = np.nan
    tss = float(((rv - rv.mean()) ** 2).sum())
    return dict(
        mz_a=float(beta[0]),
        mz_b=float(beta[1]),
        mz_joint_p=float(stats.chi2.sf(wald, 2)) if np.isfinite(wald) else np.nan,
        mz_r2=float(1.0 - rss / tss) if tss > 0 else np.nan,
    )


def evaluate_vol_forecasts(
    backtests: pd.DataFrame,
    realized: pd.DataFrame,
    *,
    asset: str,
    window: int | str,
    mcs_cfg: dict,
    seed: int,
) -> pd.DataFrame:
    """One row per model for the given (asset, window). ``realized`` needs
    columns ``date``, ``rv`` (any asset; filtered here)."""
    amin = backtests["alpha"].min()  # sigma2 is identical across alpha
    bt = backtests[
        (backtests["asset"] == asset)
        & (backtests["window"] == window)
        & (backtests["alpha"] == amin)
    ]
    rv = realized.loc[realized["asset"] == asset, ["date", "rv"]].dropna().copy()
    rv["date"] = pd.to_datetime(rv["date"])

    wide = bt.pivot_table(index="date", columns="model", values="sigma2")
    # Drop a model if it has ANY missing sigma2, not just if it never has one
    # (matches evaluate_fz0_mcs's column-wise rule). A model that is only
    # sparsely missing (e.g. a cache-served bridge with a gap) would otherwise
    # survive here and force a row-wise .dropna() below that silently shrinks
    # the QLIKE-MCS sample for every *other* model too.
    wide = wide.dropna(axis=1, how="any")
    merged = wide.join(rv.set_index("date")["rv"], how="inner").dropna()
    if merged.empty or merged.shape[1] < 2:
        return pd.DataFrame()

    rv_vec = merged["rv"].to_numpy(float)
    models = [c for c in merged.columns if c != "rv"]
    losses = {m: qlike_loss(rv_vec, merged[m].to_numpy(float)) for m in models}
    mcs = model_confidence_set(
        losses,
        confidence=mcs_cfg["confidence"],
        block_len=mcs_cfg["block_bootstrap_len"],
        n_boot=mcs_cfg["n_boot"],
        seed=seed,
    )

    rows = []
    for m in models:
        s2 = merged[m].to_numpy(float)
        rows.append(
            {
                "asset": asset,
                "window": window,
                "model": m,
                "n": len(merged),
                "qlike": float(np.mean(losses[m])),
                "mse": float(np.mean((rv_vec - s2) ** 2)),
                "in_mcs_qlike": m in mcs.included,
                "mcs_p": float(mcs.p_values.get(m, np.nan)),
                **_mincer_zarnowitz(rv_vec, s2),
            }
        )
    out = pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)
    out.insert(3, "qlike_rank", np.arange(1, len(out) + 1))
    return out


def run() -> pd.DataFrame:
    cfg = load_config()
    res_dir = repo_root() / cfg["paths"]["results"]
    bt = pd.read_parquet(res_dir / "backtests.parquet")
    bt["date"] = pd.to_datetime(bt["date"])

    import duckdb

    con = duckdb.connect(str(repo_root() / cfg["paths"]["store"]), read_only=True)
    rv = con.execute("SELECT asset, date, rv FROM realized_daily ORDER BY date").df()
    con.close()

    frames = []
    for asset in cfg["assets"]:
        for window in sorted(bt["window"].unique(), key=str):
            frames.append(
                evaluate_vol_forecasts(
                    bt,
                    rv,
                    asset=asset,
                    window=window,
                    mcs_cfg=cfg["evaluation"]["mcs"],
                    seed=cfg["seed"],
                )
            )
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out.to_csv(res_dir / "eval_volforecast.csv", index=False)
    print(f"[vol-forecast] wrote {len(out)} rows -> data/results/eval_volforecast.csv")
    cols = ["asset", "window", "model", "qlike_rank", "qlike", "in_mcs_qlike", "mz_b", "mz_r2"]
    print(out[cols].to_string(index=False))
    return out


if __name__ == "__main__":
    run()
