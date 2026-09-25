"""Local (today's-forecast) explanation of LSTM-Vol, saved for the deployed API.

The deployed API has no torch, so it cannot fit the network. This runs where torch is
installed (your machine, the daily GitHub workflow) on the latest 500-day window from
``price_history.parquet`` and writes ``explain_lstm_local.csv``: for each asset and
level, the occlusion contribution of each of the last 20 days (see
:meth:`cryptorisk.models.lstm_vol.LstmVol.explain_local`).

    python -m cryptorisk.study.run_lstm_local
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptorisk.config import load_config, repo_root
from cryptorisk.models.base import Context
from cryptorisk.models.lstm_vol import LstmVol

WINDOW = 500


def compute(hist: pd.DataFrame, assets: list[str], alphas: list[float]) -> pd.DataFrame:
    rows = []
    for asset in assets:
        win = hist[hist["asset"] == asset].sort_values("date").tail(WINDOW)
        if len(win) < WINDOW:
            continue
        dates = win["date"].to_numpy("datetime64[D]")
        model = LstmVol()
        for alpha in alphas:
            ctx = Context(returns=win["log_return"].to_numpy(float), dates=dates, asof=dates[-1], asset=asset)
            ex = model.explain_local(ctx, alpha)   # the second alpha reuses the fitted network
            if ex is None:
                continue
            n = len(ex["dates"])
            for k in range(n):
                rows.append({
                    "asset": asset, "alpha": alpha, "asof": str(dates[-1]), "date": ex["dates"][k],
                    "lag": n - k,                                     # 1 = the latest day
                    "ret": ex["returns"][k], "per_day": ex["per_day"][k],
                    "c_return": ex["cell"][k][0], "c_squared": ex["cell"][k][1], "c_down_squared": ex["cell"][k][2],
                    "base_var": ex["base_var"], "flat_var": ex["flat_var"],
                })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    res = repo_root() / cfg["paths"]["results"]
    hist = pd.read_parquet(res / "price_history.parquet")
    hist["date"] = pd.to_datetime(hist["date"])
    out = compute(hist, list(cfg["assets"]), list(cfg["alphas"]))
    if out.empty:
        raise SystemExit("no LSTM explanation produced (is torch installed?)")
    out.to_csv(res / "explain_lstm_local.csv", index=False)
    top = out[(out.asset == out.asset.iloc[0]) & (out.alpha == out.alpha.iloc[0])].nlargest(3, "per_day")
    print(f"[lstm-local] {len(out)} rows; asof {out['asof'].iloc[0]}; biggest drivers:")
    print(top[["date", "ret", "per_day"]].round(5).to_string(index=False))
    print("all-flat VaR magnitude", round(float(np.mean(out["flat_var"])), 4),
          "vs actual", round(float(np.mean(out["base_var"])), 4))


if __name__ == "__main__":
    main()
