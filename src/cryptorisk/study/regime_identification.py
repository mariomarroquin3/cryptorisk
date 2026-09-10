"""Identification vs. estimation-window study for the 2-regime MS-GARCH
(V2_PLAN §5.5) -- a methodological contribution, not a forecasting result.

The claim carried through v1 and ``docs/methodology.tex`` §9(i): on a rolling
500-day window the two regimes are only weakly identified, so the walk-forward
filtered probability of the high-variance state carries almost no
out-of-sample signal, whereas a single full-sample fit does.  This module
quantifies that.

* :func:`analyse_cached` -- correlations of each regime-probability series in
  ``data/results/msgarch_pred_<asset>.csv``
  (``prob_crisis_filt`` / ``prob_crisis_pred`` = walk-forward W=500,
  ``prob_crisis_insample`` = one full-sample fit) with ``|r_t|``, daily ``RV``
  and 21-day ``RV``.  Runs from artefacts already on disk.

* :func:`run_window_sweep` -- shells out to
  ``msgarch/fit_msgarch_regime_windows.R`` to refit for W in
  {500, 750, 1000, 1500, 2000, expanding} (regime probabilities only, plus a
  constrained spec that shares alpha/beta across regimes), then builds the
  "corr vs window length" curve.  Multi-hour R job; gated behind ``--run-r``.

    python -m cryptorisk.study.regime_identification            # cached only
    python -m cryptorisk.study.regime_identification --run-r    # + the R sweep
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

from cryptorisk.config import load_config, repo_root

_PROB_COLS = {
    "filt_wf": "prob_crisis_filt",
    "pred_wf": "prob_crisis_pred",
    "insample": "prob_crisis_insample",
}


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan, np.nan
    return (
        float(np.corrcoef(a[ok], b[ok])[0, 1]),
        float(stats.spearmanr(a[ok], b[ok]).statistic),
    )


def analyse_cached() -> pd.DataFrame:
    cfg = load_config()
    root = repo_root()
    res_dir = root / cfg["paths"]["results"]

    import duckdb

    con = duckdb.connect(str(root / cfg["paths"]["store"]), read_only=True)
    ret = con.execute("SELECT asset, date, log_return FROM returns_daily ORDER BY date").df()
    rv = con.execute("SELECT asset, date, rv FROM realized_daily ORDER BY date").df()
    con.close()
    for d in (ret, rv):
        d["date"] = pd.to_datetime(d["date"])

    rows = []
    for asset in cfg["assets"]:
        fp = res_dir / f"msgarch_pred_{asset}.csv"
        if not fp.exists():
            print(f"[regime-id] {fp.name} missing -- run `make msgarch`", file=sys.stderr)
            continue
        pred = pd.read_csv(fp, parse_dates=["date"])
        m = (
            pred.merge(ret[ret.asset == asset][["date", "log_return"]], on="date", how="left")
            .merge(rv[rv.asset == asset][["date", "rv"]], on="date", how="left")
            .sort_values("date")
        )
        absr = m["log_return"].abs().to_numpy(float)
        rvd = m["rv"].to_numpy(float)
        rv21 = m["rv"].rolling(21, min_periods=10).mean().to_numpy(float)

        for label, col in _PROB_COLS.items():
            if col not in m:
                continue
            p = m[col].to_numpy(float)
            pear_r, spear_r = _corr(p, absr)
            pear_rv, spear_rv = _corr(p, rvd)
            pear_rv21, spear_rv21 = _corr(p, rv21)
            rows.append(
                {
                    "asset": asset,
                    "series": label,
                    "kind": "walk-forward W=500" if label.endswith("wf") else "full-sample fit",
                    "n": int(np.isfinite(p).sum()),
                    "mean_prob": float(np.nanmean(p)),
                    "corr_absret": pear_r,
                    "spearman_absret": spear_r,
                    "corr_rv": pear_rv,
                    "corr_rv21": pear_rv21,
                    "spearman_rv21": spear_rv21,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(res_dir / "regime_identification.csv", index=False)
    return out


def run_window_sweep() -> pd.DataFrame:
    cfg = load_config()
    root = repo_root()
    res_dir = root / cfg["paths"]["results"]
    rscript = shutil.which("Rscript")
    if rscript is None:
        sys.exit("Rscript not found on PATH -- cannot run the window sweep.")

    in_csv = res_dir / "msgarch_input.csv"
    if not in_csv.exists():
        sys.exit(f"{in_csv} missing -- run `make msgarch` once to export returns.")
    script = root / "msgarch" / "fit_msgarch_regime_windows.R"
    print(f"[regime-id] Rscript {script.name} (multi-hour) ...", flush=True)
    subprocess.run([rscript, str(script), str(in_csv), str(res_dir)], check=True, cwd=root)

    frames = []
    for asset in cfg["assets"]:
        fp = res_dir / f"regime_windows_{asset}.csv"
        if fp.exists():
            frames.append(pd.read_csv(fp, parse_dates=["date"]))
    if not frames:
        sys.exit("the R sweep produced no regime_windows_<asset>.csv")
    sweep = pd.concat(frames, ignore_index=True)

    import duckdb

    con = duckdb.connect(str(root / cfg["paths"]["store"]), read_only=True)
    ret = con.execute("SELECT asset, date, log_return FROM returns_daily").df()
    con.close()
    ret["date"] = pd.to_datetime(ret["date"])

    rows = []
    for (asset, window, spec), g in sweep.groupby(["asset", "window", "spec"], observed=True):
        g = g.merge(ret[ret.asset == asset][["date", "log_return"]], on="date", how="left")
        pear, spear = _corr(
            g["prob_crisis_filt"].to_numpy(float), g["log_return"].abs().to_numpy(float)
        )
        rows.append(
            {
                "asset": asset,
                "window": window,
                "spec": spec,
                "n": len(g),
                "corr_absret": pear,
                "spearman_absret": spear,
            }
        )
    curve = pd.DataFrame(rows).sort_values(["asset", "spec", "window"])
    curve.to_csv(res_dir / "regime_identification_windows.csv", index=False)
    return curve


def _summary_md(cached: pd.DataFrame, curve: pd.DataFrame | None) -> str:
    out = ["# MS-GARCH regime identification vs. estimation window", ""]
    out.append("_`python -m cryptorisk.study.regime_identification`._\n")
    out.append(
        "Correlation of the high-variance-regime probability with the realized volatility state.\n"
    )
    out.append("| asset | series | corr(\\|r\\|) | Spearman(\\|r\\|) | corr(RV) | corr(RV21) |")
    out.append("|:--|:--|--:|--:|--:|--:|")
    for _, r in cached.iterrows():
        out.append(
            f"| {r['asset']} | {r['series']} ({r['kind']}) | {r['corr_absret']:.3f} | "
            f"{r['spearman_absret']:.3f} | {r['corr_rv']:.3f} | {r['corr_rv21']:.3f} |"
        )
    out.append(
        "\nRead: the full-sample fit tracks the volatility state; the "
        "walk-forward W=500 filtered/predicted probability barely does. The "
        "regime layer is therefore descriptive only (labelled in-sample) and is "
        "kept out of the VaR backtest.\n"
    )
    if curve is not None and not curve.empty:
        out.append("## corr(P(high-vol), |r|) vs estimation-window length\n")
        out.append("| asset | spec | window | corr(\\|r\\|) | Spearman |")
        out.append("|:--|:--|--:|--:|--:|")
        for _, r in curve.iterrows():
            out.append(
                f"| {r['asset']} | {r['spec']} | {r['window']} | "
                f"{r['corr_absret']:.3f} | {r['spearman_absret']:.3f} |"
            )
        out.append("")
    return "\n".join(out)


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-r", action="store_true", help="also run the multi-hour R window sweep")
    args = ap.parse_args()

    cached = analyse_cached()
    print(cached.to_string(index=False))

    curve = None
    res_dir = repo_root() / load_config()["paths"]["results"]
    if args.run_r:
        curve = run_window_sweep()
        print(curve.to_string(index=False))
    else:
        fp = res_dir / "regime_identification_windows.csv"
        if fp.exists():
            curve = pd.read_csv(fp)

    (res_dir / "regime_identification.md").write_text(_summary_md(cached, curve), encoding="utf-8")
    print(f"[regime-id] wrote regime_identification.csv / .md -> {res_dir}")


if __name__ == "__main__":
    run()
