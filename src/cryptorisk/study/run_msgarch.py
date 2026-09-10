"""Export returns, run the R MS-GARCH walk-forward, load predictions into the
store's ``msgarch_predictions`` table (V2_PLAN §8, Phase 2c).

    python -m cryptorisk.study.run_msgarch

Requires R on PATH with the ``MSGARCH`` package installed. Takes ~10-20 min for
two assets.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pandas as pd

from cryptorisk.config import load_config, repo_root
from cryptorisk.data import store


def main() -> None:
    cfg = load_config()
    root = repo_root()
    db = str(root / cfg["paths"]["store"])
    res_dir = root / "data" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    rscript = shutil.which("Rscript")
    if rscript is None:
        sys.exit("Rscript not found on PATH. Install R + the MSGARCH package.")

    con = store.connect(db)
    parts = []
    for a in cfg["assets"]:
        r = con.execute(
            "SELECT date, log_return FROM returns_daily WHERE asset = ? ORDER BY date", [a]
        ).df()
        r["asset"] = a
        parts.append(r[["asset", "date", "log_return"]])
    in_csv = res_dir / "msgarch_input.csv"
    pd.concat(parts, ignore_index=True).to_csv(in_csv, index=False)
    print(f"[msgarch] wrote {in_csv} ({sum(len(p) for p in parts)} rows)", flush=True)

    script = root / "msgarch" / "fit_msgarch_walkforward.R"
    print(f"[msgarch] Rscript {script} (this takes a while)...", flush=True)
    subprocess.run([rscript, str(script), str(in_csv), str(res_dir)], check=True, cwd=root)

    preds = []
    for a in cfg["assets"]:
        fp = res_dir / f"msgarch_pred_{a}.csv"
        if not fp.exists():
            sys.exit(f"expected {fp} from the R run, not found")
        df = pd.read_csv(fp, parse_dates=["prev_date", "date"])
        preds.append(df)
    allp = pd.concat(preds, ignore_index=True)
    n = store.write_msgarch_predictions(con, allp)
    con.close()

    print(f"[msgarch] loaded {n} predictions into msgarch_predictions")
    summary = allp.groupby("asset").agg(
        rows=("date", "size"),
        var_0025_avg=("var_0025", "mean"),
        var_001_avg=("var_001", "mean"),
        any_positive_var=("var_0025", lambda v: bool((v > 0).any())),
        p_crisis_insample=("prob_crisis_insample", "mean"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
