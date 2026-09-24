"""One-at-a-time parameter sweep for the two-touch-low candidate
(two_touch_low_daybyday.py), on the same fixed 20-ticker universe as the
baseline run (seed=21, k=20) so results are comparable.

For each of the 7 variables in turn, holds the other six at the locked
starting values (mean_reversion_notes.md, 2026-09-22) and tries a small
set of alternative values, reusing the exact same decide()/run_ticker()
logic (and its truncation-based causal check) as the baseline script --
no separate implementation to drift out of sync.

This is a naive one-at-a-time sweep, the same shape that produced a
mirage on the birth-volume threshold sweep (tech_levels_notes.md,
"Birth-day volume") -- read any apparent improvement here as a hypothesis
for the generalization check on a different ticker set, not a confirmed
effect. Data is pulled once and reused across every combo.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_sweep.py [seed] [n_tickers]
"""
import datetime
import os
import random
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from tech_level_naive_strategy import equal_weight_curve, attach_benchmark
from tech_level_live import pull_all
from tech_level_continuation_live import LIVE_TICKERS
from daybyday_variants import summ
from two_touch_low_daybyday import PARAMS as BASE_PARAMS, run_ticker

GRID = {
    "lookback": [3, 5, 7, 10],
    "prominence_pct": [0.0, 0.005, 0.01, 0.015],
    "volume_threshold": [0.8, 1.0, 1.3, 1.6],
    "k_pct": [0.003, 0.005, 0.008, 0.012],
    "rebound_max_pct": [0.015, 0.02, 0.025, 0.03],
    "rebound_volume_threshold": [0.7, 1.0, 1.3, 1.6],
    "volume_threshold_buy": [0.6, 0.8, 1.0, 1.3],
}


def run_combo(close, volume, ew, params):
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, [(t, close[t], volume[t], params) for t in close])
    df = pd.DataFrame([t for o in out for t in o])
    if df.empty:
        return df
    return attach_benchmark(df, ew)


def row(df, label):
    if df.empty or len(df) < 3:
        return dict(label=label, n=len(df), hit=np.nan, net=np.nan, excess=np.nan, t=np.nan)
    ex = df.excess_ret
    return dict(label=label, n=len(df), hit=(df.ret > 0).mean(), net=df.ret_net.mean(),
                excess=ex.mean(), t=ex.mean() / ex.std() * np.sqrt(len(df)))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}\n", flush=True)

    series, failed = pull_all(tickers)
    today = datetime.date.today()
    close, volume = {}, {}
    for t in tickers:
        if t not in series:
            continue
        c, v = series[t]
        c = c.copy()
        c.index = pd.to_datetime(c.index)
        mask = c.index.date < today
        close[t] = c[mask]
        if v is not None:
            v = v.copy()
            v.index = pd.to_datetime(v.index)
            volume[t] = v.reindex(close[t].index)
        else:
            volume[t] = pd.Series(np.nan, index=close[t].index)
    ew = equal_weight_curve(close)

    baseline = run_combo(close, volume, ew, BASE_PARAMS)
    b = row(baseline, "baseline (locked)")
    print(f"baseline: n={b['n']}  hit={b['hit']:.1%}  net={b['net']:+.3%}  excess={b['excess']:+.3%}  t={b['t']:+.2f}\n")

    best = dict(BASE_PARAMS)
    for pname, values in GRID.items():
        print(f"--- {pname} (others held at locked values) ---")
        rows = []
        for v in values:
            p = dict(BASE_PARAMS)
            p[pname] = v
            df = run_combo(close, volume, ew, p)
            r = row(df, f"{pname}={v}")
            rows.append(r)
            tag = " <- locked" if v == BASE_PARAMS[pname] else ""
            print(f"  {pname}={v:<8} n={r['n']:4d}  hit={r['hit']:.1%}  net={r['net']:+.3%}  "
                  f"excess={r['excess']:+.3%}  t={r['t']:+.2f}{tag}")
        valid = [r for r in rows if r["n"] >= 30]
        if valid:
            top = max(valid, key=lambda r: r["t"])
            top_v = values[rows.index(top)]
            print(f"  best by t-stat (n>=30): {pname}={top_v}\n")
            best[pname] = top_v
        else:
            print("  no candidate reached n>=30, keeping locked value\n")

    print("=== combined: each parameter's best one-at-a-time value together ===")
    print(f"params: {best}\n")
    df = run_combo(close, volume, ew, best)
    r = row(df, "combined")
    print(f"combined: n={r['n']}  hit={r['hit']:.1%}  net={r['net']:+.3%}  excess={r['excess']:+.3%}  t={r['t']:+.2f}")
    if not df.empty:
        print("\nby branch:")
        for br, g in df.groupby("branch"):
            print(summ(g, br))


if __name__ == "__main__":
    main()
