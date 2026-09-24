"""One-at-a-time parameter sweep for the two-touch-low v2 candidate
(two_touch_low_v2_daybyday.py), on the same fixed 20-ticker universe as
that script's baseline run (seed=21, k=20) so results are comparable.

For each of the 5 variables in turn, holds the other four at the locked
starting values (two_touch_low_v2_daybyday.PARAMS, 2026-09-23) and tries a
small set of alternative values, reusing the exact same decide()/
run_ticker() logic (and its truncation-based causal check) as the
baseline script -- no separate implementation to drift out of sync.

This is a naive one-at-a-time sweep, the same shape that produced a
mirage on the birth-volume threshold sweep (tech_levels_notes.md,
"Birth-day volume") and that two_touch_low_sweep.py already disclaims for
the v1 rule. volume_threshold_L and volume_threshold_L2 are exactly the
kind of variable that reminder is about -- read any apparent monotonic
improvement across {0.8, 1.0, 1.2} as a hypothesis for the generalization
check on a different ticker set (or a quantile-bucket recheck against the
raw volume_ratio), not a confirmed dose-response. Data is pulled once and
reused across every combo.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_v2_sweep.py [seed] [n_tickers] [dd_pct]
"""
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
from two_touch_low_v2_daybyday import PARAMS as BASE_PARAMS, run_ticker

GRID = {
    "n_back": [3, 4, 5, 6, 7, 8, 9, 10],
    "n_fwd": [1, 2, 3, 4, 5],
    "volume_threshold_L": [0.8, 1.0, 1.2],
    "volume_threshold_L2": [0.8, 1.0, 1.2],
    "close_tolerance": [0.001, 0.0025, 0.005, 0.0075, 0.01],
}


def run_combo(close, high, low, volume, spy_close, ew, params, dd_pct):
    args = [(t, close[t], high[t], low[t], volume[t], spy_close[t], params, dd_pct) for t in close]
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, args)
    df = pd.DataFrame([row for o in out for row in o])
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
    dd_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}  (dd_pct={dd_pct:.0%})\n", flush=True)

    series, failed = pull_all(tickers + ["SPY"], include_hl=True)
    if failed:
        print(f"no data for: {failed}")
    import datetime
    today = datetime.date.today()
    close, high, low, volume = {}, {}, {}, {}
    for t in tickers:
        if t not in series:
            continue
        c, h, l, v = series[t]
        c = c.copy(); c.index = pd.to_datetime(c.index)
        mask = c.index.date < today
        close[t] = c[mask]
        h = h.copy() if h is not None else pd.Series(np.nan, index=c.index)
        h.index = pd.to_datetime(h.index)
        high[t] = h.reindex(close[t].index)
        l = l.copy() if l is not None else pd.Series(np.nan, index=c.index)
        l.index = pd.to_datetime(l.index)
        low[t] = l.reindex(close[t].index)
        if v is not None:
            v = v.copy(); v.index = pd.to_datetime(v.index)
            volume[t] = v.reindex(close[t].index)
        else:
            volume[t] = pd.Series(np.nan, index=close[t].index)
    ew = equal_weight_curve(close)
    spy_close_raw = series["SPY"][0].copy()
    spy_close_raw.index = pd.to_datetime(spy_close_raw.index)
    spy_close = {t: spy_close_raw.reindex(close[t].index, method="ffill") for t in close}

    baseline = run_combo(close, high, low, volume, spy_close, ew, BASE_PARAMS, dd_pct)
    b = row(baseline, "baseline (locked)")
    print(f"baseline: n={b['n']}  hit={b['hit']:.1%}  net={b['net']:+.3%}  excess={b['excess']:+.3%}  t={b['t']:+.2f}\n")

    best = dict(BASE_PARAMS)
    for pname, values in GRID.items():
        print(f"--- {pname} (others held at locked values) ---", flush=True)
        rows = []
        for v in values:
            p = dict(BASE_PARAMS)
            p[pname] = v
            df = run_combo(close, high, low, volume, spy_close, ew, p, dd_pct)
            r = row(df, f"{pname}={v}")
            rows.append(r)
            tag = " <- locked" if v == BASE_PARAMS[pname] else ""
            print(f"  {pname}={v:<8} n={r['n']:4d}  hit={r['hit']:.1%}  net={r['net']:+.3%}  "
                  f"excess={r['excess']:+.3%}  t={r['t']:+.2f}{tag}", flush=True)
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
    df = run_combo(close, high, low, volume, spy_close, ew, best, dd_pct)
    r = row(df, "combined")
    print(f"combined: n={r['n']}  hit={r['hit']:.1%}  net={r['net']:+.3%}  excess={r['excess']:+.3%}  t={r['t']:+.2f}")
    if not df.empty:
        print(f"\nexit reasons: {df.exit_reason.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
