"""Holding-period sweep for the two-touch-low candidate: same locked entry
parameters (mean_reversion_notes.md, 2026-09-22), same fixed 20-ticker
universe (seed=21, k=20) as the baseline/parameter-sweep runs, only the
exit rule changes -- fixed hold of 1, 2, 3, or 5 sessions instead of always
5. Reuses decide()/run_ticker() from two_touch_low_daybyday.py (including
its truncation-based causal check) so there's no separate exit-rule logic
to drift out of sync.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_hold_sweep.py [seed] [n_tickers]
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
from two_touch_low_daybyday import PARAMS, run_ticker

HOLDS = [1, 2, 3, 5]


def run_combo(close, volume, ew, hold):
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, [(t, close[t], volume[t], PARAMS, hold) for t in close])
    df = pd.DataFrame([t for o in out for t in o])
    if df.empty:
        return df
    return attach_benchmark(df, ew)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}")
    print(f"params: {PARAMS}\n", flush=True)

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

    results = {}
    for hold in HOLDS:
        df = run_combo(close, volume, ew, hold)
        results[hold] = df
        print(summ(df, f"hold={hold}d"))

    print("\nby branch:")
    for hold, df in results.items():
        if df.empty:
            continue
        for br, g in df.groupby("branch"):
            print(summ(g, f"hold={hold}d {br}"))


if __name__ == "__main__":
    main()
