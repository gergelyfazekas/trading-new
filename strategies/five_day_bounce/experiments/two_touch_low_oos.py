"""Generalization check for the retest_gap=3 finding: same locked entry
params, same 5-day hold, but on a FRESH cross-section of tickers -- every
LIVE_TICKERS name NOT in the original seed=21/k=20 sample every other
two_touch_low_* script has used so far. Never touched while choosing
retest_gap=3 over 2, so this is the decisive check, not another in-sample
variant (see validation-first-quant-work: "the decisive check is a fresh
cross-section, not a held-out time window").

Runs both retest_gap=2 (locked) and retest_gap=3 (the candidate) on this
fresh set for contrast.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_oos.py
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
from two_touch_low_daybyday import PARAMS, HOLD, run_ticker

ORIGINAL_SEED, ORIGINAL_K = 21, 20


def run_combo(close, volume, ew, retest_gap):
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, [(t, close[t], volume[t], PARAMS, HOLD, retest_gap) for t in close])
    df = pd.DataFrame([t for o in out for t in o])
    if df.empty:
        return df
    return attach_benchmark(df, ew)


def main():
    original = set(random.Random(ORIGINAL_SEED).sample(LIVE_TICKERS, ORIGINAL_K))
    fresh = sorted(set(LIVE_TICKERS) - original)
    print(f"original in-sample set (excluded, n={len(original)}): {sorted(original)}")
    print(f"fresh OOS set (n={len(fresh)}): {fresh}\n")

    series, failed = pull_all(fresh)
    if failed:
        print(f"no data for: {failed}")
    today = datetime.date.today()
    close, volume = {}, {}
    for t in fresh:
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

    for gap in (2, 3):
        df = run_combo(close, volume, ew, gap)
        print(f"--- OOS, retest_gap={gap} ---")
        print(summ(df, "ALL"))
        if not df.empty:
            for br, g in df.groupby("branch"):
                print(summ(g, br))
            rr = df[df.branch == "rebound_retest"]
            if len(rr) >= 3:
                rng = np.random.default_rng(0)
                rows = []
                for t, cnt in rr.groupby("stock").size().items():
                    c = close[t]
                    for i in rng.choice(np.arange(30, len(c) - HOLD), size=cnt, replace=False):
                        rows.append({"entry_date": c.index[i], "exit_date": c.index[i + HOLD],
                                     "ret_net": c.iloc[i + HOLD] / c.iloc[i] - 1 - 10 / 1e4})
                pe = attach_benchmark(pd.DataFrame(rows), ew).excess_ret.mean()
                print(f"  {'placebo excess (matched n, rebound_retest)':40} {pe:+.3%}   "
                      f"causal lift = {rr.excess_ret.mean() - pe:+.3%}")
        print()


if __name__ == "__main__":
    main()
