"""Fresh-cross-section check for the day-by-day market-drawdown exit
(two_touch_low_daybyday_dd_exit.py): same causal loop, same 81 tickers
never touched while tuning params, choosing retest_gap, or choosing
dd_pct -- the decisive check per validation-first-quant-work.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_dd_exit_oos.py [dd_pct]
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
from two_touch_low_daybyday import PARAMS
from two_touch_low_daybyday_dd_exit import run_ticker

ORIGINAL_SEED, ORIGINAL_K = 21, 20


def main():
    dd_pct = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    original = set(random.Random(ORIGINAL_SEED).sample(LIVE_TICKERS, ORIGINAL_K))
    fresh = sorted(set(LIVE_TICKERS) - original)
    print(f"fresh OOS set, n={len(fresh)} (never touched while tuning params, retest_gap, or dd_pct={dd_pct:.0%})\n",
          flush=True)

    series, failed = pull_all(fresh + ["SPY"])
    if failed:
        print(f"no data for: {failed}")
    today = datetime.date.today()
    close, volume = {}, {}
    for t in fresh:
        if t not in series:
            continue
        c, v = series[t]
        c = c.copy(); c.index = pd.to_datetime(c.index)
        mask = c.index.date < today
        close[t] = c[mask]
        if v is not None:
            v = v.copy(); v.index = pd.to_datetime(v.index)
            volume[t] = v.reindex(close[t].index)
        else:
            volume[t] = pd.Series(np.nan, index=close[t].index)
    ew = equal_weight_curve(close)
    spy_close, _ = series["SPY"]
    spy_close = spy_close.copy(); spy_close.index = pd.to_datetime(spy_close.index)

    for pct, label in [(1.0, "no early exit"), (dd_pct, f"market-dd exit {dd_pct:.0%}")]:
        args = [(t, close[t], volume[t], spy_close.reindex(close[t].index, method="ffill"), PARAMS, pct)
                for t in close]
        with Pool(min(len(close), cpu_count())) as pool:
            out = pool.map(run_ticker, args)
        df = pd.DataFrame([row for o in out for row in o])
        df = attach_benchmark(df, ew)
        print(summ(df, label))
        if pct < 1.0:
            print(f"  exit reasons: {df.exit_reason.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
