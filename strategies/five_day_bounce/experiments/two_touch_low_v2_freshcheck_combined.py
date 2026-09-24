"""Out-of-sample check of the actual combined-best combo (the real
OR-gated rule -- buyers_took_over OR volume_ratio at both L and L2, not
the ungated diagnostic population used for the tercile/volume-cross
checks) on the 81 LIVE_TICKERS never touched while tuning any of this,
same "fresh-81" discipline as the v1 rule's validation and as
two_touch_low_v2_l2_volume_freshcheck.py.

params: two_touch_low_v2_charts.BEST_PARAMS (n_back=5, n_fwd=3,
volume_threshold_L=1.0, volume_threshold_L2=0.8, close_tolerance=0.005),
the same combo two_touch_low_v2_sweep.py's one-at-a-time results
recommended on the seed=21/k=20 in-sample set (n=182, hit=53.8%,
net=+0.506%, excess=+0.294%, t=+1.50).

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_v2_freshcheck_combined.py
"""
import datetime
import os
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
from two_touch_low_v2_daybyday import run_ticker
from two_touch_low_v2_charts import BEST_PARAMS

DD_PCT = 0.01
IN_SAMPLE_20 = ["KO", "GILD", "EMR", "UPS", "MRK", "CRM", "XOM", "ORCL", "TXN", "COST",
                "QCOM", "MU", "HON", "APD", "MMM", "ABT", "TGT", "BLK", "CVS", "AXP"]


def main():
    tickers = [t for t in LIVE_TICKERS if t not in IN_SAMPLE_20]
    print(f"fresh {len(tickers)} tickers (LIVE_TICKERS minus the seed=21/k=20 in-sample set)")
    print(f"params: {BEST_PARAMS}, dd_pct={DD_PCT}\n", flush=True)

    series, failed = pull_all(tickers + ["SPY"], include_hl=True)
    if failed:
        print(f"no data for: {failed}")
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

    args = [(t, close[t], high[t], low[t], volume[t], spy_close[t], BEST_PARAMS, DD_PCT) for t in close]
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, args)
    df = pd.DataFrame([r for o in out for r in o])
    if df.empty:
        print("no trades fired on the fresh-81 set at this combo")
        return
    df = attach_benchmark(df, ew)
    df.to_csv(os.path.join(HERE, "two_touch_low_v2_freshcheck_combined_trades.csv"), index=False)

    print(f"{len(close)} tickers, ~{min(len(c) for c in close.values())}+ sessions each\n")
    print(summ(df, "fresh-81 (combined-best)"))
    print("  [in-sample: n=182, hit=53.8%, net=+0.506%, excess=+0.294%, t=+1.50]\n")
    print(f"exit reasons: {df.exit_reason.value_counts().to_dict()}\n")

    print("by year:")
    for y, g in df.groupby(pd.DatetimeIndex(df.entry_date).year):
        print(summ(g, str(y)))

    print("\nper ticker:")
    for t in sorted(close):
        sub = df[df.stock == t]
        if len(sub):
            print(summ(sub, t))


if __name__ == "__main__":
    main()
