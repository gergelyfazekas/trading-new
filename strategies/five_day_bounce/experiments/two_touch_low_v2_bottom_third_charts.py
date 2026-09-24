"""Charts for two_touch_low_v2_position_terciles.py's bottom-third finding
-- samples signals where L, L2, or both closed in the bottom third of
their own day's high-low range (the bucket that outperformed), drawn from
the same ungated population (Stage 1 + Stage 2 only, no candle/volume
gate) so nothing here is filtered by whichever confirmation heuristic
ends up chosen.

Reuses two_touch_low_v2_charts.py's plot_signal() as-is (same markers,
tolerance band, OHLC whiskers, params box) -- the gate label on each chart
still reports whether the REAL volume/candle thresholds (BASE_PARAMS)
would have passed, so you can see at a glance which of these the current
combined-best combo would actually have traded.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_v2_bottom_third_charts.py [n_per_bucket] [seed] [k]
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
from two_touch_low_v2_daybyday import run_ticker
from two_touch_low_v2_position_terciles import UNGATED_PARAMS, close_position, tercile
from two_touch_low_v2_charts import plot_signal, BEST_PARAMS as BASE_PARAMS

CHART_DIR = os.path.join(HERE, "data", "two_touch_low_v2_bottom_third_charts")


def main():
    n_per_bucket = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}\n", flush=True)

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

    args = [(t, close[t], high[t], low[t], volume[t], spy_close[t], UNGATED_PARAMS, 0.01) for t in close]
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, args)
    df = pd.DataFrame([r for o in out for r in o])
    df = attach_benchmark(df, ew)

    terc_L, terc_L2 = [], []
    for _, tr in df.iterrows():
        t = tr.stock
        L = close[t].index.get_loc(pd.Timestamp(tr.low_date))
        L2 = close[t].index.get_loc(pd.Timestamp(tr.entry_date))
        terc_L.append(tercile(close_position(high[t], low[t], close[t], L)))
        terc_L2.append(tercile(close_position(high[t], low[t], close[t], L2)))
    df = df.assign(terc_L=terc_L, terc_L2=terc_L2)

    both = df[(df.terc_L == "bottom_third") & (df.terc_L2 == "bottom_third")]
    l_only = df[(df.terc_L == "bottom_third") & (df.terc_L2 != "bottom_third")]
    l2_only = df[(df.terc_L != "bottom_third") & (df.terc_L2 == "bottom_third")]
    print(f"both bottom_third: n={len(both)}  L-only: n={len(l_only)}  L2-only: n={len(l2_only)}\n")

    os.makedirs(CHART_DIR, exist_ok=True)
    picked = []
    for label, bucket in (("both", both), ("L_only", l_only), ("L2_only", l2_only)):
        sample = bucket.sample(min(n_per_bucket, len(bucket)), random_state=seed) if len(bucket) else bucket
        for _, row in sample.iterrows():
            picked.append((label, row))

    for label, row in picked:
        t = row.stock
        out_path = os.path.join(CHART_DIR, f"{label}_{t}_{row.entry_date.date()}_v2.png")
        plot_signal(t, close[t], high[t], low[t], volume[t], row, BASE_PARAMS, out_path)
        print(f"  [{label:8}] {t} {row.entry_date.date()} (ret_net={row.ret_net:+.2%}) -> {out_path}")

    print(f"\nsaved {len(picked)} chart(s) to {CHART_DIR}")


if __name__ == "__main__":
    main()
