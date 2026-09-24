"""Refines the binary buyers_took_over test (close above/below the day's
own high-low midpoint) into a continuous position and buckets it into
thirds, to see whether a stricter "closed near the top of its range" test
carries real signal, and whether the bottom/middle thirds carry a
*negative* signal worth avoiding rather than just "no signal."

position[j] = (close[j] - low[j]) / (high[j] - low[j]); the original
buyers_took_over is exactly position > 0.5. Bucketed here into
bottom_third (<1/3), middle_third, top_third (>=2/3), no_range (high==low,
can't be computed) -- undefined for zero-range days rather than forced
into a bucket.

Population: reruns two_touch_low_v2_daybyday's decide()/run_ticker with
both volume thresholds set to -inf, so the OR gate is always satisfied
regardless of candle shape at L or L2 -- this yields every match of
Stage 1 (local min + supersession) and Stage 2 (close-tolerance), with NO
confirmation heuristic applied at all, the clean population to slice by
position after the fact. n_back, n_fwd, close_tolerance stay at their
swept values.

Also reports the "strict BTO" variant directly: position >= 2/3 required
at BOTH L and L2 (no volume alternative) -- just a filter on the same
ungated population, comparable to the earlier midpoint BTO-only variant
(two_touch_low_v2_bto_check.py: n=17, t=-0.83).

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_v2_position_terciles.py [seed] [k]
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

BASE_PARAMS = dict(
    n_back=5,
    n_fwd=3,
    volume_threshold_L=1.0,
    volume_threshold_L2=0.8,
    close_tolerance=0.005,
)
UNGATED_PARAMS = dict(BASE_PARAMS, volume_threshold_L=float("-inf"), volume_threshold_L2=float("-inf"))
DD_PCT = 0.01


def row(df, label):
    if df.empty or len(df) < 3:
        return dict(label=label, n=len(df), hit=np.nan, net=np.nan, excess=np.nan, t=np.nan)
    ex = df.excess_ret
    return dict(label=label, n=len(df), hit=(df.ret > 0).mean(), net=df.ret_net.mean(),
                excess=ex.mean(), t=ex.mean() / ex.std() * np.sqrt(len(df)))


def print_row(r):
    print(f"  {r['label']:24} n={r['n']:4d}  hit={r['hit']:.1%}  net={r['net']:+.3%}  "
          f"excess={r['excess']:+.3%}  t={r['t']:+.2f}")


def close_position(high_s, low_s, close_s, j):
    h, l, c = high_s.iloc[j], low_s.iloc[j], close_s.iloc[j]
    if pd.isna(h) or pd.isna(l) or pd.isna(c) or h == l:
        return None
    return (c - l) / (h - l)


def tercile(pos):
    if pos is None:
        return "no_range"
    if pos < 1 / 3:
        return "bottom_third"
    if pos < 2 / 3:
        return "middle_third"
    return "top_third"


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
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

    args = [(t, close[t], high[t], low[t], volume[t], spy_close[t], UNGATED_PARAMS, DD_PCT) for t in close]
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, args)
    df = pd.DataFrame([r for o in out for r in o])
    df = attach_benchmark(df, ew)
    print(f"ungated population (Stage 1 + Stage 2 only, no candle/volume confirmation): n={len(df)}\n")

    pos_L, pos_L2, terc_L, terc_L2 = [], [], [], []
    for _, tr in df.iterrows():
        t = tr.stock
        L = close[t].index.get_loc(pd.Timestamp(tr.low_date))
        L2 = close[t].index.get_loc(pd.Timestamp(tr.entry_date))
        pL = close_position(high[t], low[t], close[t], L)
        pL2 = close_position(high[t], low[t], close[t], L2)
        pos_L.append(pL); pos_L2.append(pL2)
        terc_L.append(tercile(pL)); terc_L2.append(tercile(pL2))
    df = df.assign(pos_L=pos_L, pos_L2=pos_L2, terc_L=terc_L, terc_L2=terc_L2)

    order = ["bottom_third", "middle_third", "top_third", "no_range"]
    print("=== bucketed by L2's (buy day) close position in its own high-low range ===")
    for g in order:
        sub = df[df.terc_L2 == g]
        if len(sub):
            print_row(row(sub, g))

    print("\n=== bucketed by L's (first trough) close position in its own high-low range ===")
    for g in order:
        sub = df[df.terc_L == g]
        if len(sub):
            print_row(row(sub, g))

    print("\n=== strict BTO variant: top_third required at BOTH L and L2 (no volume alternative) ===")
    strict = df[(df.terc_L == "top_third") & (df.terc_L2 == "top_third")]
    print_row(row(strict, "strict top-third gate"))
    print("  (compare to the midpoint BTO-only variant: n=17, hit=29.4%, net=-0.843%, t=-0.83)")


if __name__ == "__main__":
    main()
