"""Relaxed rebound_retest variant: loosen the two tightest gates found in
the funnel breakdown (2026-09-22 session) -- (a) rebound_volume_threshold
(currently requires the rebound day's volume < 1.0x its trailing average;
survival 46.0% of candidates that reach it) and (b) the retest's upper
bound at close[L+1] (survival 27.3%, the single tightest gate in the
chain). decide() itself is untouched -- this is a separate, explicitly
experimental function so nothing here can accidentally change the
validated live rule.

Small-subset prototype only (default: the same 20-ticker seed=21 sample
used throughout this candidate's testing), before deciding whether it's
worth a full-101 run.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_relaxed_variant.py [seed] [n_tickers]
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
from two_touch_low_daybyday import PARAMS, HOLD, COST, WARMUP, volume_ratio

RETEST_GAP = 2


def decide_relaxed(close, volume, L, p, rebound_volume_threshold, cap_retest_at_rebound):
    """Same Stage 1 as decide(); Stage 2's bigger-rebound branch relaxed:
    rebound_volume_threshold is a separate (looser) value than the locked
    spec's 1.0, and cap_retest_at_rebound=False drops the close[L2] <=
    close[L+1] upper bound entirely (only close[L2] >= close[L] still
    required). flat branch is unchanged and not returned here, since this
    script only cares about the branch actually traded."""
    n = len(close)
    if L < p["lookback"] or L + 1 >= n:
        return None
    window = close.iloc[L - p["lookback"]:L]
    if not bool((close.iloc[L] < window).all()):
        return None
    if not (close.iloc[L - 1] >= close.iloc[L] * (1 + p["prominence_pct"])):
        return None
    vr_L = volume_ratio(volume, L)
    if not (vr_L > p["volume_threshold"]):
        return None

    c_L = float(close.iloc[L])
    c_L1 = float(close.iloc[L + 1])
    pct1 = (c_L1 - c_L) / c_L
    if not (pct1 > p["k_pct"]) or pct1 > p["rebound_max_pct"]:
        return None  # only the bigger-rebound branch, same size band as locked spec

    vr_L1 = volume_ratio(volume, L + 1)
    if not (vr_L1 < rebound_volume_threshold):
        return None

    L2 = L + RETEST_GAP
    if L2 >= n:
        return None
    c_L2 = float(close.iloc[L2])
    if not (c_L2 >= c_L):
        return None
    if cap_retest_at_rebound and not (c_L2 <= c_L1):
        return None

    vr_L2 = volume_ratio(volume, L2)
    if not (vr_L2 > p["volume_threshold_buy"]):
        return None
    return dict(offset=RETEST_GAP, pct1=pct1, vr_L=vr_L, vr_L1=vr_L1, vr_L2=vr_L2)


def run_ticker(args):
    ticker, close, volume, p, rebound_vol_thr, cap_retest = args
    idx, n = close.index, len(close)
    busy, trades = -1, []
    for L in range(WARMUP, n - HOLD - RETEST_GAP - 1):
        decision = decide_relaxed(close, volume, L, p, rebound_vol_thr, cap_retest)
        if decision is None:
            continue
        buy_day = L + decision["offset"]
        if buy_day <= busy:
            continue
        exit_i = buy_day + HOLD
        if exit_i >= n:
            continue
        ret = float(close.iloc[exit_i] / close.iloc[buy_day] - 1)
        trades.append(dict(stock=ticker, low_date=idx[L], entry_date=idx[buy_day], exit_date=idx[exit_i],
                            pct1=decision["pct1"], vr_L1=decision["vr_L1"],
                            ret=ret, ret_net=ret - COST))
        busy = exit_i
    return trades


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
        c = c.copy(); c.index = pd.to_datetime(c.index)
        mask = c.index.date < today
        close[t] = c[mask]
        if v is not None:
            v = v.copy(); v.index = pd.to_datetime(v.index)
            volume[t] = v.reindex(close[t].index)
        else:
            volume[t] = pd.Series(np.nan, index=close[t].index)
    ew = equal_weight_curve(close)

    variants = [
        ("locked (baseline)", 1.0, True),
        ("volume<1.3x, capped", 1.3, True),
        ("volume<1.5x, capped", 1.5, True),
        ("volume<1.0x, uncapped", 1.0, False),
        ("volume<1.3x, uncapped", 1.3, False),
        ("volume<1.5x, uncapped", 1.5, False),
    ]
    for label, vol_thr, cap in variants:
        args = [(t, close[t], volume[t], PARAMS, vol_thr, cap) for t in close]
        with Pool(min(len(close), cpu_count())) as pool:
            out = pool.map(run_ticker, args)
        df = pd.DataFrame([row for o in out for row in o])
        if df.empty:
            print(f"  {label:24} no trades")
            continue
        df = attach_benchmark(df, ew)
        print(summ(df, label))


if __name__ == "__main__":
    main()
