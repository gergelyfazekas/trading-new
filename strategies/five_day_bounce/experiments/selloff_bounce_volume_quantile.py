"""Quantile-bucket the capitulation-volume gate on top of the selloff
trigger, instead of a naive threshold sweep. mean_reversion_notes.md's
"Reminder for later sweeps" (from the birth-volume-gate experiment)
found that threshold-sweeping a volume_ratio-style variable produces a
smooth-looking improvement curve purely from nested subsets (Spearman
rho=0.022, p=0.45 once corrected) -- not a real dose-response. Equal-
sized, DISJOINT quantile buckets avoid that: each bucket is a separate
population, not a superset/subset of its neighbors.

Trigger held fixed (price-only, ungated on volume): n_window=10,
z_thresh=2.5 -- a middling point on selloff_bounce_sweep.py's grid,
chosen for trade count (n~3800 on the 20-ticker prototype), not for its
(negative) raw excess. decide() already reports vr (volume_ratio at the
trigger day) on every decision without gating on it, so the population
here is exactly the same one selloff_bounce_sweep.py already rejected on
price alone -- this asks only "does slicing that same population by
volume reveal a subset with real edge," same style as
two_touch_low_v2_position_terciles.py's after-the-fact slicing.

Run: ./venv/bin/python strategies/five_day_bounce/experiments/selloff_bounce_volume_quantile.py
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
from selloff_bounce_daybyday import run_ticker, rolling_z, HOLD, COST, WARMUP

SEED, K = 21, 20
TRIGGER_PARAMS = dict(n_window=10, z_thresh=2.5, spy_z_thresh=1.0, spy_neutral=0.3)
N_BUCKETS = 5


def main():
    tickers = random.Random(SEED).sample(LIVE_TICKERS, K)
    series, failed = pull_all(tickers + ["SPY"])
    today = datetime.date.today()

    spy_c, _ = series["SPY"]
    spy_c = spy_c.copy()
    spy_c.index = pd.to_datetime(spy_c.index)
    spy_c = spy_c[spy_c.index.date < today]
    spy_z_full = rolling_z(spy_c, TRIGGER_PARAMS["n_window"])

    close, volume = {}, {}
    for t in tickers:
        if t not in series:
            continue
        c, v = series[t]
        c = c.copy()
        c.index = pd.to_datetime(c.index)
        close[t] = c[c.index.date < today]
        v = v.copy()
        v.index = pd.to_datetime(v.index)
        volume[t] = v.reindex(close[t].index)

    ew = equal_weight_curve(close)
    spy_z_aligned = {t: spy_z_full.reindex(close[t].index, method="ffill") for t in close}

    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, [(t, close[t], volume[t], spy_z_aligned[t], TRIGGER_PARAMS, HOLD) for t in close])
    df = pd.DataFrame([tr for o in out for tr in o])
    df = df.dropna(subset=["vr"])
    df = attach_benchmark(df, ew)
    print(f"fixed trigger population: n={len(df)} (vr known for {len(df)} of them)\n")
    print(summ(df, "ALL (ungated on volume)"))

    df["vr_bucket"] = pd.qcut(df.vr, N_BUCKETS, labels=[f"Q{i+1}" for i in range(N_BUCKETS)])
    print("\nby volume_ratio[trigger day] quantile bucket (Q1=lowest volume, Q5=highest):")
    for b in [f"Q{i+1}" for i in range(N_BUCKETS)]:
        g = df[df.vr_bucket == b]
        lo, hi = g.vr.min(), g.vr.max()
        print(summ(g, f"{b} [{lo:.2f}-{hi:.2f}]"))

    # Spearman correlation, vr vs excess return -- the direct dose-response check
    from scipy.stats import spearmanr
    rho, pval = spearmanr(df.vr, df.excess_ret)
    print(f"\nSpearman(vr, excess_ret): rho={rho:+.3f}  p={pval:.3f}  n={len(df)}")

    print("\nsame breakdown, split by branch:")
    for br in ["market_wide", "idiosyncratic"]:
        sub = df[df.branch == br]
        if sub.empty:
            continue
        print(f"\n  -- {br} (n={len(sub)}) --")
        for b in [f"Q{i+1}" for i in range(N_BUCKETS)]:
            g = sub[sub.vr_bucket == b]
            if len(g) < 3:
                continue
            print(" ", summ(g, b))


if __name__ == "__main__":
    main()
