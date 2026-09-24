"""One-shot param sweep for selloff_bounce_daybyday.py's trigger, on the
same 20-ticker seed=21 prototype -- pulls data once, then loops over
(n_window, z_thresh) so tightening the threshold doesn't cost a re-pull
per point. Quantile-style sweep isn't needed here (this isn't the
volume-ratio-style variable the birth-volume-gate warning in
mean_reversion_notes.md applies to) -- z_thresh is a magnitude cutoff on
a single continuous score, a plain grid is fine.

Run: ./venv/bin/python strategies/five_day_bounce/experiments/selloff_bounce_sweep.py
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
from selloff_bounce_daybyday import run_ticker, HOLD, COST, WARMUP

SEED, K = 21, 20


def main():
    tickers = random.Random(SEED).sample(LIVE_TICKERS, K)
    series, failed = pull_all(tickers + ["SPY"])
    today = datetime.date.today()

    from selloff_bounce_daybyday import rolling_z

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

    rng = np.random.default_rng(0)
    placebo_idx = {}
    for t, c in close.items():
        placebo_idx[t] = rng.choice(np.arange(WARMUP, len(c) - HOLD), size=200, replace=True)

    grid = [
        dict(n_window=10, z_thresh=2.0, spy_z_thresh=1.0, spy_neutral=0.3),
        dict(n_window=10, z_thresh=2.5, spy_z_thresh=1.0, spy_neutral=0.3),
        dict(n_window=10, z_thresh=3.0, spy_z_thresh=1.0, spy_neutral=0.3),
        dict(n_window=20, z_thresh=2.0, spy_z_thresh=1.0, spy_neutral=0.3),
        dict(n_window=20, z_thresh=2.5, spy_z_thresh=1.0, spy_neutral=0.3),
        dict(n_window=20, z_thresh=3.0, spy_z_thresh=1.0, spy_neutral=0.3),
        dict(n_window=30, z_thresh=2.5, spy_z_thresh=1.0, spy_neutral=0.3),
        dict(n_window=30, z_thresh=3.0, spy_z_thresh=1.0, spy_neutral=0.3),
    ]

    print(f"{'n_win':>6} {'z_thr':>6} {'n':>6} {'hit':>7} {'excess':>9} {'t':>7}  {'mw_n':>6} {'mw_t':>7}  {'idio_n':>6} {'idio_t':>7}  {'placebo':>9} {'lift':>9}")
    for p in grid:
        spy_c, _ = series["SPY"]
        spy_c = spy_c.copy()
        spy_c.index = pd.to_datetime(spy_c.index)
        spy_c = spy_c[spy_c.index.date < today]
        spy_z_full = rolling_z(spy_c, p["n_window"])
        spy_z_aligned = {t: spy_z_full.reindex(close[t].index, method="ffill") for t in close}

        with Pool(min(len(close), cpu_count())) as pool:
            out = pool.map(run_ticker, [(t, close[t], volume[t], spy_z_aligned[t], p, HOLD) for t in close])
        df = pd.DataFrame([tr for o in out for tr in o])
        if df.empty:
            print(f"{p['n_window']:>6} {p['z_thresh']:>6.1f}   no trades")
            continue
        df = attach_benchmark(df, ew)

        rows = []
        for t, cnt in df.groupby("stock").size().items():
            c = close[t]
            idxs = rng.choice(np.arange(WARMUP, len(c) - HOLD), size=cnt, replace=False)
            for i in idxs:
                rows.append({"entry_date": c.index[i], "exit_date": c.index[i + HOLD],
                             "ret_net": c.iloc[i + HOLD] / c.iloc[i] - 1 - COST})
        pe = attach_benchmark(pd.DataFrame(rows), ew).excess_ret.mean()

        ex = df.excess_ret
        t_stat = ex.mean() / ex.std() * np.sqrt(len(df))
        mw = df[df.branch == "market_wide"].excess_ret
        idio = df[df.branch == "idiosyncratic"].excess_ret
        mw_t = mw.mean() / mw.std() * np.sqrt(len(mw)) if len(mw) > 2 else float("nan")
        idio_t = idio.mean() / idio.std() * np.sqrt(len(idio)) if len(idio) > 2 else float("nan")

        print(f"{p['n_window']:>6} {p['z_thresh']:>6.1f} {len(df):>6} {(df.ret>0).mean():>7.1%} "
              f"{ex.mean():>+9.3%} {t_stat:>+7.2f}  {len(mw):>6} {mw_t:>+7.2f}  {len(idio):>6} {idio_t:>+7.2f}  "
              f"{pe:>+9.3%} {ex.mean()-pe:>+9.3%}")


if __name__ == "__main__":
    main()
