"""Day-by-day (causal) test of notes' candidate #3 (streak-based
reversal) applied to the "big selloff" premise: instead of a drawdown-
depth z-score vs. the recent high (Phase 0/1, tested and rejected --
see mean_reversion_notes.md, "Mean reversion after a big selloff"), size
the selloff by the current consecutive-down-day streak, z-scored against
that ticker's own trailing streak-severity distribution. Independent
trigger definition; everything else (SPY market-wide/idiosyncratic
split, buy timing, exit, cost) is held identical to Phase 0/1 for a
clean comparison.

Streak, computed causally (backward-only, current day included since
close[i] is known at day i's own close):

  down[i]        = close[i] < close[i-1]
  streak_len[i]  = length of the consecutive run of down[i]==True ending
                   at i (0 if today is not a down day)
  streak_decline[i] = (close[i] - close[i-streak_len[i]]) / close[i-streak_len[i]]
                   -- cumulative % move from the close right before the
                   streak started to today's close (<=0 during a streak)

Two modes for "severity" (both requested in the notes' candidate #3 --
"length OR cumulative decline"), tested side by side rather than picking
one:

  mode="length"   severity[i] = streak_len[i]
  mode="decline"  severity[i] = -streak_decline[i]   (positive magnitude)

severity is then z-scored against its own trailing dist_window-day
history (shift(1) before rolling, so the baseline excludes today --
same causal construction as Phase 0's rolling_z):

  z[i] = (severity[i] - mean(severity[i-dist_window:i])) / std(severity[i-dist_window:i])

Fires when z[i] >= z_thresh (streak longer, or decline deeper, than
usual for this specific ticker). Buy at close[i], fixed 5-day hold,
10bps cost, one position per ticker -- identical convention to Phase 0/1.
SPY market-wide/idiosyncratic split reuses Phase 0/1's rolling_z (on
SPY's own close) unchanged. volume_ratio[i] reported ungated, same
reasoning as Phase 0/1 (quantile-bucket later, don't threshold-sweep).

Causality verified by truncation, same convention as the rest of this
file's history: every fired trigger is recomputed on data sliced to (and
including) its own buy day and must reproduce the identical z-score.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/selloff_bounce_streak_daybyday.py [seed] [n_tickers]
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
from selloff_bounce_daybyday import rolling_z, volume_ratio, HOLD, COST

PARAMS = dict(
    dist_window=120,     # trailing window for severity's own mean/std
    z_thresh=2.0,        # severity must be at least this many std devs above its own norm
    mode="decline",       # "length" or "decline"
    n_window=10,          # SPY's own drawdown-depth z-score window (Phase 0/1's rolling_z)
    spy_z_thresh=1.0,
    spy_neutral=0.3,
)
WARMUP_FLOOR = 30


def compute_severity_z(close, dist_window, mode):
    """Returns (z, streak_len, streak_decline) Series, all causal by
    construction: severity[i] depends only on close[..i], and z[i]'s
    baseline (mean/std) is shift(1)'d before rolling, so it excludes
    today too."""
    down = close < close.shift(1)
    grp = (down != down.shift()).cumsum()
    streak_len = down.groupby(grp).cumcount() + 1
    streak_len = streak_len.where(down, 0).astype(float)

    n = len(close)
    cvals = close.values
    slen = streak_len.values.astype(int)
    idx = np.arange(n) - slen
    valid = (slen > 0) & (idx >= 0)
    base = np.full(n, np.nan)
    base[valid] = cvals[idx[valid]]
    with np.errstate(invalid="ignore", divide="ignore"):
        decline_vals = (cvals - base) / base
    decline_vals = np.where(valid, decline_vals, 0.0)
    streak_decline = pd.Series(decline_vals, index=close.index)

    severity = streak_len if mode == "length" else -streak_decline
    prior = severity.shift(1)
    roll_mean = prior.rolling(dist_window).mean()
    roll_std = prior.rolling(dist_window).std()
    z = (severity - roll_mean) / roll_std
    return z, streak_len, streak_decline


def decide(close, volume, spy_z_aligned, i, p):
    """Evaluate the streak-based selloff trigger at position i, using
    only data up to and including i. Returns None or a dict with the
    branch."""
    warmup = max(WARMUP_FLOOR, p["dist_window"] + 5)
    if i < warmup:
        return None
    z, streak_len, streak_decline = compute_severity_z(close.iloc[:i + 1], p["dist_window"], p["mode"])
    zi = z.iloc[i]
    if pd.isna(zi) or not (zi >= p["z_thresh"]):
        return None
    szpy = spy_z_aligned.iloc[i]
    if pd.isna(szpy):
        return None
    if szpy <= -p["spy_z_thresh"]:
        branch = "market_wide"
    elif szpy > -p["spy_neutral"]:
        branch = "idiosyncratic"
    else:
        return None  # mixed zone, excluded from both buckets
    vr = volume_ratio(volume, i)
    return dict(branch=branch, z=float(zi), spy_z=float(szpy), vr=vr,
                streak_len=float(streak_len.iloc[i]), streak_decline=float(streak_decline.iloc[i]))


def run_ticker(args):
    ticker, close, volume, spy_z_aligned, p, hold = args
    n = len(close)
    busy, trades = -1, []
    warmup = max(WARMUP_FLOOR, p["dist_window"] + 5)
    for i in range(warmup, n - hold):
        decision = decide(close, volume, spy_z_aligned, i, p)
        if decision is None:
            continue
        if i <= busy:
            continue

        trunc_close = close.iloc[:i + 1]
        trunc_volume = volume.iloc[:i + 1]
        trunc_spy = spy_z_aligned.iloc[:i + 1]
        redecision = decide(trunc_close, trunc_volume, trunc_spy, i, p)
        assert redecision == decision, f"leakage: {ticker} i={close.index[i].date()} {decision} vs {redecision}"

        exit_i = i + hold
        if exit_i >= n:
            continue
        ret = float(close.iloc[exit_i] / close.iloc[i] - 1)
        trades.append(dict(
            stock=ticker, entry_date=close.index[i], exit_date=close.index[exit_i],
            branch=decision["branch"], z=decision["z"], spy_z=decision["spy_z"], vr=decision["vr"],
            streak_len=decision["streak_len"], streak_decline=decision["streak_decline"],
            ret=ret, ret_net=ret - COST,
        ))
        busy = exit_i
    return trades


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}", flush=True)
    print(f"params: {PARAMS}\n", flush=True)

    series, failed = pull_all(tickers + ["SPY"])
    if failed:
        print(f"no data for: {failed}")
    today = datetime.date.today()

    spy_c, _ = series["SPY"]
    spy_c = spy_c.copy()
    spy_c.index = pd.to_datetime(spy_c.index)
    spy_c = spy_c[spy_c.index.date < today]
    spy_z_full = rolling_z(spy_c, PARAMS["n_window"])

    close, volume = {}, {}
    for t in tickers:
        if t not in series:
            continue
        c, v = series[t]
        c = c.copy()
        c.index = pd.to_datetime(c.index)
        close[t] = c[c.index.date < today]
        if v is not None:
            v = v.copy()
            v.index = pd.to_datetime(v.index)
            volume[t] = v.reindex(close[t].index)
        else:
            volume[t] = pd.Series(np.nan, index=close[t].index)

    ew = equal_weight_curve(close)
    spy_z_aligned = {t: spy_z_full.reindex(close[t].index, method="ffill") for t in close}

    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, [(t, close[t], volume[t], spy_z_aligned[t], PARAMS, HOLD) for t in close])
    df = pd.DataFrame([t for o in out for t in o])
    if df.empty:
        print("no trades fired at these parameters")
        return
    df = attach_benchmark(df, ew)
    df.to_csv(os.path.join(HERE, "selloff_bounce_streak_daybyday_trades.csv"), index=False)

    rng = np.random.default_rng(0)
    rows = []
    for t, cnt in df.groupby("stock").size().items():
        c = close[t]
        for i in rng.choice(np.arange(WARMUP_FLOOR, len(c) - HOLD), size=cnt, replace=False):
            rows.append({"entry_date": c.index[i], "exit_date": c.index[i + HOLD],
                         "ret_net": c.iloc[i + HOLD] / c.iloc[i] - 1 - COST})
    pe = attach_benchmark(pd.DataFrame(rows), ew).excess_ret.mean()

    print(f"{len(close)} tickers, ~{min(len(c) for c in close.values())}+ sessions each\n")
    print("every trade's entry decision verified identical on truncated-to-buy-day data (causal by construction)\n")
    print(summ(df, "ALL (causal)"))
    print(f"  {'placebo excess (matched n)':22} {pe:+.3%}   causal lift = {df.excess_ret.mean() - pe:+.3%}")

    print("\nby branch (market_wide vs idiosyncratic):")
    for b, g in df.groupby("branch"):
        print(summ(g, b))

    print("\nby year:")
    for y, g in df.groupby(pd.DatetimeIndex(df.entry_date).year):
        print(summ(g, str(y)))

    print("\nper ticker:")
    for t in close:
        print(summ(df[df.stock == t], t))


if __name__ == "__main__":
    main()
