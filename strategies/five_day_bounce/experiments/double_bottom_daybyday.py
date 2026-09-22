"""Day-by-day (fully causal) test of the "big fall, then small retest" pattern.

Pattern (A, B), evaluated on day i using only close[:i+1]:
  A  a trough with >= 1% log prominence (find_peaks on -log close, distance=1)
  B  a plain local low, 2-3 sessions after A: close[b] <= close[b-1] and
     close[b+1] > close[b] (so B is confirmed by one higher close, b+1 <= i)
  drop_A = log(max close over the K=10 sessions before A / close[A]) >= 1.5%
  drop_B = log(max close between A and B / close[B])  <  drop_A
  price:  'hold'   -> A <= B < A*(1+0.8%)       (retest holds, higher/equal low)
          'within' -> |B/A - 1| < 0.8%           (also allows a slight undercut)
  band = [min(A,B), max(A,B)], born on B's date, must be unbroken
         (tech_levels.mark_broken) as of day i.
Entry on day i: level age < 5 calendar days and close within 1% above band top.
Exit: fixed 5-session hold (no resistance exit -- trough-only levels never
produced one in the earlier trough test). 10 bps cost, one position/ticker.
Benchmark: equal-weight of the tested tickers; placebo: random entries with the
same per-ticker trade count.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/double_bottom_daybyday.py
"""
import datetime
import os
import random
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from tech_levels import Level, mark_broken
from tech_level_naive_strategy import equal_weight_curve, attach_benchmark
from tech_level_live import pull_all
from tech_level_continuation_live import LIVE_TICKERS
from daybyday_variants import summ, NEAR, HOLD, MAX_AGE, WARMUP

K, MIN_DROP_A, WIDTH, WINDOW = 10, 0.015, 0.008, 250
VARIANTS = ("hold", "within")


def find_pattern(close, i, variant):
    """Most recent qualifying (A, B) pair as of day i, or None. Returns Level + info."""
    win = close.iloc[max(0, i + 1 - WINDOW):i + 1]
    off = i + 1 - len(win)
    v = win.to_numpy()
    lc = np.log(v)
    troughs, _ = find_peaks(-lc, prominence=0.01, distance=1)
    troughs = set(int(t) + off for t in troughs)
    c = close.to_numpy()
    for b in range(i - 1, max(i - 6, K + 3), -1):          # B confirmed by close[b+1], b+1 <= i
        if not (c[b] <= c[b - 1] and c[b + 1] > c[b]):
            continue
        for a in (b - 2, b - 3):
            if a not in troughs or a < K:
                continue
            drop_a = np.log(c[a - K:a].max() / c[a])
            drop_b = np.log(c[a + 1:b].max() / c[b])
            if drop_a < MIN_DROP_A or drop_b >= drop_a:
                continue
            ratio = c[b] / c[a] - 1
            ok = (0 <= ratio < WIDTH) if variant == "hold" else (abs(ratio) < WIDTH)
            if ok:
                lvl = Level(band=(min(c[a], c[b]), max(c[a], c[b])), birth_date=close.index[b])
                return lvl, drop_a, drop_b
    return None


def run_ticker(ticker, close):
    idx, n = close.index, len(close)
    busy = {v: -1 for v in VARIANTS}
    trades = []
    for i in range(WARMUP, n - HOLD):
        date, price = idx[i], float(close.iloc[i])
        for v in VARIANTS:
            if busy[v] >= i:
                continue
            found = find_pattern(close, i, v)
            if found is None:
                continue
            lvl, drop_a, drop_b = found
            mark_broken([lvl], close.iloc[:i + 1])
            if lvl.broken_at is not None:
                continue
            dist = (price - lvl.band[1]) / price
            age_days = (date - pd.Timestamp(lvl.birth_date)).days
            if not (0 <= dist <= NEAR and age_days < MAX_AGE):
                continue
            ret = float(close.iloc[i + HOLD] / price - 1)
            trades.append(dict(variant=v, stock=ticker, entry_date=date, exit_date=idx[i + HOLD],
                               ret=ret, ret_net=ret - 10 / 1e4, drop_a=drop_a, drop_b=drop_b))
            busy[v] = i + HOLD
    return trades


def placebo_excess(close, ew, trades, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t, k in trades.groupby("stock").size().items():
        c = close[t]
        starts = rng.choice(np.arange(WARMUP, len(c) - HOLD), size=k, replace=False)
        rows += [{"entry_date": c.index[i], "exit_date": c.index[i + HOLD],
                  "ret_net": c.iloc[i + HOLD] / c.iloc[i] - 1 - 10 / 1e4} for i in starts]
    return attach_benchmark(pd.DataFrame(rows), ew).excess_ret.mean()


def run_group(tickers, label):
    series, _ = pull_all(tickers)
    today = datetime.date.today()
    close = {}
    for t in tickers:
        if t in series:
            c = series[t][0].copy()
            c.index = pd.to_datetime(c.index)
            close[t] = c[c.index.date < today]
    ew = equal_weight_curve(close)
    rows = []
    for t, c in close.items():
        rows += run_ticker(t, c)
    df = attach_benchmark(pd.DataFrame(rows), ew)
    df.to_csv(os.path.join(HERE, f"double_bottom_trades_{label}.csv"), index=False)
    print(f"\n=== group {label}: {', '.join(close)} ===")
    for v in VARIANTS:
        d = df[df.variant == v]
        print(summ(d, v))
        if len(d) >= 3:
            pe = placebo_excess(close, ew, d)
            print(f"  {'placebo excess (matched n)':22} {pe:+.3%}   lift = {d.excess_ret.mean() - pe:+.3%}")
    return df


def main():
    g1 = random.Random(7).sample(LIVE_TICKERS, 10)              # same 10 as the trough-pair test
    rest = [t for t in LIVE_TICKERS if t not in g1]
    g2 = random.Random(8).sample(rest, 10)                      # fresh replication group
    run_group(g1, "seed7")
    run_group(g2, "seed8")


if __name__ == "__main__":
    main()
