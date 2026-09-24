"""Third rebuild of the original premise, requested 2026-09-22 after the
free-floating double-bottom (no L2-near-L1 constraint) came back
indistinguishable from a random-entry placebo. Fixed offsets again, like
the original locked spec, but the two touches are now defined purely by
price, no volume anywhere:

  L    -- a real local low, made "distinguishable" by requiring it follow
          a LARGER fall than the old spec's prominence check: close[L]
          must sit at least fall_pct below the highest close in the prior
          `lookback` window (not just a token drop vs the single
          preceding bar, which is all the old prominence_pct=0.5% check
          enforced).
  L+1  -- "some wiggle" -- deliberately unconstrained. No condition is
          checked here at all; it's just the day between the two touches.
  L+2  -- the second touch: close[L+2] must be close to L's level but not
          below it -- close[L] <= close[L+2] <= close[L] * (1+near_pct).
          This replaces the earlier version's unconstrained "another
          local low" with the actual "two-touch, same level" premise this
          whole candidate is named for.
  BUY  -- at L+3's close (one session after the retest confirms, not
          same-day -- a realistic one-day execution lag past the signal).

fall_pct and near_pct are free choices, stated plainly rather than swept:
fall_pct=3% (a real decline, not a shrug), near_pct=1% (close to the
first low, generous enough that "some wiggle" at L+1 doesn't need to
undo itself perfectly). lookback=10 (wider than the old 5-day window,
since "a larger fall" needs more room to show up than a single-day
prominence check did).

Exit: swept over hold in {2, 3, 4, 5} sessions from the buy day, per the
request -- reuses the same causal truncation check and one-trade-at-a-
time busy tracking as every other variant here. 10bps round-trip cost.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_distinguishable_retest.py [seed] [n_tickers]
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
from two_touch_low_daybyday import COST, WARMUP

PARAMS = dict(
    lookback=10,
    fall_pct=0.03,
    near_pct=0.01,
)
HOLDS = [2, 3, 4, 5]


def decide(close, L, p):
    """L (distinguishable low) -> L+1 (unconstrained) -> L+2 (same-level
    retest) -> buy at L+3's close. Only ever reads close up to whatever
    length the caller passed in; the signal is fully determined by L+2,
    L+3 is a pure execution-lag day with no condition of its own."""
    n = len(close)
    if L < p["lookback"] or L + 2 >= n:
        return None
    window = close.iloc[L - p["lookback"]:L]
    if not bool((close.iloc[L] < window).all()):
        return None
    recent_high = float(window.max())
    c_L = float(close.iloc[L])
    fall = (recent_high - c_L) / recent_high
    if not (fall >= p["fall_pct"]):
        return None

    c_L2 = float(close.iloc[L + 2])
    if not (c_L <= c_L2 <= c_L * (1 + p["near_pct"])):
        return None

    return dict(L=L, L2=L + 2, buy_day=L + 3, fall=fall, retest_pct=(c_L2 - c_L) / c_L)


def run_ticker(args):
    ticker, close, p, hold = args
    idx, n = close.index, len(close)
    busy, trades = -1, []
    for L in range(WARMUP, n - hold - 3):
        decision = decide(close, L, p)
        if decision is None:
            continue
        buy_day = decision["buy_day"]
        if buy_day >= n or buy_day <= busy:
            continue

        # causal check: recompute on data truncated to (and including) the
        # buy day only -- must reproduce the exact same decision.
        redecision = decide(close.iloc[:buy_day + 1], L, p)
        assert redecision == decision, f"leakage: {ticker} L={idx[L].date()} {decision} vs {redecision}"

        exit_i = buy_day + hold
        if exit_i >= n:
            continue
        ret = float(close.iloc[exit_i] / close.iloc[buy_day] - 1)
        trades.append(dict(
            stock=ticker, L_date=idx[L], L2_date=idx[decision["L2"]],
            entry_date=idx[buy_day], exit_date=idx[exit_i],
            fall=decision["fall"], retest_pct=decision["retest_pct"],
            ret=ret, ret_net=ret - COST,
        ))
        busy = exit_i
    return trades


def run_combo(close, ew, hold):
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, [(t, close[t], PARAMS, hold) for t in close])
    df = pd.DataFrame([row for o in out for row in o])
    if df.empty:
        return df
    return attach_benchmark(df, ew)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}", flush=True)
    print(f"params: {PARAMS}\n", flush=True)

    series, failed = pull_all(tickers)
    if failed:
        print(f"no data for: {failed}")
    today = datetime.date.today()
    close = {}
    for t in tickers:
        if t not in series:
            continue
        c, _ = series[t]
        c = c.copy(); c.index = pd.to_datetime(c.index)
        close[t] = c[c.index.date < today]
    ew = equal_weight_curve(close)

    dfs = {}
    for hold in HOLDS:
        df = run_combo(close, ew, hold)
        dfs[hold] = df
        print(summ(df, f"hold={hold}") if not df.empty else f"  hold={hold:22} no trades")

    best_hold = max((h for h, d in dfs.items() if not d.empty),
                     key=lambda h: dfs[h].excess_ret.mean() / dfs[h].excess_ret.std() * np.sqrt(len(dfs[h])),
                     default=None)
    if best_hold is None:
        print("\nno trades fired at any hold")
        return
    df = dfs[best_hold]
    df.to_csv(os.path.join(HERE, "two_touch_low_distinguishable_retest_trades.csv"), index=False)

    # placebo: random entries, matched count per ticker, same hold/cost
    rng = np.random.default_rng(0)
    rows = []
    for t, cnt in df.groupby("stock").size().items():
        c = close[t]
        for i in rng.choice(np.arange(WARMUP, len(c) - best_hold), size=cnt, replace=False):
            rows.append({"entry_date": c.index[i], "exit_date": c.index[i + best_hold],
                         "ret_net": c.iloc[i + best_hold] / c.iloc[i] - 1 - COST})
    pe = attach_benchmark(pd.DataFrame(rows), ew).excess_ret.mean()

    print(f"\nbest hold by t-stat: {best_hold}")
    print(f"  {'placebo excess (matched n)':22} {pe:+.3%}   causal lift = {df.excess_ret.mean() - pe:+.3%}")

    print("\nby year (best hold):")
    for y, g in df.groupby(pd.DatetimeIndex(df.entry_date).year):
        print(summ(g, str(y)))


if __name__ == "__main__":
    main()
