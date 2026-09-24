"""Follow-up to two_touch_low_distinguishable_retest.py (which came back
negative at every hold): add two confirmation checks the user asked for
-- L+1 and L+3 (the buy day) must both close above L, not just L+2's
same-level retest. Same distinguishable-low and same-level-retest logic
otherwise, unchanged from the parent script:

  L    -- distinguishable low: close[L] at least fall_pct=3% below the
          highest close in the prior 10-day window.
  L+1  -- must close ABOVE L (previously unconstrained "wiggle" --
          this is the new constraint).
  L+2  -- the same-level retest: close[L+2] in [close[L], close[L]*1.01].
  L+3  -- must ALSO close above L (new constraint) -- buy at L+3's close
          if so. Unlike the parent script, this is no longer a pure
          execution-lag day with no condition of its own: the decision is
          only fully knowable at L+3's own close, same causal convention
          as buying on a retest day's own price/volume elsewhere in this
          candidate's history.

Net effect of both new checks: rules out any candidate where the
"wiggle" at L+1 or the price on the buy day itself has dropped back to
or below the original low -- only trades where the whole pattern stays
above L throughout are taken.

Exit: swept over hold in {2, 3, 4, 5} sessions, 10bps cost, one position
per ticker -- same convention as the parent script.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_retest_confirmed.py [seed] [n_tickers]
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
    """L (distinguishable low) -> L+1 (must close above L) -> L+2
    (same-level retest) -> L+3 (must also close above L; buy here if so).
    Only ever reads close up to whatever length the caller passed in."""
    n = len(close)
    if L < p["lookback"] or L + 3 >= n:
        return None
    window = close.iloc[L - p["lookback"]:L]
    if not bool((close.iloc[L] < window).all()):
        return None
    recent_high = float(window.max())
    c_L = float(close.iloc[L])
    fall = (recent_high - c_L) / recent_high
    if not (fall >= p["fall_pct"]):
        return None

    c_L1 = float(close.iloc[L + 1])
    if not (c_L1 > c_L):
        return None

    c_L2 = float(close.iloc[L + 2])
    if not (c_L <= c_L2 <= c_L * (1 + p["near_pct"])):
        return None

    c_L3 = float(close.iloc[L + 3])
    if not (c_L3 > c_L):
        return None

    return dict(L=L, L2=L + 2, buy_day=L + 3, fall=fall, retest_pct=(c_L2 - c_L) / c_L)


def run_ticker(args):
    ticker, close, p, hold = args
    idx, n = close.index, len(close)
    busy, trades = -1, []
    for L in range(WARMUP, n - hold - 4):
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
    df.to_csv(os.path.join(HERE, "two_touch_low_retest_confirmed_trades.csv"), index=False)

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
