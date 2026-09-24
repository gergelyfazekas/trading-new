"""Back to the pattern this whole candidate started from, with every
volume knob removed (requested 2026-09-22, after gate-relaxation,
100-ticker-expansion, and no-volume+early-exit all failed to raise trade
frequency without destroying the edge -- see mean_reversion_notes.md).
Pure price structure, no volume anywhere:

  L1 -- a local low.
  H1 -- a "smallish" local high after L1 (a real turning point, not just
        the next bar being higher -- see local_high() below).
  L2 -- another local low after H1 (the second touch).
  BUY -- the first close after L2 that breaks back above H1's close --
        "confirmed by another high." This is the classic double-bottom
        breakout: L1/L2 are the two touches of the level, H1 is the
        neckline, and the pattern isn't traded until price actually
        clears the neckline again.

"Local low"/"local high" are both backward-only rolling extrema (never a
find_peaks-style condition that needs bars after the point itself to
confirm it) -- same causality discipline as two_touch_low_daybyday.py's
design rule, applied symmetrically to highs for the first time here:
  local_low(i):  close[i] < close[i-lookback:i] (all)  and
                 close[i-1] >= close[i] * (1 + prominence_pct)
  local_high(i): close[i] > close[i-lookback:i] (all)  and
                 close[i-1] <= close[i] * (1 - prominence_pct)

H1's rise over L1 must be "smallish": 0 < (close[H1]-close[L1])/close[L1]
<= high_max_pct. If the first local high found after L1 doesn't clear
that bar, the candidate is rejected outright (no hunting for a later,
smaller peak -- the pattern is specifically "one small bounce, then a
second dip"). L2 and the breakout confirmation carry no magnitude
constraint of their own, only the leg_max search-window cap, since
nothing was specified for them beyond "another local low" / "another
high."

leg_max bounds how far forward each leg (L1->H1, H1->L2, L2->breakout) is
allowed to search, so a pattern can't stay "open" indefinitely on stale,
unrelated price action -- an arbitrary but explicit knob (10 sessions),
not swept here.

Exit: fixed 5-session hold from the buy day, 10bps round-trip cost, one
position per ticker at a time -- same baseline convention as
two_touch_low_daybyday.py, so this is a clean first look at the entry
alone before any exit variant (drawdown, early-reversal) is layered on.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_double_bottom.py [seed] [n_tickers]
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
from two_touch_low_daybyday import HOLD, COST, WARMUP

PARAMS = dict(
    lookback=5,
    prominence_pct=0.005,
    high_max_pct=0.02,
    leg_max=10,
)


def local_low(close, i, lookback, prominence_pct):
    if i < lookback:
        return False
    window = close.iloc[i - lookback:i]
    if not bool((close.iloc[i] < window).all()):
        return False
    return bool(close.iloc[i - 1] >= close.iloc[i] * (1 + prominence_pct))


def local_high(close, i, lookback, prominence_pct):
    if i < lookback:
        return False
    window = close.iloc[i - lookback:i]
    if not bool((close.iloc[i] > window).all()):
        return False
    return bool(close.iloc[i - 1] <= close.iloc[i] * (1 - prominence_pct))


def decide(close, L1, p):
    """L1 -> smallish H1 -> L2 -> breakout-above-H1 buy. Only ever reads
    close up to whatever length the caller passed in. Returns None or a
    dict describing the fired pattern (L1, H1, L2, buy_day, magnitudes)."""
    n = len(close)
    if not local_low(close, L1, p["lookback"], p["prominence_pct"]):
        return None
    c_L1 = float(close.iloc[L1])

    H1 = None
    for j in range(L1 + 1, min(n, L1 + p["leg_max"] + 1)):
        if local_high(close, j, p["lookback"], p["prominence_pct"]):
            rise = (float(close.iloc[j]) - c_L1) / c_L1
            if 0 < rise <= p["high_max_pct"]:
                H1 = j
            break  # first local high after L1 decides the pattern, qualifying or not
    if H1 is None:
        return None
    c_H1 = float(close.iloc[H1])

    L2 = None
    for k in range(H1 + 1, min(n, H1 + p["leg_max"] + 1)):
        if local_low(close, k, p["lookback"], p["prominence_pct"]):
            L2 = k
            break
    if L2 is None:
        return None
    c_L2 = float(close.iloc[L2])

    for m in range(L2 + 1, min(n, L2 + p["leg_max"] + 1)):
        if float(close.iloc[m]) > c_H1:
            return dict(L1=L1, H1=H1, L2=L2, buy_day=m,
                        rise1=(c_H1 - c_L1) / c_L1, drop2=(c_H1 - c_L2) / c_H1)
    return None


def run_ticker(args):
    ticker, close, p, hold = args
    idx, n = close.index, len(close)
    busy, trades = -1, []
    for L1 in range(WARMUP, n - hold - 1):
        decision = decide(close, L1, p)
        if decision is None:
            continue
        buy_day = decision["buy_day"]
        if buy_day <= busy:
            continue

        # causal check: recompute on data truncated to (and including) the
        # buy day only -- must reproduce the exact same decision.
        redecision = decide(close.iloc[:buy_day + 1], L1, p)
        assert redecision == decision, f"leakage: {ticker} L1={idx[L1].date()} {decision} vs {redecision}"

        exit_i = buy_day + hold
        if exit_i >= n:
            continue
        ret = float(close.iloc[exit_i] / close.iloc[buy_day] - 1)
        trades.append(dict(
            stock=ticker, L1_date=idx[L1], H1_date=idx[decision["H1"]], L2_date=idx[decision["L2"]],
            entry_date=idx[buy_day], exit_date=idx[exit_i],
            rise1=decision["rise1"], drop2=decision["drop2"],
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

    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, [(t, close[t], PARAMS, HOLD) for t in close])
    df = pd.DataFrame([row for o in out for row in o])
    if df.empty:
        print("no trades fired at these parameters")
        return
    df = attach_benchmark(df, ew)
    df.to_csv(os.path.join(HERE, "two_touch_low_double_bottom_trades.csv"), index=False)

    # placebo: random entries, matched count per ticker, same hold/cost
    rng = np.random.default_rng(0)
    rows = []
    for t, cnt in df.groupby("stock").size().items():
        c = close[t]
        for i in rng.choice(np.arange(WARMUP, len(c) - HOLD), size=cnt, replace=False):
            rows.append({"entry_date": c.index[i], "exit_date": c.index[i + HOLD],
                         "ret_net": c.iloc[i + HOLD] / c.iloc[i] - 1 - COST})
    pe = attach_benchmark(pd.DataFrame(rows), ew).excess_ret.mean()

    print(f"{len(close)} tickers, ~{min(len(c) for c in close.values())}+ sessions each\n")
    print("every trade's entry decision verified identical on truncated-to-buy-day data (causal by construction)\n")
    print(summ(df, "double-bottom, no volume"))
    print(f"  {'placebo excess (matched n)':22} {pe:+.3%}   causal lift = {df.excess_ret.mean() - pe:+.3%}")

    print("\nby year:")
    for y, g in df.groupby(pd.DatetimeIndex(df.entry_date).year):
        print(summ(g, str(y)))


if __name__ == "__main__":
    main()
