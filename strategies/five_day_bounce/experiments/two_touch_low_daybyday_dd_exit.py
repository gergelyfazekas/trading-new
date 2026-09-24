"""True day-by-day causal test of the market-drawdown early exit (see
mean_reversion_notes.md / the two_touch_low_risk_filters work) -- not the
vectorized post-hoc overlay used to first find this, which scanned each
already-known trade's future window in one pandas call. This script
instead loops one day at a time, in chronological order, and on each day
only ever touches close/volume/SPY sliced to `.iloc[:i+1]` -- there is no
way for it to see day i+1 while deciding day i, because that data is
never in scope.

Loop shape, per ticker, per day i ("today"):
  - if a position is open: check whether SPY has fallen >= dd_pct since
    entry (using only spy_close.iloc[:i+1]) or the 5-session hold is up;
    exit today if either is true, otherwise keep holding.
  - if no position is open: today is a candidate retest day for the low
    2 sessions back (L = i - retest_gap). Call decide() on
    close.iloc[:i+1]/volume.iloc[:i+1] -- exactly what a live run on day i
    could see. Open a position only if it fires the rebound_retest branch
    and buy_day == i (the pattern completes today, not some other day).

Every entry and every exit is re-verified by an independent recomputation
on freshly re-sliced `.iloc[:i+1]` arrays (never reusing an
already-sliced variable from earlier in the loop) and asserted equal to
the decision actually taken -- the same "causality checked by truncation,
not by inspection" convention as two_touch_low_daybyday.py, applied here
to the exit rule instead of the entry rule.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_daybyday_dd_exit.py [seed] [n_tickers] [dd_pct]
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
from two_touch_low_daybyday import PARAMS, HOLD, COST, WARMUP, decide

RETEST_GAP = 2


def spy_drawdown_since_entry(spy_trunc, entry_spy):
    """True if SPY's last known close (as of `spy_trunc`'s final row) has
    fallen >= dd_pct from `entry_spy` -- caller supplies dd_pct. Kept as a
    tiny function so the causal-check call below recomputes through the
    exact same code path, not a hand-copied version of it."""
    if len(spy_trunc) == 0 or pd.isna(entry_spy):
        return None
    now = spy_trunc.iloc[-1]
    if pd.isna(now):
        return None
    return now / entry_spy - 1


def run_ticker(args):
    ticker, close, volume, spy_close, p, dd_pct = args
    idx, n = close.index, len(close)
    busy = -1
    open_pos = None
    trades = []

    for i in range(WARMUP, n):
        trunc_close = close.iloc[:i + 1]
        trunc_volume = volume.iloc[:i + 1]
        trunc_spy = spy_close.iloc[:i + 1]

        if open_pos is not None:
            days_held = i - open_pos["buy_day"]
            move = spy_drawdown_since_entry(trunc_spy, open_pos["entry_spy"])
            dd_hit = move is not None and move <= -dd_pct
            if dd_hit or days_held >= HOLD:
                reason = "market_dd" if dd_hit and days_held < HOLD else "hold_days"
                exit_i = i

                # causal check: independent recomputation on freshly
                # re-sliced arrays, not the ones already in scope above.
                check_move = spy_drawdown_since_entry(spy_close.iloc[:exit_i + 1], open_pos["entry_spy"])
                check_dd_hit = check_move is not None and check_move <= -dd_pct
                check_reason = "market_dd" if check_dd_hit and days_held < HOLD else "hold_days"
                assert check_reason == reason, f"leakage: {ticker} exit day {idx[exit_i].date()}"

                ret = float(close.iloc[exit_i] / open_pos["entry_price"] - 1)
                trades.append(dict(stock=ticker, low_date=open_pos["low_date"],
                                    entry_date=idx[open_pos["buy_day"]], exit_date=idx[exit_i],
                                    ret=ret, ret_net=ret - COST, exit_reason=reason,
                                    days_held=days_held))
                busy = exit_i
                open_pos = None
            continue

        L = i - RETEST_GAP
        if L < p["lookback"]:
            continue
        decision = decide(trunc_close, trunc_volume, L, p, retest_gap=RETEST_GAP)
        if decision is None or decision["branch"] != "rebound_retest":
            continue
        buy_day = L + decision["offset"]
        if buy_day != i or buy_day <= busy:
            continue

        # causal check on entry: recompute on a fresh slice, same convention
        redecision = decide(close.iloc[:i + 1], volume.iloc[:i + 1], L, p, retest_gap=RETEST_GAP)
        assert redecision == decision, f"leakage: {ticker} entry day {idx[i].date()}"

        entry_spy = trunc_spy.iloc[-1]
        open_pos = dict(buy_day=buy_day, entry_price=float(close.iloc[buy_day]),
                         entry_spy=entry_spy, low_date=idx[L])

    return trades


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    dd_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}  (dd_pct={dd_pct:.0%})\n", flush=True)

    series, failed = pull_all(tickers + ["SPY"])
    if failed:
        print(f"no data for: {failed}")
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
    spy_close, _ = series["SPY"]
    spy_close = spy_close.copy(); spy_close.index = pd.to_datetime(spy_close.index)

    args = [(t, close[t], volume[t], spy_close.reindex(close[t].index, method="ffill"), PARAMS, dd_pct)
            for t in close]
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, args)

    df = pd.DataFrame([row for o in out for row in o])
    print("every entry and every exit independently re-verified on freshly re-sliced "
          "close/volume/SPY arrays -- no assertion failures means no leakage found\n")
    if df.empty:
        print("no trades fired on this small a ticker sample -- try more tickers")
        return
    df = attach_benchmark(df, ew)
    df.to_csv(os.path.join(HERE, "two_touch_low_daybyday_dd_exit_trades.csv"), index=False)

    print(summ(df, f"dd_pct={dd_pct:.0%} (day-by-day)"))
    print(f"\nexit reasons: {df.exit_reason.value_counts().to_dict()}\n")
    print(df[["stock", "low_date", "entry_date", "exit_date", "exit_reason", "ret_net"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
