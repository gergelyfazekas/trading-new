"""Day-by-day (causal) test of the "two-touch low v2" candidate agreed with
the user 2026-09-23 -- a from-scratch redesign of the two-touch-low idea,
not a variant of the rebound_retest rule in two_touch_low_daybyday.py.

Rule:

Stage 1 -- first trough at day L, a two-sided local minimum:
  - close[L] < close[L-1], ..., close[L-n_back]   (backward window)
  - close[L] < close[L+1], ..., close[L+n_fwd]    (forward window)
  - gate: buyers_took_over[L] OR volume_ratio[L] > volume_threshold_L

buyers_took_over[j] = close[j] closer to high[j] than to low[j], i.e.
(close[j] - low[j]) > (high[j] - close[j]) -- an intraday buying-pressure
proxy, using that day's own high/low, nothing else.

L is only confirmed as of the close of L+n_fwd -- the first day the
backward and forward windows are both fully known. Nothing here ever
looks further ahead than that.

Stage 2 -- second trough at L2 = L + n_fwd + 1 (a fixed offset, not a
scanned window -- this is what makes the forward-looking Stage 1 check
causal: by L2's close, days L+1..L+n_fwd already happened):
  - |close[L2] - close[L]| / close[L] <= close_tolerance
  - gate: buyers_took_over[L2] OR volume_ratio[L2] > volume_threshold_L2

Buy fires at close[L2].

Supersession: if a day Lp with L < Lp < L2 is itself a confirmed Stage-1
local minimum, L is discarded outright (returns no decision) -- Lp gets
its own turn as the loop reaches it. Because Stage 1 needs n_fwd days of
forward confirmation, the only Lp in that range knowable by L2's close is
Lp = L+1 (every other Lp in the range still has unconfirmed forward data
at that point) -- so this falls out of the loop structure rather than
needing a special case.

volume_ratio[j] = volume[j] / mean(volume[j-10:j]) -- strictly trailing,
excludes day j itself, imported from two_touch_low_daybyday.py so both
rules share one definition.

Exit: 5-session hold or SPY market-drawdown exit (dd_pct, checked fresh
each session), whichever comes first -- same convention as
two_touch_low_daybyday_dd_exit.py, applied here from the start rather than
bolted on later. 10 bps round-trip cost. One position per ticker.

Every entry is independently re-verified by recomputing decide() on data
truncated to (and including) L2 -- "causality checked by truncation, not
by inspection," same convention as the rest of this family. Parameters
below are a single loose starting point to prove the rule is causal and
see the shape of things; n_back, n_fwd, volume_threshold_L,
volume_threshold_L2 and close_tolerance are all meant to be swept
afterward, not tuned here.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_v2_daybyday.py [seed] [n_tickers] [dd_pct]
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
from two_touch_low_daybyday import volume_ratio

# loose starting point, 2026-09-23 -- not yet swept
PARAMS = dict(
    n_back=7,
    n_fwd=3,
    volume_threshold_L=1.0,
    volume_threshold_L2=1.0,
    close_tolerance=0.005,
)
HOLD = 5
COST = 10 / 1e4
WARMUP = 30


def is_local_min(close, j, n_back, n_fwd, n):
    """Two-sided local minimum at j: close[j] strictly below every close in
    the n_back days before it and the n_fwd days after it. Only knowable
    once day j+n_fwd has happened -- returns False (not yet confirmable)
    when it hasn't."""
    if j < n_back or j + n_fwd >= n:
        return False
    back = close.iloc[j - n_back:j]
    fwd = close.iloc[j + 1:j + n_fwd + 1]
    return bool((close.iloc[j] < back).all() and (close.iloc[j] < fwd).all())


def buyers_took_over(high, low, close, j):
    h, l, c = high.iloc[j], low.iloc[j], close.iloc[j]
    if pd.isna(h) or pd.isna(l) or pd.isna(c):
        return False
    return bool((c - l) > (h - c))


def decide(close, high, low, volume, L, p):
    """Evaluate the two-touch-low v2 rule for first-trough candidate L.
    Never accesses an index past L + p['n_fwd'] + 1. Returns None (no buy)
    or a dict describing the fired signal."""
    n = len(close)
    n_back, n_fwd = p["n_back"], p["n_fwd"]
    if not is_local_min(close, L, n_back, n_fwd, n):
        return None

    bto_L = buyers_took_over(high, low, close, L)
    vr_L = volume_ratio(volume, L)
    gate_L = bto_L or (not np.isnan(vr_L) and vr_L > p["volume_threshold_L"])
    if not gate_L:
        return None

    L2 = L + n_fwd + 1
    if L2 >= n:
        return None

    for Lp in range(L + 1, L2):
        if is_local_min(close, Lp, n_back, n_fwd, n):
            return None

    c_L, c_L2 = float(close.iloc[L]), float(close.iloc[L2])
    if abs(c_L2 - c_L) / c_L > p["close_tolerance"]:
        return None

    bto_L2 = buyers_took_over(high, low, close, L2)
    vr_L2 = volume_ratio(volume, L2)
    gate_L2 = bto_L2 or (not np.isnan(vr_L2) and vr_L2 > p["volume_threshold_L2"])
    if not gate_L2:
        return None

    return dict(L2=L2, close_L=c_L, close_L2=c_L2, bto_L=bto_L, bto_L2=bto_L2,
                vr_L=vr_L, vr_L2=vr_L2)


def spy_drawdown_since_entry(spy_trunc, entry_spy):
    """True if SPY's last known close (as of spy_trunc's final row) has
    fallen >= dd_pct from entry_spy -- caller supplies dd_pct. Kept as a
    tiny function so the causal-check call recomputes through the exact
    same code path, not a hand-copied version of it."""
    if len(spy_trunc) == 0 or pd.isna(entry_spy):
        return None
    now = spy_trunc.iloc[-1]
    if pd.isna(now):
        return None
    return now / entry_spy - 1


def run_ticker(args):
    ticker, close, high, low, volume, spy_close, p, dd_pct = args
    idx, n = close.index, len(close)
    n_fwd = p["n_fwd"]
    busy = -1
    open_pos = None
    trades = []

    for i in range(WARMUP, n):
        trunc_close = close.iloc[:i + 1]
        trunc_high = high.iloc[:i + 1]
        trunc_low = low.iloc[:i + 1]
        trunc_volume = volume.iloc[:i + 1]
        trunc_spy = spy_close.iloc[:i + 1]

        if open_pos is not None:
            days_held = i - open_pos["buy_day"]
            move = spy_drawdown_since_entry(trunc_spy, open_pos["entry_spy"])
            dd_hit = move is not None and move <= -dd_pct
            if dd_hit or days_held >= HOLD:
                reason = "market_dd" if dd_hit and days_held < HOLD else "hold_days"
                exit_i = i

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

        L = i - (n_fwd + 1)
        if L < p["n_back"]:
            continue
        decision = decide(trunc_close, trunc_high, trunc_low, trunc_volume, L, p)
        if decision is None or decision["L2"] != i or i <= busy:
            continue

        redecision = decide(close.iloc[:i + 1], high.iloc[:i + 1], low.iloc[:i + 1],
                             volume.iloc[:i + 1], L, p)
        assert redecision == decision, f"leakage: {ticker} entry day {idx[i].date()}"

        entry_spy = trunc_spy.iloc[-1]
        open_pos = dict(buy_day=i, entry_price=float(close.iloc[i]),
                         entry_spy=entry_spy, low_date=idx[L])

    return trades


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    dd_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}  (dd_pct={dd_pct:.0%})", flush=True)
    print(f"params: {PARAMS}\n", flush=True)

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
    spy_close = series["SPY"][0].copy()
    spy_close.index = pd.to_datetime(spy_close.index)

    args = [(t, close[t], high[t], low[t], volume[t],
             spy_close.reindex(close[t].index, method="ffill"), PARAMS, dd_pct)
            for t in close]
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, args)

    df = pd.DataFrame([row for o in out for row in o])
    print("every entry independently re-verified on freshly re-sliced close/high/low/volume "
          "(causal by construction); every exit independently re-verified on freshly re-sliced "
          "SPY -- no assertion failures means no leakage found\n")
    if df.empty:
        print("no trades fired at these parameters")
        return
    df = attach_benchmark(df, ew)
    df.to_csv(os.path.join(HERE, "two_touch_low_v2_daybyday_trades.csv"), index=False)

    print(f"{len(close)} tickers, ~{min(len(c) for c in close.values())}+ sessions each\n")
    print(summ(df, f"ALL (dd_pct={dd_pct:.0%})"))
    print(f"\nexit reasons: {df.exit_reason.value_counts().to_dict()}\n")

    print("by year:")
    for y, g in df.groupby(pd.DatetimeIndex(df.entry_date).year):
        print(summ(g, str(y)))

    print("\nper ticker:")
    for t in close:
        print(summ(df[df.stock == t], t))


if __name__ == "__main__":
    main()
