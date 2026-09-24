"""New variant, requested 2026-09-22 (same day as the rejected 100-ticker
expansion): instead of widening the ticker universe, change the rule
itself to fire more often. Two independent changes, tested separately and
combined so the effect of each is isolated (per validation-first-quant-work
-- don't conflate two changes into one number):

  (a) DROP the rebound_retest branch's two volume gates entirely --
      vr_L1 < rebound_volume_threshold (L+1's weak-volume rebound
      requirement, survival 46.0% per the funnel breakdown) and
      vr_L2 > volume_threshold_buy (L+2's retest-volume floor, survival
      58.6%). Stage 1's own volume gate at L is untouched. The price-band
      structure (k_pct < pct1 <= rebound_max_pct, close[L] <= close[L+2]
      <= close[L+1]) is untouched -- still a real two-touch pattern, just
      no longer volume-confirmed on the second and third days.

  (b) ADD a one-day-after early exit: if close[buy_day+1] (i.e. L+3, one
      session after the L+2 buy) is below close[buy_day] (L+2), exit
      right there instead of waiting out the fixed 5-session hold. This
      is a fast bail-out on an immediate reversal, distinct from the
      already-tested below-L stop-loss (rejected at every buffer size)
      and from the market-drawdown exit (SPY-based, not price-based).
      decide() itself and two_touch_low_daybyday.py are untouched --
      this file is a separate, explicitly experimental copy so nothing
      here can accidentally change the validated live rule.

Four variants on the same 20-ticker seed=21 subset used throughout this
candidate's testing (same convention as two_touch_low_relaxed_variant.py:
small-subset prototype before deciding whether a full-101/fresh-81 run is
worth it):
  1. locked (baseline)              -- current live rule, fixed 5-day hold
  2. + early exit only              -- locked entry, add the L+3<L+2 bail
  3. no volume gates only           -- drop vr_L1/vr_L2, fixed 5-day hold
  4. no volume gates + early exit   -- both changes together (what was asked)

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_novolume_earlyexit.py [seed] [n_tickers]
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


def decide_variant(close, volume, L, p, require_volume):
    """Same Stage 1 as decide(); Stage 2 restricted to the bigger-rebound
    (rebound_retest) branch only. require_volume=True reproduces decide()
    exactly (for the baseline variants); False drops the vr_L1/vr_L2 gates
    but keeps the price-band conditions."""
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
        return None  # flat branch excluded -- never traded live, not this file's concern

    vr_L1 = volume_ratio(volume, L + 1)
    if require_volume and not (vr_L1 < p["rebound_volume_threshold"]):
        return None

    L2 = L + RETEST_GAP
    if L2 >= n:
        return None
    c_L2 = float(close.iloc[L2])
    if not (c_L <= c_L2 <= c_L1):
        return None

    vr_L2 = volume_ratio(volume, L2)
    if require_volume and not (vr_L2 > p["volume_threshold_buy"]):
        return None
    return dict(offset=RETEST_GAP, pct1=pct1, vr_L=vr_L, vr_L1=vr_L1, vr_L2=vr_L2)


def run_ticker(args):
    ticker, close, volume, p, require_volume, early_exit, hold = args
    idx, n = close.index, len(close)
    busy, trades = -1, []
    for L in range(WARMUP, n - hold - RETEST_GAP - 1):
        decision = decide_variant(close, volume, L, p, require_volume)
        if decision is None:
            continue
        buy_day = L + decision["offset"]
        if buy_day <= busy:
            continue

        # causal check: recompute on data truncated to (and including) the
        # buy day only -- must reproduce the exact same decision.
        redecision = decide_variant(close.iloc[:buy_day + 1], volume.iloc[:buy_day + 1], L, p, require_volume)
        assert redecision == decision, f"leakage: {ticker} L={idx[L].date()} {decision} vs {redecision}"

        exit_i, reason = buy_day + hold, "hold_days"
        if early_exit:
            next_i = buy_day + 1
            if next_i >= n:
                continue
            if close.iloc[next_i] < close.iloc[buy_day]:
                exit_i, reason = next_i, "early_reversal"
        if exit_i >= n:
            continue
        ret = float(close.iloc[exit_i] / close.iloc[buy_day] - 1)
        trades.append(dict(stock=ticker, low_date=idx[L], entry_date=idx[buy_day], exit_date=idx[exit_i],
                            pct1=decision["pct1"], exit_reason=reason,
                            ret=ret, ret_net=ret - COST, days_held=exit_i - buy_day))
        busy = exit_i
    return trades


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}\n", flush=True)

    series, failed = pull_all(tickers)
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

    variants = [
        ("locked (baseline)", True, False),
        ("+ early exit only", True, True),
        ("no volume gates only", False, False),
        ("no volume + early exit", False, True),
    ]
    for label, require_volume, early_exit in variants:
        args = [(t, close[t], volume[t], PARAMS, require_volume, early_exit, HOLD) for t in close]
        with Pool(min(len(close), cpu_count())) as pool:
            out = pool.map(run_ticker, args)
        df = pd.DataFrame([row for o in out for row in o])
        if df.empty:
            print(f"  {label:24} no trades")
            continue
        df = attach_benchmark(df, ew)
        print(summ(df, label))
        if "early exit" in label:
            print(f"    exit reasons: {df.exit_reason.value_counts().to_dict()}")
        if label == "no volume + early exit":
            df.to_csv(os.path.join(HERE, "two_touch_low_novolume_earlyexit_trades.csv"), index=False)


if __name__ == "__main__":
    main()
