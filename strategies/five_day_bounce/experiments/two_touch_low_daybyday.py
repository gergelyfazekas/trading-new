"""Day-by-day (causal) test of the "two-touch low" candidate from
mean_reversion_notes.md ("Two-touch low, fully specified", 2026-09-22).

Rule, exactly as specified there:

Stage 1 -- first low at day L, confirmed as of day L+1's close:
  - close[L] < close[L-lookback], ..., close[L-1]  (backward-only rolling
    local minimum -- the causal replacement for find_peaks(distance=N))
  - close[L-1] >= close[L] * (1 + prominence_pct)   (prominence vs L-1 only)
  - volume_ratio[L] > volume_threshold

Stage 2 -- buy trigger, branching on pct1 = (close[L+1] - close[L]) / close[L]:
  - flat:            0 < pct1 <= k_pct                        -> buy at L+1
  - bigger rebound:  k_pct < pct1 <= rebound_max_pct
                      and volume_ratio[L+1] < rebound_volume_threshold
                      and close[L] <= close[L+2] <= close[L+1]
                      and volume_ratio[L+2] > volume_threshold_buy -> buy at L+2
  - otherwise: no buy

volume_ratio[j] = volume[j] / mean(volume[j-10:j]) -- strictly trailing,
excludes day j itself (same definition as
tech_level_walkforward_demo.py's trailing_avg_volume/volume_ratio).

Exit: fixed 5-session hold (this signal defines no resistance/level
structure, so unlike the old frozen rule there is no early exit-on-
resistance branch). 10 bps round-trip cost. One position per ticker.

Unlike the old find_peaks-based rule, nothing here ever looks more than 2
days past L, and the buy is dated at the day the pattern actually
completes -- so a single pass over the full series is causal by
construction, with no persistent level-birth/break state to leak through.
That claim is checked empirically, not just asserted: every candidate is
evaluated once on the full series and once on the series truncated to
(and including) its own buy day, and the two decisions must match exactly
-- same convention ("causality is checked by truncation, not by
inspection") as original_rule_daybyday.py, applied here as a regression
check on the code rather than a correction to the theory.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_daybyday.py [seed] [n_tickers]
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

# locked starting values, mean_reversion_notes.md 2026-09-22
PARAMS = dict(
    lookback=5,
    prominence_pct=0.005,
    volume_threshold=1.0,
    k_pct=0.005,
    rebound_max_pct=0.02,
    rebound_volume_threshold=1.0,
    volume_threshold_buy=0.8,
)
HOLD = 5
COST = 10 / 1e4
VOL_LOOKBACK = 10
WARMUP = 30


def volume_ratio(volume, j, lookback=VOL_LOOKBACK):
    """Trailing lookback-day average volume strictly before position j,
    same definition as tech_level_walkforward_demo.trailing_avg_volume."""
    seg = volume.iloc[max(0, j - lookback):j]
    avg = float(seg.mean()) if len(seg) and seg.notna().any() else float("nan")
    if avg in (0.0,) or np.isnan(avg) or np.isnan(volume.iloc[j]):
        return float("nan")
    return float(volume.iloc[j]) / avg


def decide(close, volume, L, p, retest_gap=2):
    """Evaluate the two-touch-low rule for candidate low at position L,
    using only close/volume up to whatever length the caller passed in
    (never accesses index > L+retest_gap). Returns None (no buy) or a dict
    describing the fired trigger.

    retest_gap generalizes the rebound_retest branch's second-touch day
    from the locked spec's fixed L+2 to L+retest_gap (2 or 3) -- how many
    days after the first low the two troughs sit apart. The flat branch is
    unaffected (still fires at L+1, since it isn't a two-touch pattern)."""
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
    if not (pct1 > 0):
        return None

    if pct1 <= p["k_pct"]:
        return dict(offset=1, branch="flat", pct1=pct1, vr_L=vr_L)

    if pct1 <= p["rebound_max_pct"]:
        vr_L1 = volume_ratio(volume, L + 1)
        if not (vr_L1 < p["rebound_volume_threshold"]):
            return None
        L2 = L + retest_gap
        if L2 >= n:
            return None
        c_L2 = float(close.iloc[L2])
        if not (c_L <= c_L2 <= c_L1):
            return None
        vr_L2 = volume_ratio(volume, L2)
        if not (vr_L2 > p["volume_threshold_buy"]):
            return None
        return dict(offset=retest_gap, branch="rebound_retest", pct1=pct1, vr_L=vr_L, vr_L1=vr_L1, vr_L2=vr_L2)

    return None


# Stages, roughly ordered by how close a candidate got to firing --
# for sorting a diagnose() table so the closest-to-firing names surface
# first (used by two_touch_low_live.py, not the backtest). The
# "awaiting_*" stages are the only ones that can still change with the
# next session's data; every other stage is a closed case for that
# specific L -- see diagnose()'s docstring.
STAGE_RANK = {
    "buy": 0,
    "retest_volume_fail": 1, "retest_out_of_band": 1, "awaiting_retest_day": 1,
    "rebound_volume_too_high": 2,
    "awaiting_rebound_day": 2.5,
    "rebound_too_big": 3, "flat_branch_only": 3,
    "no_rebound": 4,
    "birth_volume_fail": 5,
    "prominence_fail": 6,
    "not_a_low": 7,
    "insufficient_history": 8,
}


def diagnose(close, volume, L, p, retest_gap=2):
    """Always returns a dict with a 'stage' explaining what happened for
    candidate low L -- for live-run visibility only, never used by the
    backtest or by decide() itself.

    Unlike decide() (which needs L+1 to exist just to start), this checks
    the backward-only Stage 1 conditions first, so a low that formed on
    the *most recent* available bar (no L+1 yet) is reported as
    "awaiting_rebound_day" instead of lumped in with "insufficient_history"
    -- the two mean very different things for a live read: one might
    still turn into a buy over the next session or two, the other is
    nothing at all.

    Every stage except "buy", "awaiting_rebound_day" and
    "awaiting_retest_day" is a CLOSED case for this specific L: the data
    it depends on already happened and won't change tomorrow. Only those
    three (and by extension "flat_branch_only" when it's the *most
    recent* possible low) describe something still capable of changing.

    The passing case defers entirely to decide(), so there is exactly one
    source of truth for when a trade actually fires; everything else here
    only narrates why one didn't (or hasn't yet). If decide()'s branching
    ever changes, only this function's prose can go stale -- the buy/
    no-buy decision itself cannot, since it is never recomputed here."""
    n = len(close)
    if L < p["lookback"] or L >= n:
        return {"stage": "insufficient_history"}

    window = close.iloc[L - p["lookback"]:L]
    if not bool((close.iloc[L] < window).all()):
        return {"stage": "not_a_low"}
    if not (close.iloc[L - 1] >= close.iloc[L] * (1 + p["prominence_pct"])):
        return {"stage": "prominence_fail"}
    vr_L = volume_ratio(volume, L)
    if not (vr_L > p["volume_threshold"]):
        return {"stage": "birth_volume_fail", "vr_L": vr_L}

    if L + 1 >= n:
        return {"stage": "awaiting_rebound_day", "vr_L": vr_L}

    decision = decide(close, volume, L, p, retest_gap=retest_gap)
    if decision is not None:
        # decide() fires on both branches, but only rebound_retest is
        # actually traded live (see two_touch_low_live.py) -- a flat
        # decision must not be reported as "buy" here.
        stage = "buy" if decision["branch"] == "rebound_retest" else "flat_branch_only"
        return {"stage": stage, **decision}

    c_L = float(close.iloc[L])
    c_L1 = float(close.iloc[L + 1])
    pct1 = (c_L1 - c_L) / c_L
    if not (pct1 > 0):
        return {"stage": "no_rebound", "vr_L": vr_L, "pct1": pct1}
    if pct1 <= p["k_pct"]:
        return {"stage": "flat_branch_only", "vr_L": vr_L, "pct1": pct1}  # not traded live; unreachable (decide() already caught it above), kept for safety
    if pct1 > p["rebound_max_pct"]:
        return {"stage": "rebound_too_big", "vr_L": vr_L, "pct1": pct1}

    vr_L1 = volume_ratio(volume, L + 1)
    if not (vr_L1 < p["rebound_volume_threshold"]):
        return {"stage": "rebound_volume_too_high", "vr_L": vr_L, "pct1": pct1, "vr_L1": vr_L1}

    L2 = L + retest_gap
    if L2 >= n:
        return {"stage": "awaiting_retest_day", "vr_L": vr_L, "pct1": pct1, "vr_L1": vr_L1}

    c_L2 = float(close.iloc[L2])
    if not (c_L <= c_L2 <= c_L1):
        return {"stage": "retest_out_of_band", "vr_L": vr_L, "pct1": pct1, "vr_L1": vr_L1, "close_L2": c_L2}

    vr_L2 = volume_ratio(volume, L2)
    return {"stage": "retest_volume_fail", "vr_L": vr_L, "pct1": pct1, "vr_L1": vr_L1, "vr_L2": vr_L2}


def run_ticker(args):
    if len(args) == 6:
        ticker, close, volume, p, hold, retest_gap = args
    elif len(args) == 5:
        ticker, close, volume, p, hold = args
        retest_gap = 2
    else:
        ticker, close, volume, p = args
        hold = HOLD
        retest_gap = 2
    idx, n = close.index, len(close)
    busy, trades = -1, []
    for L in range(WARMUP, n - hold - max(retest_gap, 2) - 1):
        decision = decide(close, volume, L, p, retest_gap=retest_gap)
        if decision is None:
            continue
        buy_day = L + decision["offset"]
        if buy_day <= busy:
            continue

        # causal check: recompute on data truncated to (and including) the
        # buy day only -- must reproduce the exact same decision.
        trunc_close = close.iloc[:buy_day + 1]
        trunc_volume = volume.iloc[:buy_day + 1]
        redecision = decide(trunc_close, trunc_volume, L, p, retest_gap=retest_gap)
        assert redecision == decision, f"leakage: {ticker} L={idx[L].date()} {decision} vs {redecision}"

        exit_i = buy_day + hold
        if exit_i >= n:
            continue
        ret = float(close.iloc[exit_i] / close.iloc[buy_day] - 1)
        trades.append(dict(
            stock=ticker, low_date=idx[L], entry_date=idx[buy_day], exit_date=idx[exit_i],
            branch=decision["branch"], pct1=decision["pct1"],
            ret=ret, ret_net=ret - COST,
        ))
        busy = exit_i
    return trades


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    retest_gap = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}", flush=True)
    print(f"params: {PARAMS}, retest_gap={retest_gap}\n", flush=True)

    series, failed = pull_all(tickers)
    if failed:
        print(f"no data for: {failed}")
    today = datetime.date.today()
    close, volume = {}, {}
    for t in tickers:
        if t not in series:
            continue
        c, v = series[t]
        c = c.copy()
        c.index = pd.to_datetime(c.index)
        mask = c.index.date < today
        close[t] = c[mask]
        if v is not None:
            v = v.copy()
            v.index = pd.to_datetime(v.index)
            volume[t] = v.reindex(close[t].index)
        else:
            volume[t] = pd.Series(np.nan, index=close[t].index)

    ew = equal_weight_curve(close)

    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, [(t, close[t], volume[t], PARAMS, HOLD, retest_gap) for t in close])
    df = pd.DataFrame([t for o in out for t in o])
    if df.empty:
        print("no trades fired at these parameters")
        return
    df = attach_benchmark(df, ew)
    suffix = "" if retest_gap == 2 else f"_gap{retest_gap}"
    df.to_csv(os.path.join(HERE, f"two_touch_low_daybyday_trades{suffix}.csv"), index=False)

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
    print(summ(df, "ALL (causal)"))
    print(f"  {'placebo excess (matched n)':22} {pe:+.3%}   causal lift = {df.excess_ret.mean() - pe:+.3%}")

    print("\nby branch:")
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
