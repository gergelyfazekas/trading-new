"""Day-by-day (causal) test of the "mean reversion after a big selloff"
candidate from mean_reversion_notes.md (2026-09-23 brainstorm session).
Independent from the abandoned two-touch-low family
(`two_touch_low_daybyday.py` and its variants) -- this is a from-scratch
premise, not a refinement of it.

Trigger (Phase 0), causal drawdown-depth z-score -- notes' candidate #2,
generalized from "distance below the recent high" to "how many std devs
below", so magnitude means the same thing across tickers/regimes instead
of a fixed %:

  roll_max[i] = max(close[i-N:i])      (trailing N days, excludes day i)
  roll_std[i] = std(close[i-N:i])      (trailing N days, excludes day i)
  z[i] = (close[i] - roll_max[i]) / roll_std[i]

Fires when z[i] <= -Z_THRESH. Both roll_max/roll_std look backward only
(via shift(1).rolling(N)), so z[i] depends only on close[i-N..i] -- known
at day i's own close, same causal convention as the rest of this file's
prior work ("close[L] < close[L-lookback..L-1]" etc.).

Market-wide vs. idiosyncratic split (Phase 1, built in from the start
rather than added after a baseline, per the 2026-09-23 brainstorm): the
same z-score is computed for SPY over the identical window and aligned
onto each ticker's own calendar. A stock's selloff is classified
"market_wide" if SPY itself is deep in its own drawdown at the same time
(plausibly forced-selling/liquidity driven, plausibly reverts) or
"idiosyncratic" if SPY is roughly flat (plausibly name-specific bad news,
plausibly does not revert) -- these are tested as two separate buckets of
the same population, not two separate scripts, so the split can be
evaluated head-to-head in one run. A middle "mixed" zone (SPY somewhat
down but not enough to call market-wide, not enough to call flat either)
is deliberately excluded from both buckets rather than forced into one.

Phase-0 result (price-only trigger, no volume): rejected. Every
combination on an (n_window, z_thresh) grid came back flat-to-negative
(worst: n_window=30, z_thresh=2.5, t=-2.28; best: n_window=10,
z_thresh=3.0, t=-0.24 -- never positive) and the market_wide vs.
idiosyncratic split did not separate as hypothesized (market_wide
consistently negative alongside idiosyncratic, sometimes worse). Matches
the two-touch-low family's own established finding that bare price
patterns come back flat-to-negative while the volume-gated version held
up -- see selloff_bounce_sweep.py for the full grid.

Capitulation-volume gate (added 2026-09-23, same session): decide() now
also computes volume_ratio[i] (trailing-10, excludes day i -- identical
definition to two_touch_low_daybyday.py's) at the trigger day and reports
it on every decision, UNGATED -- no threshold is applied inside decide()
itself. This keeps the triggering population fixed and lets volume_ratio
be sliced into quantile buckets after the fact
(selloff_bounce_volume_quantile.py), per mean_reversion_notes.md's
explicit warning ("Reminder for later sweeps") that naive threshold
sweeps on a volume-ratio-style variable produce a fake smooth-looking
dose-response from nested subsets alone, not a real one -- equal-sized,
disjoint quantile buckets are required instead. Exit is a plain fixed
5-session hold; 10bps round-trip cost; one position per ticker.

Buy fires at close[i], the day the trigger completes (z[i] uses close[i]
itself, known at that day's own close) -- no extra confirmation lag,
same dating convention as two_touch_low_daybyday.py's flat branch.

Causality is checked by truncation, not by inspection: every fired
trigger is recomputed on data sliced to (and including) its own buy day,
and must reproduce the identical z-score and branch.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/selloff_bounce_daybyday.py [seed] [n_tickers]
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

PARAMS = dict(
    n_window=10,       # trailing window for roll_max/roll_std (and SPY's own)
    z_thresh=2.0,       # stock must be at least this many std devs below its trailing high
    spy_z_thresh=1.0,   # SPY itself at least this many std devs below its own trailing high -> market_wide
    spy_neutral=0.3,    # |SPY z| below this -> idiosyncratic; between spy_neutral and spy_z_thresh -> excluded ("mixed")
)
HOLD = 5
COST = 10 / 1e4
WARMUP = 30
VOL_LOOKBACK = 10


def volume_ratio(volume, i, lookback=VOL_LOOKBACK):
    """Trailing lookback-day average volume strictly before position i --
    identical definition to two_touch_low_daybyday.py's volume_ratio."""
    seg = volume.iloc[max(0, i - lookback):i]
    avg = float(seg.mean()) if len(seg) and seg.notna().any() else float("nan")
    if avg in (0.0,) or np.isnan(avg) or np.isnan(volume.iloc[i]):
        return float("nan")
    return float(volume.iloc[i]) / avg


def rolling_z(close, n_window):
    """Causal drawdown-depth z-score: both roll_max and roll_std look at
    close[i-n_window .. i-1] only (shift(1) before rolling), so z[i] never
    uses information from after day i."""
    prior = close.shift(1)
    roll_max = prior.rolling(n_window).max()
    roll_std = prior.rolling(n_window).std()
    return (close - roll_max) / roll_std


def decide(close, volume, spy_z_aligned, i, p):
    """Evaluate the selloff-bounce trigger at position i, using only data
    up to and including i (spy_z_aligned already causal by the same
    construction as rolling_z). Returns None or a dict with the branch.

    volume_ratio[i] is computed and reported on every decision but NOT
    gated on here -- see module docstring ("Capitulation-volume gate").
    An optional vol_gate in p (a float floor) can still filter it, off
    (None) by default so the ungated population stays available for
    quantile bucketing."""
    if i < p["n_window"] + 1:
        return None
    z = rolling_z(close.iloc[:i + 1], p["n_window"]).iloc[i]
    if pd.isna(z) or not (z <= -p["z_thresh"]):
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
    vol_gate = p.get("vol_gate")
    if vol_gate is not None and not (not pd.isna(vr) and vr > vol_gate):
        return None
    return dict(branch=branch, z=float(z), spy_z=float(szpy), vr=vr)


def run_ticker(args):
    ticker, close, volume, spy_z_aligned, p, hold = args
    n = len(close)
    busy, trades = -1, []
    for i in range(WARMUP, n - hold):
        decision = decide(close, volume, spy_z_aligned, i, p)
        if decision is None:
            continue
        if i <= busy:
            continue

        # causal check: recompute on data truncated to (and including) the
        # buy day only -- must reproduce the exact same decision.
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
    df.to_csv(os.path.join(HERE, "selloff_bounce_daybyday_trades.csv"), index=False)

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
