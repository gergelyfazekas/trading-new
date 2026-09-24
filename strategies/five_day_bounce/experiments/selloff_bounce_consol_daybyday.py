"""Day-by-day (causal) test of "big selloff, then a sideways consolidation,
then buy" -- adds a Stage 2 to the two rejected selloff-magnitude triggers
(Phase 0/1's drawdown-depth z-score, and the streak-based z-score) instead
of buying immediately at the selloff day. mean_reversion_notes.md, session
of 2026-09-23.

Stage 1 (selloff day L) -- unchanged from the earlier rejected
candidates, either definition selectable via p["selloff_type"]:
  "drawdown": z[L] = (close[L] - roll_max(close,n_window)[L]) / roll_std(...)[L],
              fires when z[L] <= -z_thresh (selloff_bounce_daybyday.rolling_z)
  "streak":   z[L] = z-scored consecutive-down-day streak severity
              (length or decline) vs. its own trailing dist_window-day
              history, fires when z[L] >= z_thresh
              (selloff_bounce_streak_daybyday.compute_severity_z)
Same SPY market-wide/idiosyncratic split as both earlier candidates,
unchanged (own dedicated spy_n_window, decoupled from whichever
stock-side window the selloff definition uses).

Stage 2 (new) -- sideways consolidation, exactly consol_days long:
  requires close[L+1 .. L+consol_days] to ALL sit within
  [close[L]*(1-band_pct), close[L]*(1+band_pct)] -- no breakout either
  direction, i.e. actually sideways, not "still falling" or "already
  ripping back." Buy fires at close[L+consol_days] if every one of those
  days held inside the band; otherwise no trade for this L. consol_days
  is a fixed parameter per run (swept 1-5), not an auto-detected
  as-long-as-it-holds window -- each value is tested as its own
  hypothesis, not the greedy max.

Exit: fixed hold-session hold from the buy day (swept 3-10), 10bps cost,
one position per ticker. Causality verified by truncation, same
convention as this file's whole history: Stage 2 only ever looks at
L+1..L+consol_days, all in the past relative to the buy day itself, so
truncating to (and including) the buy day must reproduce the identical
decision.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/selloff_bounce_consol_daybyday.py [seed] [n_tickers]
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
from selloff_bounce_daybyday import rolling_z, volume_ratio, COST
from selloff_bounce_streak_daybyday import compute_severity_z

PARAMS = dict(
    selloff_type="drawdown",   # "drawdown" or "streak"
    n_window=10,                # drawdown-type stock trigger window
    dist_window=120,            # streak-type severity baseline window
    z_thresh=2.5,                # drawdown default; streak sweep overrides
    mode="decline",              # streak only
    spy_n_window=10,             # SPY's own drawdown z-score window (fixed, decoupled from stock side)
    spy_z_thresh=1.0,
    spy_neutral=0.3,
    band_pct=0.02,
    consol_days=3,
    hold=5,
)
WARMUP_FLOOR = 30


def stage1_z(close, i, p):
    """Selloff-magnitude score at day i, either definition. Causal by
    construction (only looks at close[..i])."""
    if p["selloff_type"] == "drawdown":
        n_window = p["n_window"]
        if i < n_window + 1:
            return None
        z = rolling_z(close.iloc[:i + 1], n_window).iloc[i]
        if pd.isna(z) or not (z <= -p["z_thresh"]):
            return None
        return float(z)
    else:
        dist_window = p["dist_window"]
        warmup = max(WARMUP_FLOOR, dist_window + 5)
        if i < warmup:
            return None
        z, _, _ = compute_severity_z(close.iloc[:i + 1], dist_window, p["mode"])
        zi = z.iloc[i]
        if pd.isna(zi) or not (zi >= p["z_thresh"]):
            return None
        return float(zi)


def decide(close, volume, spy_z_aligned, i, p):
    """Stage 1 (selloff) + Stage 2 (sideways consolidation for exactly
    consol_days) + SPY market-wide/idiosyncratic split. Returns None or a
    dict with the branch and buy_offset (= consol_days, buy fires at
    close[i + consol_days])."""
    z = stage1_z(close, i, p)
    if z is None:
        return None
    szpy = spy_z_aligned.iloc[i]
    if pd.isna(szpy):
        return None
    if szpy <= -p["spy_z_thresh"]:
        branch = "market_wide"
    elif szpy > -p["spy_neutral"]:
        branch = "idiosyncratic"
    else:
        return None  # mixed zone, excluded

    n = len(close)
    K = p["consol_days"]
    if i + K >= n:
        return None
    c_L = float(close.iloc[i])
    lo, hi = c_L * (1 - p["band_pct"]), c_L * (1 + p["band_pct"])
    for j in range(i + 1, i + K + 1):
        cj = float(close.iloc[j])
        if not (lo <= cj <= hi):
            return None

    vr = volume_ratio(volume, i)
    return dict(branch=branch, z=z, spy_z=float(szpy), vr=vr, buy_offset=K)


def run_ticker(args):
    ticker, close, volume, spy_z_aligned, p, hold = args
    n = len(close)
    busy, trades = -1, []
    warmup = max(WARMUP_FLOOR, p.get("dist_window", 0) + 5, p.get("n_window", 0) + 1)
    max_lookahead = p["consol_days"] + hold + 1
    for i in range(warmup, n - max_lookahead):
        decision = decide(close, volume, spy_z_aligned, i, p)
        if decision is None:
            continue
        buy_day = i + decision["buy_offset"]
        if buy_day <= busy:
            continue

        trunc_close = close.iloc[:buy_day + 1]
        trunc_volume = volume.iloc[:buy_day + 1]
        trunc_spy = spy_z_aligned.iloc[:buy_day + 1]
        redecision = decide(trunc_close, trunc_volume, trunc_spy, i, p)
        assert redecision == decision, f"leakage: {ticker} i={close.index[i].date()} {decision} vs {redecision}"

        exit_i = buy_day + hold
        if exit_i >= n:
            continue
        ret = float(close.iloc[exit_i] / close.iloc[buy_day] - 1)
        trades.append(dict(
            stock=ticker, selloff_date=close.index[i], entry_date=close.index[buy_day], exit_date=close.index[exit_i],
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
    spy_z_full = rolling_z(spy_c, PARAMS["spy_n_window"])

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
        out = pool.map(run_ticker, [(t, close[t], volume[t], spy_z_aligned[t], PARAMS, PARAMS["hold"]) for t in close])
    df = pd.DataFrame([t for o in out for t in o])
    if df.empty:
        print("no trades fired at these parameters")
        return
    df = attach_benchmark(df, ew)
    df.to_csv(os.path.join(HERE, "selloff_bounce_consol_daybyday_trades.csv"), index=False)

    rng = np.random.default_rng(0)
    rows = []
    for t, cnt in df.groupby("stock").size().items():
        c = close[t]
        for i in rng.choice(np.arange(WARMUP_FLOOR, len(c) - PARAMS["hold"]), size=cnt, replace=False):
            rows.append({"entry_date": c.index[i], "exit_date": c.index[i + PARAMS["hold"]],
                         "ret_net": c.iloc[i + PARAMS["hold"]] / c.iloc[i] - 1 - COST})
    pe = attach_benchmark(pd.DataFrame(rows), ew).excess_ret.mean()

    print(f"{len(close)} tickers, ~{min(len(c) for c in close.values())}+ sessions each\n")
    print("every trade's entry decision verified identical on truncated-to-buy-day data (causal by construction)\n")
    print(summ(df, "ALL (causal)"))
    print(f"  {'placebo excess (matched n)':22} {pe:+.3%}   causal lift = {df.excess_ret.mean() - pe:+.3%}")

    print("\nby branch (market_wide vs idiosyncratic):")
    for b, g in df.groupby("branch"):
        print(summ(g, b))


if __name__ == "__main__":
    main()
