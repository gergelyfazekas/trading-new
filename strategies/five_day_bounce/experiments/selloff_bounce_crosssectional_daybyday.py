"""Day-by-day (causal) test of notes' candidate #8 (cross-sectional rank
reversal) applied to the "big selloff" premise -- a genuinely different
mechanism from Phase 0-4 above, not another way of sizing the same
single-name signal. Instead of measuring a stock's decline against ITS
OWN trailing history (drawdown z-score, streak z-score), rank each
stock's trailing N-day return against its PEERS (the same 20-ticker
universe) on that specific date, and buy whichever names rank worst
that day. mean_reversion_notes.md, session of 2026-09-23.

  ret_N[t, ticker] = close[t]/close[t-N] - 1                 (causal)
  rank[t, ticker]  = percentile rank of ret_N[t, ticker] within the
                     cross-section of all tickers' ret_N[t] on that same
                     date t (pandas .rank(axis=1, pct=True); NaNs
                     excluded per row) -- causal, since day t's rank only
                     uses day-t values from every ticker, all known at
                     that day's close.

Fires when rank[t, ticker] <= bottom_pct (the worst X% of the
cross-section that day). No SPY split here -- the whole premise is
already relative-to-peers, and a market-wide selloff moves the *whole*
cross-section together, so this signal structurally selects the
relative laggards within whatever is happening that day, market-wide or
not, by construction. Buy at close[t], hold swept, 10bps cost, one
position per ticker.

Causality verified by truncation: rank[t] depends only on
close[t-N..t] across all tickers, so truncating every ticker's series to
(and including) t and recomputing must reproduce the identical rank.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/selloff_bounce_crosssectional_daybyday.py [seed] [n_tickers]
"""
import datetime
import os
import random
import sys

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

PARAMS = dict(n_return=5, bottom_pct=0.10, hold=5)
COST = 10 / 1e4
WARMUP = 30


def compute_rank_matrix(close_df, n_return):
    """Causal cross-sectional percentile rank of trailing n_return-day
    return, per row (date). Each ret_N[t] uses only close[t-n_return..t]
    for its own column, and rank[t] only compares values already present
    at row t -- no cross-time leakage in either direction."""
    ret_n = close_df / close_df.shift(n_return) - 1
    rank = ret_n.rank(axis=1, pct=True)
    return rank


def run_ticker_args(t, close_df, rank_df, p):
    close = close_df[t]
    rank = rank_df[t]
    n = len(close)
    hold = p["hold"]
    busy, trades = -1, []
    for i in range(WARMUP, n - hold):
        r = rank.iloc[i]
        if pd.isna(r) or not (r <= p["bottom_pct"]):
            continue
        if i <= busy:
            continue

        # causal check: recompute the rank at i using data truncated to
        # (and including) day i across every ticker, must match exactly.
        trunc = close_df.iloc[:i + 1]
        trunc_rank = compute_rank_matrix(trunc, p["n_return"])[t].iloc[i]
        assert (pd.isna(trunc_rank) and pd.isna(r)) or abs(trunc_rank - r) < 1e-9, \
            f"leakage: {t} i={close.index[i].date()} rank {r} vs {trunc_rank}"

        exit_i = i + hold
        if exit_i >= n:
            continue
        ret = float(close.iloc[exit_i] / close.iloc[i] - 1)
        trades.append(dict(
            stock=t, entry_date=close.index[i], exit_date=close.index[exit_i],
            rank=float(r), ret=ret, ret_net=ret - COST,
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
        c = c.copy()
        c.index = pd.to_datetime(c.index)
        close[t] = c[c.index.date < today]

    ew = equal_weight_curve(close)

    close_df = pd.DataFrame(close).dropna(how="any")  # inner join: only dates all tickers have
    rank_df = compute_rank_matrix(close_df, PARAMS["n_return"])

    trades = []
    for t in close_df.columns:
        trades.extend(run_ticker_args(t, close_df, rank_df, PARAMS))
    df = pd.DataFrame(trades)
    if df.empty:
        print("no trades fired at these parameters")
        return
    df = attach_benchmark(df, ew)
    df.to_csv(os.path.join(HERE, "selloff_bounce_crosssectional_trades.csv"), index=False)

    rng = np.random.default_rng(0)
    rows = []
    for t, cnt in df.groupby("stock").size().items():
        c = close[t]
        for i in rng.choice(np.arange(WARMUP, len(c) - PARAMS["hold"]), size=cnt, replace=False):
            rows.append({"entry_date": c.index[i], "exit_date": c.index[i + PARAMS["hold"]],
                         "ret_net": c.iloc[i + PARAMS["hold"]] / c.iloc[i] - 1 - COST})
    pe = attach_benchmark(pd.DataFrame(rows), ew).excess_ret.mean()

    print(f"{len(close_df.columns)} tickers, {len(close_df)} common sessions\n")
    print("every trade's rank verified identical on truncated-to-entry-day data (causal by construction)\n")
    print(summ(df, "ALL (causal)"))
    print(f"  {'placebo excess (matched n)':22} {pe:+.3%}   causal lift = {df.excess_ret.mean() - pe:+.3%}")

    print("\nby year:")
    for y, g in df.groupby(pd.DatetimeIndex(df.entry_date).year):
        print(summ(g, str(y)))

    print("\nper ticker:")
    for t in close_df.columns:
        print(summ(df[df.stock == t], t))


if __name__ == "__main__":
    main()
