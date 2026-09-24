"""Correlation-first side test of a new candidate feature: does the LOW
DAY's own candle shape predict the bounce, on top of (or independent of)
the existing two-touch-low Stage 1 gates?

Hypothesis (user, 2026-09-23): a local low at day L is more likely to
bounce if day L itself closed closer to its own high than its own low --
i.e. intraday sellers pushed price down but buyers took it back before
the close, classic hammer/pin-bar absorption. This is a different axis
from the existing volume_ratio[L] gate (which measures how much activity
happened, not which side of it won).

clv[L] ("close location value") = (close[L] - low[L]) / (high[L] - low[L])
in [0, 1]; clv near 1 means the close sat near the day's high.

Candidate population is EXACTLY Stage 1 of the locked two-touch-low rule
(two_touch_low_daybyday.decide()'s first three checks, unchanged) -- this
says something about the existing candidate's own qualifying lows, not
some other definition of "low". Outcomes are measured directly from L,
bypassing the Stage 2 buy-trigger machinery entirely, since the
hypothesis is about the low day itself, not about the two-touch pattern
completing.

Same discipline as the rest of mean_reversion_notes.md: Spearman rho
(returns are noisy/heavy-tailed, no linearity assumed), quantile-bucketed
robustness check (not a naive threshold sweep -- see the file's own
"Reminder for later sweeps"), year-by-year breakdown, and an explicit
confound check against volume_ratio[L] before treating any correlation
as independent information.

Pure read-only diagnostic. Nothing here touches decide() or the live
script -- if this survives, the next step is turning it into an actual
Stage 1 gate and re-running the causal day-by-day harness with costs.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_clv_correlation.py [seed] [n_tickers]
"""
import datetime
import os
import random
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from stock_class import StockList
from tech_level_continuation_live import LIVE_TICKERS
from two_touch_low_daybyday import PARAMS, volume_ratio, WARMUP

START_DATE = datetime.date(2015, 5, 28)
FWD_HOLD = 5


def pull_all_ohlc(tickers):
    """Same shape as tech_level_live.pull_all, but also keeps high/low --
    those columns already come back from yf.download via StockList, pull_all
    just never extracted them because nothing needed intraday range before."""
    end = datetime.date.today() + datetime.timedelta(days=1)
    s = StockList(list(tickers))
    s.pull_data(start=START_DATE, end=end)

    series, failed = {}, []
    for ticker in tickers:
        try:
            data = s[ticker].data
            close = data['close'].dropna()
            if close.empty:
                failed.append(ticker)
                continue
            volume = data['volume'].reindex(close.index) if 'volume' in data.columns else None
            high = data['high'].reindex(close.index) if 'high' in data.columns else None
            low = data['low'].reindex(close.index) if 'low' in data.columns else None
            series[ticker] = (close, volume, high, low)
        except Exception:
            failed.append(ticker)
    return series, failed


def qualifying_lows(close, volume, high, low, p):
    """Stage 1 of decide() (two_touch_low_daybyday.py), unchanged, plus
    clv[L] and the two outcome variables measured directly from L."""
    n = len(close)
    rows = []
    for L in range(WARMUP, n - FWD_HOLD - 1):
        window = close.iloc[L - p["lookback"]:L]
        if not bool((close.iloc[L] < window).all()):
            continue
        if not (close.iloc[L - 1] >= close.iloc[L] * (1 + p["prominence_pct"])):
            continue
        vr_L = volume_ratio(volume, L)
        if not (vr_L > p["volume_threshold"]):
            continue

        h, lo, c = float(high.iloc[L]), float(low.iloc[L]), float(close.iloc[L])
        if not (h > lo):  # zero-range day: undefined, drop
            continue
        clv = (c - lo) / (h - lo)

        c_L1 = float(close.iloc[L + 1])
        pct1 = (c_L1 - c) / c
        c_fwd = float(close.iloc[L + FWD_HOLD])
        fwd5 = (c_fwd - c) / c

        rows.append(dict(date=close.index[L], clv=clv, vr_L=vr_L, pct1=pct1, fwd5=fwd5,
                          year=close.index[L].year))
    return rows


def bucket_table(df, feature, outcome, q=5):
    df = df.dropna(subset=[feature, outcome])
    df = df.copy()
    df["bucket"] = pd.qcut(df[feature], q, labels=[f"Q{i+1}" for i in range(q)], duplicates="drop")
    return df.groupby("bucket", observed=True)[outcome].agg(["mean", "count"])


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}", flush=True)
    print(f"params (Stage 1 unchanged): lookback={PARAMS['lookback']}, "
          f"prominence_pct={PARAMS['prominence_pct']}, volume_threshold={PARAMS['volume_threshold']}\n",
          flush=True)

    series, failed = pull_all_ohlc(tickers)
    if failed:
        print(f"no data for: {failed}")
    today = datetime.date.today()

    all_rows = []
    for t in tickers:
        if t not in series:
            continue
        close, volume, high, low = series[t]
        close = close.copy(); close.index = pd.to_datetime(close.index)
        mask = close.index.date < today
        close = close[mask]
        volume = volume.copy(); volume.index = pd.to_datetime(volume.index); volume = volume.reindex(close.index)
        high = high.copy(); high.index = pd.to_datetime(high.index); high = high.reindex(close.index)
        low = low.copy(); low.index = pd.to_datetime(low.index); low = low.reindex(close.index)

        rows = qualifying_lows(close, volume, high, low, PARAMS)
        for r in rows:
            r["stock"] = t
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("no qualifying lows found")
        return
    print(f"{len(df)} qualifying lows (Stage 1 passes) across {df.stock.nunique()} tickers\n")

    for outcome in ["pct1", "fwd5"]:
        rho, p = spearmanr(df["clv"], df[outcome])
        print(f"clv vs {outcome}: rho={rho:+.3f}, p={p:.3f}, n={len(df)}")

    rho, p = spearmanr(df["clv"], df["vr_L"])
    print(f"\nconfound check -- clv vs volume_ratio[L]: rho={rho:+.3f}, p={p:.3f}")

    print("\nquantile buckets (clv), pct1:")
    print(bucket_table(df, "clv", "pct1"))
    print("\nquantile buckets (clv), fwd5:")
    print(bucket_table(df, "clv", "fwd5"))

    print("\nby year -- rho(clv, fwd5):")
    for y, g in df.groupby("year"):
        if len(g) < 10:
            print(f"  {y}: n={len(g)} (too few, skipped)")
            continue
        rho, p = spearmanr(g["clv"], g["fwd5"])
        print(f"  {y}: n={len(g):4d}  rho={rho:+.3f}  p={p:.3f}")


if __name__ == "__main__":
    main()
