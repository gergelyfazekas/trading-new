"""Close-location-value (CLV) on the selloff trigger day -- correlation-
first test, not a new gate, same methodology as this file's earlier
two-touch-low CLV test (mean_reversion_notes.md, 2026-09-23, "Close-
location-value (clv) on the low day, tested and rejected"). User
hypothesis there: a qualifying day is more likely to bounce if it closed
closer to its own high than its own low (hammer/pin-bar absorption).
That test found the OPPOSITE sign twice on the two-touch-low candidate
(weak-into-the-close outperformed) -- untested until now on the
independent "big selloff" premise.

Takes the existing Stage 1 trigger population UNCHANGED (both selloff
definitions, drawdown-depth z-score and streak z-score -- same params as
selloff_bounce_consol_daybyday.py's defaults), computes
clv[L] = (close[L]-low[L])/(high[L]-low[L]) for each qualifying day, and
correlates (Spearman) against two outcomes measured directly from L,
bypassing any buy-trigger/consolidation machinery entirely: pct1
(next-day move) and fwd5 (5-day forward return). No busy-ticker
deduplication either -- same as the two-touch-low precedent, this is a
population-level correlation check, not a portfolio simulation.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/selloff_bounce_clv_correlation.py [seed] [n_tickers]
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
from tech_level_live import pull_all
from tech_level_continuation_live import LIVE_TICKERS
from selloff_bounce_daybyday import rolling_z, volume_ratio
from selloff_bounce_consol_daybyday import stage1_z

DRAWDOWN_PARAMS = dict(selloff_type="drawdown", n_window=10, z_thresh=2.5,
                        spy_z_thresh=1.0, spy_neutral=0.3)
STREAK_PARAMS = dict(selloff_type="streak", dist_window=120, z_thresh=2.0, mode="decline",
                      spy_z_thresh=1.0, spy_neutral=0.3)
SPY_N_WINDOW = 10
WARMUP_FLOOR = 30
N_BUCKETS = 5


def gather_population(close, high, low, volume, spy_z_aligned, p):
    n = len(close)
    warmup = max(WARMUP_FLOOR, p.get("dist_window", 0) + 5, p.get("n_window", 0) + 1)
    rows = []
    for i in range(warmup, n - 5 - 1):
        z = stage1_z(close, i, p)
        if z is None:
            continue
        szpy = spy_z_aligned.iloc[i]
        if pd.isna(szpy):
            continue
        if szpy <= -p["spy_z_thresh"]:
            branch = "market_wide"
        elif szpy > -p["spy_neutral"]:
            branch = "idiosyncratic"
        else:
            continue
        hi, lo, cl = float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
        if pd.isna(hi) or pd.isna(lo) or hi == lo:
            continue
        clv = (cl - lo) / (hi - lo)
        pct1 = (float(close.iloc[i + 1]) - cl) / cl
        fwd5 = (float(close.iloc[i + 5]) - cl) / cl
        vr = volume_ratio(volume, i)
        rows.append(dict(branch=branch, clv=clv, pct1=pct1, fwd5=fwd5, vr=vr, z=z))
    return pd.DataFrame(rows)


def report(df, label):
    print(f"\n=== {label} (n={len(df)}) ===")
    if len(df) < 20:
        print("  too few observations")
        return
    for col in ["pct1", "fwd5"]:
        sub = df.dropna(subset=["clv", col])
        rho, p = spearmanr(sub.clv, sub[col])
        print(f"  clv vs {col:6}  rho={rho:+.3f}  p={p:.3f}  n={len(sub)}")
    sub = df.dropna(subset=["clv", "vr"])
    if len(sub) > 20:
        rho, p = spearmanr(sub.clv, sub.vr)
        print(f"  clv vs vr (confound check)  rho={rho:+.3f}  p={p:.3f}  n={len(sub)}")

    sub = df.dropna(subset=["clv", "fwd5"]).copy()
    if len(sub) >= N_BUCKETS * 5:
        sub["bucket"] = pd.qcut(sub.clv, N_BUCKETS, labels=[f"Q{i+1}" for i in range(N_BUCKETS)])
        print("  fwd5 by clv quantile bucket (Q1=close near low, Q5=close near high):")
        for b in [f"Q{i+1}" for i in range(N_BUCKETS)]:
            g = sub[sub.bucket == b]
            print(f"    {b}  n={len(g):4d}  mean fwd5={g.fwd5.mean():+.3%}")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}", flush=True)

    series, failed = pull_all(tickers + ["SPY"], include_hl=True)
    if failed:
        print(f"no data for: {failed}")
    today = datetime.date.today()

    spy_c = series["SPY"][0].copy()
    spy_c.index = pd.to_datetime(spy_c.index)
    spy_c = spy_c[spy_c.index.date < today]
    spy_z_full = rolling_z(spy_c, SPY_N_WINDOW)

    close, high, low, volume = {}, {}, {}, {}
    for t in tickers:
        if t not in series:
            continue
        c, h, l, v = series[t]
        c = c.copy(); c.index = pd.to_datetime(c.index)
        mask = c.index.date < today
        close[t] = c[mask]
        h = h.copy(); h.index = pd.to_datetime(h.index)
        high[t] = h.reindex(close[t].index)
        l = l.copy(); l.index = pd.to_datetime(l.index)
        low[t] = l.reindex(close[t].index)
        if v is not None:
            v = v.copy(); v.index = pd.to_datetime(v.index)
            volume[t] = v.reindex(close[t].index)
        else:
            volume[t] = pd.Series(np.nan, index=close[t].index)

    spy_z_aligned = {t: spy_z_full.reindex(close[t].index, method="ffill") for t in close}

    for label, p in [("drawdown", DRAWDOWN_PARAMS), ("streak", STREAK_PARAMS)]:
        frames = []
        for t in close:
            df_t = gather_population(close[t], high[t], low[t], volume[t], spy_z_aligned[t], p)
            frames.append(df_t)
        df = pd.concat(frames, ignore_index=True)
        report(df, label)
        for br in ["market_wide", "idiosyncratic"]:
            report(df[df.branch == br], f"{label} / {br}")


if __name__ == "__main__":
    main()
