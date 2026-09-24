"""Stress-test of selloff_bounce_clv_correlation.py's finding: raw fwd5
correlated negatively with clv[L] (weak-into-the-close outperforms) on
both selloff definitions, significantly on the streak population
(rho=-0.088, p<0.001, n=2986) and more weakly on drawdown (rho=-0.027,
p=0.003, n=11784), with a near-monotonic bucket spread on the streak
population (Q1 +1.02% -> Q5 -0.17%). Per
[[validation-first-quant-work]], that raw-fwd5 correlation has to survive
the same backtest rigor as everything else this session before it counts
for anything: real trades (busy-ticker tracking, no overlap), cost-net
EXCESS return over the equal-weight benchmark (not raw price return), a
matched-n placebo, and a t-stat -- not just a bucket table on an
unconstrained population.

Same Stage 1 selloff trigger as selloff_bounce_daybyday.py /
selloff_bounce_streak_daybyday.py (immediate buy at close[i], no
consolidation stage -- that was separately rejected), 5-day hold, 10bps
cost, one position per ticker. clv[i] = (close[i]-low[i])/(high[i]-low[i])
computed and reported on every trade, same "quantile-bucket, don't
threshold-sweep" discipline as the volume gate test. A concrete low-clv
gate (clv <= median of the fired population) is then tested as an actual
candidate rule, not just described as a bucket.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/selloff_bounce_clv_gate.py [seed] [n_tickers]
"""
import datetime
import os
import random
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from tech_level_naive_strategy import equal_weight_curve, attach_benchmark
from tech_level_live import pull_all
from tech_level_continuation_live import LIVE_TICKERS
from daybyday_variants import summ
from selloff_bounce_daybyday import rolling_z, volume_ratio, COST, HOLD
from selloff_bounce_consol_daybyday import stage1_z, WARMUP_FLOOR

DRAWDOWN_PARAMS = dict(selloff_type="drawdown", n_window=10, z_thresh=2.5,
                        spy_z_thresh=1.0, spy_neutral=0.3)
STREAK_PARAMS = dict(selloff_type="streak", dist_window=120, z_thresh=2.0, mode="decline",
                      spy_z_thresh=1.0, spy_neutral=0.3)
SPY_N_WINDOW = 10
N_BUCKETS = 5


def decide(close, high, low, volume, spy_z_aligned, i, p):
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
        return None
    hi, lo, cl = float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
    clv = (cl - lo) / (hi - lo) if (not pd.isna(hi) and not pd.isna(lo) and hi != lo) else float("nan")
    vr = volume_ratio(volume, i)
    return dict(branch=branch, z=z, spy_z=float(szpy), vr=vr, clv=clv)


def run_ticker(args):
    ticker, close, high, low, volume, spy_z_aligned, p, hold = args
    n = len(close)
    warmup = max(WARMUP_FLOOR, p.get("dist_window", 0) + 5, p.get("n_window", 0) + 1)
    busy, trades = -1, []
    for i in range(warmup, n - hold):
        decision = decide(close, high, low, volume, spy_z_aligned, i, p)
        if decision is None:
            continue
        if i <= busy:
            continue

        trunc_close = close.iloc[:i + 1]
        trunc_high = high.iloc[:i + 1]
        trunc_low = low.iloc[:i + 1]
        trunc_volume = volume.iloc[:i + 1]
        trunc_spy = spy_z_aligned.iloc[:i + 1]
        redecision = decide(trunc_close, trunc_high, trunc_low, trunc_volume, trunc_spy, i, p)
        assert redecision == decision or (pd.isna(decision["clv"]) and pd.isna(redecision["clv"])), \
            f"leakage: {ticker} i={close.index[i].date()} {decision} vs {redecision}"

        exit_i = i + hold
        if exit_i >= n:
            continue
        ret = float(close.iloc[exit_i] / close.iloc[i] - 1)
        trades.append(dict(
            stock=ticker, entry_date=close.index[i], exit_date=close.index[exit_i],
            branch=decision["branch"], z=decision["z"], spy_z=decision["spy_z"],
            vr=decision["vr"], clv=decision["clv"],
            ret=ret, ret_net=ret - COST,
        ))
        busy = exit_i
    return trades


def run_backtest(close, high, low, volume, spy_z_aligned, p, ew, close_dict):
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, [(t, close[t], high[t], low[t], volume[t], spy_z_aligned[t], p, HOLD) for t in close])
    df = pd.DataFrame([tr for o in out for tr in o])
    if df.empty:
        return df
    df = attach_benchmark(df, ew)
    return df


def placebo_excess(df, close_dict, ew, hold=HOLD):
    rng = np.random.default_rng(0)
    rows = []
    for t, cnt in df.groupby("stock").size().items():
        c = close_dict[t]
        for i in rng.choice(np.arange(WARMUP_FLOOR, len(c) - hold), size=cnt, replace=False):
            rows.append({"entry_date": c.index[i], "exit_date": c.index[i + hold],
                         "ret_net": c.iloc[i + hold] / c.iloc[i] - 1 - COST})
    return attach_benchmark(pd.DataFrame(rows), ew).excess_ret.mean()


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

    ew = equal_weight_curve(close)
    spy_z_aligned = {t: spy_z_full.reindex(close[t].index, method="ffill") for t in close}

    for label, p in [("drawdown", DRAWDOWN_PARAMS), ("streak", STREAK_PARAMS)]:
        df = run_backtest(close, high, low, volume, spy_z_aligned, p, ew, close)
        if df.empty:
            print(f"\n=== {label}: no trades ===")
            continue
        df = df.dropna(subset=["clv"])
        pe = placebo_excess(df, close, ew)

        print(f"\n=== {label} (n={len(df)}) ===")
        print(summ(df, "ALL (ungated on clv)"))
        print(f"  {'placebo excess (matched n)':22} {pe:+.3%}   causal lift = {df.excess_ret.mean() - pe:+.3%}")

        rho, pval = spearmanr(df.clv, df.excess_ret)
        print(f"  Spearman(clv, excess_ret): rho={rho:+.3f}  p={pval:.3f}  n={len(df)}")

        df["bucket"] = pd.qcut(df.clv, N_BUCKETS, labels=[f"Q{i+1}" for i in range(N_BUCKETS)])
        print("  by clv quantile bucket (Q1=close near low, Q5=close near high):")
        for b in [f"Q{i+1}" for i in range(N_BUCKETS)]:
            g = df[df.bucket == b]
            print("   ", summ(g, b))

        # concrete gate: bottom half of clv (below this population's own median)
        med = df.clv.median()
        gated = df[df.clv <= med]
        gated_pe = placebo_excess(gated, close, ew)
        print(f"\n  low-clv gate (clv <= median={med:.3f}):")
        print("   ", summ(gated, f"clv<={med:.2f}"))
        print(f"    {'placebo excess (matched n)':20} {gated_pe:+.3%}   causal lift = {gated.excess_ret.mean() - gated_pe:+.3%}")
        for br in ["market_wide", "idiosyncratic"]:
            sub = gated[gated.branch == br]
            if len(sub) > 5:
                print("   ", summ(sub, f"  {br}"))


if __name__ == "__main__":
    main()
