"""Sensitivity checks for the naive support-strategy edge found after adding
tech_levels.mark_broken (levels retired once price fully crosses the band).

A rule change that took pooled hold=5 excess from -0.03% (t=-1.3) to +0.41%
(t=8.2) on the OOS universe deserves stress-testing before being trusted, not
just accepted because year/stock breadth checked out. Three independent
checks here, all on the same fixed a-priori combo and the same two universes
(config.ticker_list, config.oos_ticker_list minus NVDA -- see
tech_level_oos_strategy.py for why NVDA is dropped):

  1. near_pct x hold_days grid -- is the reported (near_pct=1%, hold_days=5)
     cell a coincidence, or does the edge hold across the neighbourhood of
     reasonable execution parameters? Reports the whole grid, not the best
     cell, to avoid just relocating the cherry-pick.
  2. cost_bps sweep -- how much of the edge survives realistic round-trip
     costs, and roughly where the breakeven cost sits.
  3. Level age at entry -- is the edge concentrated in levels that were just
     born (i.e. arguably a short-term continuation/momentum effect wearing a
     "support" label), or does it persist for levels that have stood
     unbroken for a while?

Run: ./venv/bin/python tech_level_sensitivity.py
"""
import datetime

import numpy as np
import pandas as pd

import config
from stock_class import StockList
from tech_level_naive_strategy import load_fixed_combo, build_levels, simulate, equal_weight_curve, attach_benchmark

EXCLUDE_OOS = {"NVDA"}  # see tech_level_oos_strategy.py docstring
START_DATE = datetime.date(2015, 5, 28)


def load_universe(which):
    if which == "ticker_list":
        tickers = list(config.ticker_list)
        sl = StockList(tickers)
        sl.pull_data(start=START_DATE, end=datetime.date.today())
    else:
        tickers = [t for t in config.oos_ticker_list if t not in EXCLUDE_OOS]
        sl = StockList(tickers)
        sl.load_data(config.oos)

    series = {}
    for t in tickers:
        try:
            c = sl[t].data["close"].dropna()
            c.index = pd.to_datetime(c.index)
            if not c.empty:
                series[t] = c
        except Exception:
            continue
    return series


def pooled(trades_df):
    if trades_df.empty:
        return {"n": 0, "mean_excess": np.nan, "t": np.nan}
    n = len(trades_df)
    mean = trades_df.excess_ret.mean()
    t = mean / trades_df.excess_ret.std() * np.sqrt(n) if n > 2 else np.nan
    return {"n": n, "mean_excess": mean, "t": t}


def grid_sweep(series, ew, levels_by_ticker, near_pcts, hold_days_list):
    rows = []
    for near_pct in near_pcts:
        for hold_days in hold_days_list:
            all_trades = []
            for t, c in series.items():
                all_trades.extend(simulate(t, c, levels_by_ticker[t], hold_days=hold_days, near_pct=near_pct))
            trades_df = pd.DataFrame(all_trades)
            if not trades_df.empty:
                trades_df = attach_benchmark(trades_df, ew)
            rows.append({"near_pct": near_pct, "hold_days": hold_days, **pooled(trades_df)})
    return pd.DataFrame(rows)


def cost_sweep(base_trades_df, ew, cost_bps_list):
    rows = []
    for cost_bps in cost_bps_list:
        df = base_trades_df.copy()
        df["ret_net"] = df["ret"] - cost_bps / 1e4
        df = attach_benchmark(df, ew)
        rows.append({"cost_bps": cost_bps, **pooled(df)})
    return pd.DataFrame(rows)


def age_bucket_sweep(base_trades_df, bins):
    df = base_trades_df.dropna(subset=["support_age_days"]).copy()
    labels = [f"{lo}-{hi}d" for lo, hi in zip(bins[:-1], bins[1:])]
    df["age_bucket"] = pd.cut(df["support_age_days"], bins=bins, labels=labels, right=False)
    rows = []
    for bucket, g in df.groupby("age_bucket", observed=True):
        rows.append({"age_bucket": bucket, **pooled(g)})
    return pd.DataFrame(rows)


def run_for_universe(which):
    print(f"\n########## {which} ##########")
    combo = load_fixed_combo()
    series = load_universe(which)
    print(f"{len(series)} tickers loaded")

    ew = equal_weight_curve(series)
    levels_by_ticker = {t: build_levels(c, combo) for t, c in series.items()}

    print("\n=== near_pct x hold_days grid (pooled excess, t-stat) ===")
    grid = grid_sweep(series, ew, levels_by_ticker,
                       near_pcts=[0.005, 0.01, 0.015, 0.02, 0.03],
                       hold_days_list=[1, 2, 3, 5, 7, 10])
    pivot_mean = grid.pivot(index="near_pct", columns="hold_days", values="mean_excess") * 100
    pivot_t = grid.pivot(index="near_pct", columns="hold_days", values="t")
    print("mean_excess (%):")
    print(pivot_mean.round(3).to_string())
    print("\nt-stat:")
    print(pivot_t.round(2).to_string())

    # base run for cost + age sensitivity: the already-reported cell
    base_trades = []
    for t, c in series.items():
        base_trades.extend(simulate(t, c, levels_by_ticker[t], hold_days=5, near_pct=0.01, cost_bps=0.0))
    base_df = pd.DataFrame(base_trades)
    base_df = attach_benchmark(base_df, ew)

    print("\n=== cost_bps sweep (near_pct=1%, hold_days=5) ===")
    costs = cost_sweep(base_df, ew, [0, 5, 10, 15, 20, 30, 40, 50])
    print(costs.round(4).to_string(index=False))

    print("\n=== level age at entry (near_pct=1%, hold_days=5, cost=10bp) ===")
    base_df_10bp = base_df.copy()
    base_df_10bp["ret_net"] = base_df_10bp["ret"] - 10.0 / 1e4
    base_df_10bp = attach_benchmark(base_df_10bp, ew)
    ages = age_bucket_sweep(base_df_10bp, bins=[0, 5, 20, 60, 180, 365, 100000])
    print(ages.round(4).to_string(index=False))

    return grid, costs, ages


if __name__ == "__main__":
    for universe in ("oos", "ticker_list"):
        run_for_universe(universe)
