"""Direct test of the "provisional levels" caveat in NOTES.md / tech_levels_notes.md:
the naive-strategy backtest runs find_touches once over each ticker's *whole* history,
so a touch born <5 calendar days before a trade may have only been confirmed as a real
peak/trough because the detector could see up to `distance`=10 sessions of price action
*after* that date -- data a live trader at entry time would not have had yet. The notes
flag this as likely making the +0.41%/+0.27% (t=8.18/4.20) headline numbers optimistic,
but never quantified it. Two independent checks here:

  1. Per-trade causal recheck (the direct test). For every already-logged young-level
     trade (support_age_days < 5, hold_days=5, near_pct=1%, in naive_strategy_trades.csv /
     oos_strategy_trades.csv), truncate that ticker's close series to *only* data up to
     and including the entry date, rebuild levels from that truncated series with the
     exact same fixed combo, and check whether the same buy condition still fires. A
     trade that fires on the full-history levels but not on the truncated ones is a
     hindsight artifact -- it could not have been taken live. Reports the survival rate
     and the pooled excess return / t-stat of the surviving subset only, vs. the original
     full (hindsight) set.
  2. distance sensitivity sweep (cheap complementary check, not a substitute for #1 --
     it changes what counts as a touch at all, not just how much lookahead a touch needs,
     so a held-up effect here is reassuring but not proof; a collapsed one is suggestive
     but confounded). Rebuilds levels at distance in {3, 5, 10, 15, 20} and re-scores the
     same age<5-day bucket at each.

Run (from repo root): ./venv/bin/python strategies/five_day_bounce/tech_level_causal_check.py
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import config
from stock_class import StockList
from tech_level_naive_strategy import (
    load_fixed_combo, build_levels, active_support_resistance,
    simulate, equal_weight_curve, attach_benchmark,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EXCLUDE_OOS = {"NVDA"}  # see tech_level_oos_strategy.py docstring
START_DATE = datetime.date(2015, 5, 28)
NEAR_PCT = 0.01
HOLD_DAYS = 5
AGE_CUTOFF_DAYS = 5


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


def causal_fires(close_full, entry_date, entry_price, combo):
    """Would the buy rule fire using only data up to and including entry_date?"""
    trunc = close_full.loc[:pd.Timestamp(entry_date)]
    levels = build_levels(trunc, combo)
    support, _ = active_support_resistance(levels, pd.Timestamp(entry_date), entry_price)
    if support is None:
        return False
    dist = (entry_price - support.band[1]) / entry_price
    return 0 <= dist <= NEAR_PCT


def per_trade_causal_check(which, series, trades_path):
    print(f"\n########## per-trade causal recheck: {which} ##########")
    combo = load_fixed_combo()
    trades_df = pd.read_csv(trades_path, parse_dates=["entry_date", "exit_date"])
    young = trades_df[(trades_df.hold_days == HOLD_DAYS) & (trades_df.near_pct == NEAR_PCT) &
                       (trades_df.support_age_days < AGE_CUTOFF_DAYS)].copy()
    young = young[young.stock.isin(series.keys())]
    print(f"{len(young)} young-level (<{AGE_CUTOFF_DAYS}d) trades to recheck")

    survives = []
    for row in young.itertuples():
        close_full = series[row.stock]
        try:
            ok = causal_fires(close_full, row.entry_date, row.entry_price, combo)
        except Exception:
            ok = False
        survives.append(ok)
    young["causally_confirmed"] = survives

    ew = equal_weight_curve(series)
    young = attach_benchmark(young, ew)

    full_stats = pooled(young)
    surv_stats = pooled(young[young.causally_confirmed])

    print(f"survival rate: {young.causally_confirmed.mean():.1%} "
          f"({young.causally_confirmed.sum()}/{len(young)})")
    print(f"full (hindsight) set   -- n={full_stats['n']:5d}  "
          f"mean_excess={full_stats['mean_excess']:+.4%}  t={full_stats['t']:.2f}")
    print(f"causally-confirmed only -- n={surv_stats['n']:5d}  "
          f"mean_excess={surv_stats['mean_excess']:+.4%}  t={surv_stats['t']:.2f}")

    young.to_csv(os.path.join(DATA_DIR, f"causal_check_{which}.csv"), index=False)
    return young


def concentration_report(which, young):
    """Do the causally-confirmed survivors cluster in a few tickers/years, or
    spread broadly? A real, general effect should survive being broad; an
    effect propped up by a handful of names/periods is much weaker evidence.
    """
    surv = young[young.causally_confirmed].copy()
    surv["year"] = pd.DatetimeIndex(surv.entry_date).year
    n = len(surv)
    print(f"\n--- concentration of causally-confirmed survivors: {which} (n={n}) ---")

    by_stock = surv.groupby("stock").agg(n=("excess_ret", "size"), total_excess=("excess_ret", "sum"))
    by_stock = by_stock.sort_values("total_excess", ascending=False)
    total_excess_all = by_stock["total_excess"].sum()
    by_stock["pct_of_trades"] = by_stock["n"] / n
    by_stock["pct_of_total_excess"] = by_stock["total_excess"] / total_excess_all
    print(f"\n{surv.stock.nunique()} distinct tickers among survivors")
    print("top 10 tickers by total excess return contributed:")
    print(by_stock.head(10).round(4).to_string())
    top5_trade_share = by_stock["n"].head(5).sum() / n
    top5_excess_share = by_stock["total_excess"].head(5).sum() / total_excess_all
    print(f"top 5 tickers: {top5_trade_share:.1%} of trades, {top5_excess_share:.1%} of total excess return")

    by_year = surv.groupby("year").agg(n=("excess_ret", "size"), mean_excess=("excess_ret", "mean"),
                                        total_excess=("excess_ret", "sum"))
    by_year["pct_of_trades"] = by_year["n"] / n
    print("\nby year:")
    print(by_year.round(4).to_string())


def distance_sensitivity(which, series, distances):
    print(f"\n########## distance sensitivity: {which} ##########")
    base_combo = load_fixed_combo()
    ew = equal_weight_curve(series)
    rows = []
    for d in distances:
        combo = dict(base_combo)
        combo["distance"] = d
        levels_by_ticker = {t: build_levels(c, combo) for t, c in series.items()}
        all_trades = []
        for t, c in series.items():
            all_trades.extend(simulate(t, c, levels_by_ticker[t], hold_days=HOLD_DAYS, near_pct=NEAR_PCT))
        trades_df = pd.DataFrame(all_trades)
        if trades_df.empty:
            rows.append({"distance": d, "n_all": 0, "n_young": 0, "mean_excess": np.nan, "t": np.nan})
            continue
        trades_df = attach_benchmark(trades_df, ew)
        young = trades_df[trades_df.support_age_days < AGE_CUTOFF_DAYS]
        stats = pooled(young)
        rows.append({"distance": d, "n_all": len(trades_df), "n_young": stats["n"],
                      "mean_excess": stats["mean_excess"], "t": stats["t"]})
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    return out


def main():
    for which, trades_file in (("ticker_list", "naive_strategy_trades.csv"),
                                ("oos", "oos_strategy_trades.csv")):
        series = load_universe(which)
        print(f"\n{len(series)} tickers loaded for {which}")
        young = per_trade_causal_check(which, series, os.path.join(DATA_DIR, trades_file))
        concentration_report(which, young)
        distance_sensitivity(which, series, distances=[3, 5, 10, 15, 20])


if __name__ == "__main__":
    main()
