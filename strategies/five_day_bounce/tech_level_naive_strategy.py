"""Literal test of the hand-trading idea: buy near a support level, sell after
N days or when price reaches the nearest known resistance, whichever is first.

Distinct from tech_level_trades.py's trade_events, which only opens a trade once
price actually touches a band and exits on bounce-confirm/breakout. This module
tests the plainer rule someone would actually hand-trade:

  - entry: price is within `near_pct` of (above) a support band that was already
    born as of that date -- causal, no lookahead -- and no position is currently
    open in that name. `near_pct=0` means "must actually touch the band", which
    reduces to roughly the trade_events entry condition; the default (1%) is
    looser, matching "close to" a level rather than "at" it. A level stops
    counting as support/resistance once price has fully crossed its band
    (tech_levels.mark_broken) -- a band price has already punched through
    isn't live structure for a fresh decision, and without this a stock's
    entire multi-year band history piles up as "current" levels regardless of
    how many of them price has since blown through (visibly so once plotted).
  - exit: the earlier of (a) N trading days later, or (b) the first day price
    enters the nearest known resistance band that existed above the entry price
    at entry time. A trade that hits neither by the end of the series is
    dropped, not truncated, to avoid biasing the sample toward whatever the last
    few days happened to do (same convention as trade_events).

Uses the fixed a-priori combo from data/fixed_combo.json (distance=10,
prominence=0.01, tech_width=0.008) -- chosen without looking at any stock's
outcome, so results here carry no per-stock combo-selection bias. That combo was
itself picked from universe-wide calibration, not from this test, so it is not
selected on this outcome either.

Benchmarked against the equal-weight universe (same convention as
tech_level_trades.py, at the repo root) and against random-entry placebo draws
matched on trade count, per-stock frequency, and holding-length distribution --
megacaps beating an equal-weight book from *any* entry point has overturned
"wins" here before (see tech_levels_notes.md, GOOG/MSFT/CSCO).

Part of the strategies/five_day_bounce package -- see NOTES.md in this folder
for how this fits with the other scripts here.
"""
import json
import datetime
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import config
from stock_class import StockList
from tech_levels import find_touches, build_levels_causal, mark_broken

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
COMBO_FILE = os.path.join(DATA_DIR, "fixed_combo.json")
START_DATE = datetime.date(2015, 5, 28)


def load_fixed_combo(path=COMBO_FILE):
    with open(path) as f:
        return json.load(f)


def build_levels(close, combo):
    """Build causal levels and mark each broken from the date price fully
    crosses its band onward (see tech_levels.mark_broken) -- a level that has
    already been punched through top-to-bottom or bottom-to-top isn't live
    support/resistance for a fresh decision, even though it's still kept
    around (and still counted in tech_level_search's hit-rate scoring, which
    wants exactly those post-break touches). simulate() below is what
    actually excludes broken levels from being traded.
    """
    combo = dict(combo)
    tech_width = combo.pop("tech_width")
    consider_volume = combo.pop("consider_volume", False)
    volume_height = combo.pop("volume_height", 0.0)
    volume_prominence = combo.pop("volume_prominence", 0.0)
    touches = find_touches(close, consider_volume=consider_volume,
                            volume_height=volume_height, volume_prominence=volume_prominence,
                            **combo)
    levels = build_levels_causal(touches, tech_width)
    mark_broken(levels, close)
    return levels


def active_support_resistance(levels, date, price):
    """Closest already-born, not-yet-broken level below/above price on `date`.

    Shared by simulate() (backtest) and tech_level_continuation_live.py (live
    signals) so the two can't silently drift into different rules.
    """
    born = [lvl for lvl in levels
            if pd.Timestamp(lvl.birth_date) <= date
            and (lvl.broken_at is None or date < pd.Timestamp(lvl.broken_at))]
    below = [lvl for lvl in born if lvl.band[1] < price]
    above = [lvl for lvl in born if lvl.band[0] > price]
    support = max(below, key=lambda lvl: lvl.band[1]) if below else None
    resistance = min(above, key=lambda lvl: lvl.band[0]) if above else None
    return support, resistance


def simulate(ticker, close, levels, hold_days, near_pct=0.01, cost_bps=10.0,
             random_entries=False, rng=None):
    """One (ticker, hold_days) pass. Returns a list of trade dicts.

    random_entries=True replaces the support-proximity entry rule with uniformly
    random entry dates (same count target not enforced here -- the caller resamples
    to match count/frequency), used for the placebo control.
    """
    idx = close.index
    n = len(idx)
    levels_sorted = sorted(levels, key=lambda lvl: lvl.birth_date)

    trades = []
    in_position_until = None  # date index of current exit, one position per ticker at a time

    for i, date in enumerate(idx):
        if in_position_until is not None and i <= in_position_until:
            continue
        in_position_until = None

        price = close.iloc[i]
        support, resistance = active_support_resistance(levels_sorted, date, price)

        if random_entries:
            is_entry = rng.random() < 0.05  # loose density; caller resamples to match count
        else:
            if support is None:
                is_entry = False
            else:
                dist = (price - support.band[1]) / price
                is_entry = 0 <= dist <= near_pct

        if not is_entry:
            continue

        entry_date, entry_price = date, price
        exit_pos = min(i + hold_days, n - 1)
        exit_reason = "hold_days"

        for j in range(i + 1, min(i + hold_days, n - 1) + 1):
            if resistance is not None and resistance.band[0] <= close.iloc[j] <= resistance.band[1]:
                exit_pos = j
                exit_reason = "resistance"
                break

        if i + hold_days >= n and exit_reason == "hold_days":
            continue  # would need to truncate -- drop, same convention as trade_events

        exit_date, exit_price = idx[exit_pos], close.iloc[exit_pos]
        support_age_days = (pd.Timestamp(date) - pd.Timestamp(support.birth_date)).days if support is not None else None
        trades.append({
            "stock": ticker, "hold_days": hold_days, "near_pct": near_pct,
            "entry_date": entry_date, "exit_date": exit_date,
            "entry_price": float(entry_price), "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "ret": float(exit_price / entry_price - 1.0),
            "ret_net": float(exit_price / entry_price - 1.0) - cost_bps / 1e4,
            "support_age_days": support_age_days,
        })
        in_position_until = exit_pos

    return trades


def equal_weight_curve(series_by_ticker):
    rets = {t: c.pct_change() for t, c in series_by_ticker.items()}
    R = pd.DataFrame(rets).sort_index()
    return (1 + R.mean(axis=1).fillna(0)).cumprod()


def attach_benchmark(trades_df, ew):
    bench = ew.reindex(pd.DatetimeIndex(trades_df.exit_date)).to_numpy() / \
            ew.reindex(pd.DatetimeIndex(trades_df.entry_date)).to_numpy() - 1.0
    trades_df["bench_ret"] = bench
    trades_df["excess_ret"] = trades_df["ret_net"] - trades_df["bench_ret"]
    return trades_df


def summarise(trades_df, by=None):
    def _row(g, key=None):
        n = len(g)
        row = {
            "n": n,
            "hit_rate_raw": (g.ret > 0).mean(),
            "mean_ret": g.ret.mean(),
            "mean_ret_net": g.ret_net.mean(),
            "mean_excess": g.excess_ret.mean(),
            "t_stat_excess": g.excess_ret.mean() / g.excess_ret.std() * np.sqrt(n) if n > 2 else np.nan,
            "pct_exit_resistance": (g.exit_reason == "resistance").mean(),
            "median_holding_days": (pd.DatetimeIndex(g.exit_date) - pd.DatetimeIndex(g.entry_date)).days.to_numpy().mean() if n else np.nan,
        }
        if key is not None:
            row[by] = key
        return row

    if by is None:
        return pd.DataFrame([_row(trades_df)])
    return pd.DataFrame([_row(g, k) for k, g in trades_df.groupby(by)]).sort_values("mean_excess", ascending=False)


def run_backtest(series, combo, output_path, label=""):
    """Shared engine: build levels, sweep hold_days, benchmark, placebo-test.

    series: {ticker: close_series}. combo: the (fixed or tuned) param dict to
    build every ticker's levels with -- same combo for the whole universe,
    since mixing per-ticker tuned combos back in here would reintroduce the
    selection bias the fixed arm exists to avoid. label: appended to print
    headers only (e.g. " (OOS universe)"), no effect on the computation.

    Returns (trades_df, placebo_df) and writes trades_df to output_path.
    """
    ew = equal_weight_curve(series)

    levels_by_ticker = {t: build_levels(c, combo) for t, c in series.items()}
    print("levels built; running strategy sweep...")

    all_trades = []
    for hold_days in (1, 2, 3, 4, 5):
        for t, c in series.items():
            trades = simulate(t, c, levels_by_ticker[t], hold_days=hold_days, near_pct=0.01)
            all_trades.extend(trades)

    trades_df = pd.DataFrame(all_trades)
    trades_df = attach_benchmark(trades_df, ew)
    trades_df.to_csv(output_path, index=False)

    print(f"\n=== by hold_days (near_pct=1%){label} ===")
    print(summarise(trades_df, by="hold_days").to_string(index=False))

    sub = trades_df[trades_df.hold_days == 5]
    print(f"\n=== by stock, hold_days=5{label} ===")
    print(summarise(sub, by="stock").to_string(index=False))

    print(f"\n=== pooled, hold_days=5{label} ===")
    print(summarise(sub).to_string(index=False))

    # placebo: random entries, same tickers, same hold_days set, resampled to
    # match each ticker's real trade count so frequency/name-mix isn't a confound
    rng = np.random.default_rng(0)
    real_counts = trades_df[trades_df.hold_days == 5].groupby("stock").size()
    placebo_trades = []
    for t, c in series.items():
        target_n = int(real_counts.get(t, 0))
        if target_n == 0:
            continue
        idx = c.index
        n = len(idx)
        candidate_starts = rng.choice(np.arange(0, n - 6), size=min(target_n * 20, n - 6), replace=False)
        picked = 0
        for i in sorted(candidate_starts):
            if picked >= target_n:
                break
            exit_pos = min(i + 5, n - 1)
            if i + 5 >= n:
                continue
            placebo_trades.append({
                "stock": t, "hold_days": 5,
                "entry_date": idx[i], "exit_date": idx[exit_pos],
                "entry_price": float(c.iloc[i]), "exit_price": float(c.iloc[exit_pos]),
                "exit_reason": "hold_days",
                "ret": float(c.iloc[exit_pos] / c.iloc[i] - 1.0),
                "ret_net": float(c.iloc[exit_pos] / c.iloc[i] - 1.0) - 10.0 / 1e4,
            })
            picked += 1

    placebo_df = pd.DataFrame(placebo_trades)
    placebo_df = attach_benchmark(placebo_df, ew)
    print(f"\n=== placebo (random entries, matched count per stock, hold_days=5){label} ===")
    print(summarise(placebo_df).to_string(index=False))

    print(f"\nreal mean_excess (hold=5){label}: {sub.excess_ret.mean():.4%}")
    print(f"placebo mean_excess (hold=5){label}: {placebo_df.excess_ret.mean():.4%}")
    print(f"lift{label}: {sub.excess_ret.mean() - placebo_df.excess_ret.mean():.4%}")

    return trades_df, placebo_df


def main():
    combo = load_fixed_combo()
    tickers = config.ticker_list

    print(f"pulling {len(tickers)} tickers...")
    sl = StockList(tickers)
    sl.pull_data(start=START_DATE, end=datetime.date.today())

    series = {}
    for t in tickers:
        try:
            c = sl[t].data["close"].dropna()
            c.index = pd.to_datetime(c.index)
            if not c.empty:
                series[t] = c
        except Exception:
            continue
    print(f"got price series for {len(series)} tickers")

    run_backtest(series, combo, os.path.join(DATA_DIR, "naive_strategy_trades.csv"))


if __name__ == "__main__":
    main()
