"""Does filtering to bounce_1d<median make MORE total money, or does
throwing away half the trades cost more than the per-trade quality
improvement gains back?

Per-trade mean excess answers "is the average trade better" -- it doesn't
answer "am I more profitable overall," since that also depends on how many
trades you're giving up and whether capital would otherwise have sat idle
anyway (i.e. whether concurrency is actually a binding constraint for this
strategy, per tech_level_causal_check.py's own portfolio-simulation
convention). Reuses that exact machinery (slotted_trades/slotted_equity_curve)
read-only, run on the two-touch young-level population, unfiltered vs.
bounce_1d<median-filtered, at matched slot counts.

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_bounce_filter_profitability.py
"""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIVE_DAY_BOUNCE = os.path.join(_HERE, "..", "..")
_REPO_ROOT = os.path.join(_FIVE_DAY_BOUNCE, "..", "..")
sys.path.insert(0, os.path.abspath(_REPO_ROOT))
sys.path.insert(0, os.path.abspath(_FIVE_DAY_BOUNCE))
sys.path.insert(0, _HERE)

import config
from tech_level_naive_strategy import load_fixed_combo
from tech_level_continuation_live import MAX_AGE_DAYS
from tech_level_causal_check import slotted_trades, slotted_equity_curve
from experiment_lib import build_young_level_trades
from run_momentum_mood import HOLD_DAYS, NEAR_PCT
from run_oos_validation import load_oos_series
from run_post_birth_bounce import bounce_after_birth

SLOT_SCENARIOS = (5, 10, 24)


def compare(label, trades_df, close_series):
    trades_df = trades_df.dropna(subset=["bounce_1d"]).copy()
    threshold = trades_df.bounce_1d.median()
    filtered = trades_df[trades_df.bounce_1d < threshold]

    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"unfiltered: n={len(trades_df)}, mean_excess={trades_df.excess_ret.mean():.4%}")
    print(f"filtered:   n={len(filtered)}, mean_excess={filtered.excess_ret.mean():.4%}  "
          f"({len(trades_df) - len(filtered)} trades thrown out, "
          f"{(len(trades_df) - len(filtered)) / len(trades_df):.0%} of the total)")

    print(f"\n{'n_slots':>8}  {'unfiltered accepted':>20}  {'unfiltered CAGR':>16}  {'unfiltered Sharpe':>18}  "
          f"{'filtered accepted':>18}  {'filtered CAGR':>14}  {'filtered Sharpe':>16}")
    for n_slots in SLOT_SCENARIOS:
        acc_u = slotted_trades(trades_df, n_slots)
        _eq_u, stats_u = slotted_equity_curve(acc_u, close_series, n_slots)
        acc_f = slotted_trades(filtered, n_slots)
        _eq_f, stats_f = slotted_equity_curve(acc_f, close_series, n_slots)
        if not stats_u or not stats_f:
            continue
        print(f"{n_slots:>8}  {stats_u['n_trades_accepted']:>10}/{len(trades_df):<8}  "
              f"{stats_u['cagr']:>+15.2%}  {stats_u['sharpe']:>18.2f}  "
              f"{stats_f['n_trades_accepted']:>9}/{len(filtered):<7}  "
              f"{stats_f['cagr']:>+13.2%}  {stats_f['sharpe']:>16.2f}")


def main():
    combo = load_fixed_combo()

    print("building two-touch population: ticker_list...")
    tl_trades, tl_close, _v, _ew, failed = build_young_level_trades(
        config.ticker_list, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS
    )
    tl_trades["bounce_1d"] = [
        bounce_after_birth(tl_close[r.stock], r.support_birth_date, r.entry_date, 1) for r in tl_trades.itertuples()
    ]
    compare("ticker_list", tl_trades, tl_close)

    print("\nbuilding two-touch population: oos (-NVDA)...")
    oos_tickers = [t for t in config.oos_ticker_list if t != "NVDA"]
    oos_series = load_oos_series(oos_tickers)
    oos_trades, oos_close, _v, _ew, failed = build_young_level_trades(
        oos_tickers, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS, series=oos_series
    )
    oos_trades["bounce_1d"] = [
        bounce_after_birth(oos_close[r.stock], r.support_birth_date, r.entry_date, 1) for r in oos_trades.itertuples()
    ]
    compare("oos", oos_trades, oos_close)


if __name__ == "__main__":
    main()
