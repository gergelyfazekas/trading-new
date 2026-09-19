"""Is rel_mom_5d (run_momentum_mood.py's strongest, cleanest finding) doing
something specific to the young-support-level setup, or is it just generic
short-term cross-sectional reversal that would show up on ANY entry date --
in which case it's not telling us anything about *this* strategy, just
re-discovering an unrelated, already well-known market microstructure
effect and layering it on top?

Test: compute the identical rel_mom_5d feature (5-day stock return minus
5-day equal-weight-book return, ending the day before entry) for a matched
random-entry placebo (same tickers, same count/frequency per ticker as the
real young-level trades), then correlate it with the placebo trades'
excess_ret the same way. If the real population's correlation is
meaningfully stronger than the placebo's, the effect is specific to buying
near a fresh support level (an "oversold-into-support" dynamic). If they're
similar, it's just generic reversal and not specific to this setup.

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_momentum_specificity_check.py
"""
import os
import sys

import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIVE_DAY_BOUNCE = os.path.join(_HERE, "..", "..")
_REPO_ROOT = os.path.join(_FIVE_DAY_BOUNCE, "..", "..")
sys.path.insert(0, os.path.abspath(_REPO_ROOT))
sys.path.insert(0, os.path.abspath(_FIVE_DAY_BOUNCE))
sys.path.insert(0, _HERE)

import config
from tech_level_naive_strategy import load_fixed_combo, attach_benchmark
from tech_level_continuation_live import MAX_AGE_DAYS
from experiment_lib import build_young_level_trades, placebo_trades
from run_momentum_mood import trailing_return, HOLD_DAYS, NEAR_PCT

OUT_DIR = os.path.join(_HERE, "data")


def main():
    tickers = config.ticker_list
    combo = load_fixed_combo()
    print("rebuilding the young-level trade population...")
    real_df, close_series, volume_series, ew, failed = build_young_level_trades(
        tickers, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS
    )

    print("generating matched placebo (same count/frequency per ticker)...")
    placebo_df = placebo_trades(close_series, real_df.groupby("stock").size(), hold_days=HOLD_DAYS)
    placebo_df = attach_benchmark(placebo_df, ew)

    for df in (real_df, placebo_df):
        df["stock_mom_5d"] = [trailing_return(close_series[row.stock], row.entry_date, 5) for row in df.itertuples()]
        df["market_mom_5d"] = [trailing_return(ew, row.entry_date, 5) for row in df.itertuples()]
        df["rel_mom_5d"] = df["stock_mom_5d"] - df["market_mom_5d"]

    print(f"\n=== rel_mom_5d vs excess_ret: real young-level trades vs. matched placebo ===")
    for label, df in (("real (young-level)", real_df), ("placebo (random entry)", placebo_df)):
        sub = df.dropna(subset=["rel_mom_5d"])
        rho, p = stats.spearmanr(sub.rel_mom_5d, sub.excess_ret)
        print(f"{label:24s}: n={len(sub):4d}  spearman_rho={rho:+.3f}  p={p:.4f}")

    real_df.to_csv(os.path.join(OUT_DIR, "specificity_real_trades.csv"), index=False)
    placebo_df.to_csv(os.path.join(OUT_DIR, "specificity_placebo_trades.csv"), index=False)
    print(f"\nsaved specificity_real_trades.csv / specificity_placebo_trades.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
