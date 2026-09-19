"""Position-sizing test, not a gate: weight each young-level (<5d) buy
signal by market_depth_10d instead of admitting/rejecting it outright.

Motivated directly by what run_stacking_test_market_depth.py and
run_oos_validation.py actually showed: the "excluded" complement group
(market_depth_10d >= 0) was never unprofitable -- it stayed solidly
positive (0.94% ticker_list, 0.93% OOS, both t>8) -- just weaker than the
filtered group. A hard gate throws away real, profitable trades just
because they're less good. Sizing keeps every signal, invests less into
the ones with a worse market backdrop, more into the ones with a better one.

Weight function -- an a-priori, interpretable choice, NOT fit to this
data's outcomes (only to the feature's own observed range, to avoid
reopening the same selection-bias problem a fitted weight curve would):

    weight = 1.0                                  if market_depth_10d <= 0
    weight = 1.0 - (depth / CAP) * (1 - FLOOR)     if 0 < depth < CAP
    weight = FLOOR                                 if depth >= CAP

CAP=5%, FLOOR=0.2 -- roughly spans the bucket range seen in
run_dip_depth.py's chart (Q1 ~0%, Q5 ~5-6%) and never fully zeroes a
signal out, consistent with the complement group still being profitable.
This exact CAP/FLOOR pair is a choice, not a swept/optimized parameter --
flagged explicitly in the output, not hidden.

Two checks, run on BOTH config.ticker_list (fresh pull) and
config.oos_ticker_list (frozen pull, NVDA excluded -- same convention as
run_oos_validation.py), so this isn't just re-confirming the universe the
idea came from:
  1. Weighted mean excess return (capital-weighted) vs. plain (equal-size)
     mean excess, plus a permutation test: shuffle the weight-to-trade
     assignment N times and see where the real weighted-vs-unweighted gap
     falls in that null distribution.
  2. The same 4-period stability split used throughout this thread, checked
     on the weighted-vs-unweighted delta rather than a filtered-vs-unfiltered
     one.

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_position_sizing.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
from experiment_lib import build_young_level_trades
from run_momentum_mood import HOLD_DAYS, NEAR_PCT
from run_dip_depth import trailing_max_depth
from run_oos_validation import load_oos_series

FEATURE = "market_depth_10d"
DEPTH_WINDOW = 10
CAP = 0.05
FLOOR = 0.2
N_PERIODS = 4
N_PERMUTATIONS = 5000
OUT_DIR = os.path.join(_HERE, "data")


def compute_weight(depth):
    if pd.isna(depth):
        return np.nan
    frac = np.clip(depth / CAP, 0.0, 1.0)
    return 1.0 - frac * (1.0 - FLOOR)


def permutation_test(excess, weight, n=N_PERMUTATIONS, seed=0):
    rng = np.random.default_rng(seed)
    unweighted_mean = excess.mean()
    real_diff = np.average(excess, weights=weight) - unweighted_mean
    null_diffs = np.empty(n)
    w = weight.to_numpy()
    x = excess.to_numpy()
    for i in range(n):
        shuffled = rng.permutation(w)
        null_diffs[i] = np.average(x, weights=shuffled) - unweighted_mean
    if real_diff >= 0:
        p = (np.sum(null_diffs >= real_diff) + 1) / (n + 1)
    else:
        p = (np.sum(null_diffs <= real_diff) + 1) / (n + 1)
    return real_diff, p


def analyze_universe(label, tickers, combo, series=None):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    trades_df, close_series, volume_series, ew, failed = build_young_level_trades(
        tickers, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS, series=series
    )
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}")

    trades_df[FEATURE] = [trailing_max_depth(ew, r.entry_date, DEPTH_WINDOW) for r in trades_df.itertuples()]
    trades_df = trades_df.dropna(subset=[FEATURE])
    trades_df["weight"] = trades_df[FEATURE].apply(compute_weight)
    print(f"{len(trades_df)} young-level trades, avg weight={trades_df.weight.mean():.3f} "
          f"(1.0=full size; {(trades_df.weight < 1.0).mean():.1%} of trades sized down below full)")

    unweighted_mean = trades_df.excess_ret.mean()
    weighted_mean = np.average(trades_df.excess_ret, weights=trades_df.weight)
    real_diff, p_perm = permutation_test(trades_df.excess_ret, trades_df.weight)

    print(f"\nequal-size (unweighted) mean excess: {unweighted_mean:.4%}")
    print(f"capital-weighted mean excess:         {weighted_mean:.4%}")
    print(f"gain from sizing:                     {real_diff:+.4%}  (permutation p={p_perm:.4f}, "
          f"n={N_PERMUTATIONS} shuffles)")

    trades_df["period"] = pd.qcut(trades_df.entry_date, N_PERIODS, duplicates="drop")
    rows = []
    for period, g in trades_df.groupby("period", observed=True):
        u_mean = g.excess_ret.mean()
        w_mean = np.average(g.excess_ret, weights=g.weight)
        rows.append({
            "period": str(period), "n": len(g), "avg_weight": g.weight.mean(),
            "unweighted_excess": u_mean, "weighted_excess": w_mean, "delta": w_mean - u_mean,
        })
    period_df = pd.DataFrame(rows)
    print(f"\n=== {N_PERIODS}-period stability of the sizing gain ===")
    print(period_df.to_string(index=False))
    n_positive = (period_df.delta > 0).sum()
    print(f"{n_positive}/{len(period_df)} periods show a positive sizing gain "
          f"(worst-case delta: {period_df.delta.min():+.4%})")

    return trades_df, period_df, {
        "label": label, "n": len(trades_df), "avg_weight": trades_df.weight.mean(),
        "unweighted_mean": unweighted_mean, "weighted_mean": weighted_mean,
        "gain": real_diff, "p_perm": p_perm, "n_positive_periods": n_positive,
    }


def main():
    combo = load_fixed_combo()
    os.makedirs(OUT_DIR, exist_ok=True)

    tl_trades, tl_periods, tl_summary = analyze_universe("config.ticker_list (fresh pull)", config.ticker_list, combo)
    tl_trades.to_csv(os.path.join(OUT_DIR, "sizing_ticker_list_trades.csv"), index=False)
    tl_periods.to_csv(os.path.join(OUT_DIR, "sizing_ticker_list_periods.csv"), index=False)

    oos_tickers = [t for t in config.oos_ticker_list if t != "NVDA"]
    oos_series = load_oos_series(oos_tickers)
    oos_trades, oos_periods, oos_summary = analyze_universe(
        "config.oos_ticker_list minus NVDA (frozen pull)", oos_tickers, combo, series=oos_series
    )
    oos_trades.to_csv(os.path.join(OUT_DIR, "sizing_oos_trades.csv"), index=False)
    oos_periods.to_csv(os.path.join(OUT_DIR, "sizing_oos_periods.csv"), index=False)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    summary_df = pd.DataFrame([tl_summary, oos_summary])
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUT_DIR, "sizing_summary.csv"), index=False)

    plot_periods(tl_periods, oos_periods, os.path.join(OUT_DIR, "sizing_chart.png"))
    print(f"\nsaved sizing_{{ticker_list,oos}}_{{trades,periods}}.csv, sizing_summary.csv, "
          f"sizing_chart.png to {OUT_DIR}")


def plot_periods(tl_periods, oos_periods, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, df, title in zip(axes, (tl_periods, oos_periods), ("ticker_list", "oos_ticker_list (- NVDA)")):
        x = np.arange(len(df))
        w = 0.35
        ax.bar(x - w / 2, df.unweighted_excess * 100, width=w, color="#9ecae1", label="equal-size")
        ax.bar(x + w / 2, df.weighted_excess * 100, width=w, color="#1f5fa8", label="capital-weighted")
        for xi, row in zip(x, df.itertuples()):
            ax.annotate(f"n={row.n}", (xi, max(row.unweighted_excess, row.weighted_excess) * 100),
                         textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"period {i+1}" for i in range(len(df))])
        ax.set_ylabel("mean excess return (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"Capital-weighted vs. equal-size sizing by {FEATURE} (cap={CAP:.0%}, floor={FLOOR})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
