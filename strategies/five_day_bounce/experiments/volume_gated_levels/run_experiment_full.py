"""Full-scale version of run_experiment.py: config.ticker_list (40 names),
full 2015-2026 history, plus a placebo/random-entry test on the gated arm --
the small run's own printed caveat ("re-run on the full ticker_list/date
range before trusting it") is what this answers.

Still does not modify tech_levels.py, tech_level_naive_strategy.py, or
tech_level_continuation_live.py -- same read-only imports, same gated
construction from build_levels_volume_gated.py, shared with run_experiment.py
via experiment_lib.py so the two runs can't silently define "baseline" or
"gated" differently.

Batches the download (tech_level_live.pull_all, one yfinance call for all
tickers) rather than run_experiment.py's per-ticker pull_series -- serial
40x fetches is the mistake tech_level_live.py's own docstring already
flagged as having caused lost trading days before.

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_experiment_full.py
Results saved under this experiment's own data/ folder (full_*.csv), separate
from run_experiment.py's small-sample outputs so a scaled-up run doesn't
overwrite the quick-look results.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIVE_DAY_BOUNCE = os.path.join(_HERE, "..", "..")
_REPO_ROOT = os.path.join(_FIVE_DAY_BOUNCE, "..", "..")
sys.path.insert(0, os.path.abspath(_REPO_ROOT))
sys.path.insert(0, os.path.abspath(_FIVE_DAY_BOUNCE))
sys.path.insert(0, _HERE)

import config
from tech_level_live import pull_all
from tech_level_naive_strategy import load_fixed_combo, simulate, equal_weight_curve, attach_benchmark, summarise
from tech_level_continuation_live import MAX_AGE_DAYS
from experiment_lib import build_levels_baseline, build_levels_gated, simulate_age_gated, placebo_trades

HOLD_DAYS = 5
NEAR_PCT = 0.01
MIN_VOLUME_RATIO = 1.0
OUT_DIR = os.path.join(_HERE, "data")


def report_arm(name, trades_df, ew, out_csv):
    df = attach_benchmark(pd.DataFrame(trades_df), ew)
    df.to_csv(out_csv, index=False)
    print(f"\n{name} (n={len(df)}):")
    if df.empty:
        print("  no trades")
    else:
        print(summarise(df).to_string(index=False))
    return df


def lift_vs_placebo(label, real_df, close_series, ew, out_csv, hold_days=HOLD_DAYS):
    """Random-entry placebo matched to real_df's per-stock trade count/
    frequency, plus the two-sample t-test on excess_ret real-vs-placebo --
    factored out since both the volume-only and age+volume arms need it.
    """
    real_counts = real_df.groupby("stock").size()
    placebo_df = placebo_trades(close_series, real_counts, hold_days=hold_days)
    placebo_df = attach_benchmark(placebo_df, ew)
    placebo_df.to_csv(out_csv, index=False)

    print(f"\n=== placebo for {label} (random entries matched to its count/frequency per stock) ===")
    print(summarise(placebo_df).to_string(index=False))

    a, b = real_df.excess_ret.dropna(), placebo_df.excess_ret.dropna()
    real_excess, placebo_excess = a.mean(), b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t_lift = (real_excess - placebo_excess) / se if se > 0 else float("nan")

    print(f"real mean_excess ({label}): {real_excess:.4%}")
    print(f"placebo mean_excess: {placebo_excess:.4%}")
    print(f"lift: {real_excess - placebo_excess:+.4%}  (t={t_lift:.2f})")
    return placebo_df


def main():
    tickers = config.ticker_list
    print(f"pulling full history for {len(tickers)} tickers (batched)...")
    series, failed = pull_all(tickers)
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}")
    print(f"got {len(series)} tickers, building levels (full history, both arms)...")

    combo = load_fixed_combo()

    close_series = {}
    level_counts = []
    baseline_trades, gated_trades = [], []
    age_baseline_trades, age_gated_trades = [], []

    for t, (close, volume) in series.items():
        close = close.copy()
        close.index = pd.to_datetime(close.index)
        volume = volume.copy()
        volume.index = pd.to_datetime(volume.index)
        close_series[t] = close

        levels_baseline = build_levels_baseline(close, combo)
        levels_gated, n_gated_out = build_levels_gated(close, volume, combo, min_ratio=MIN_VOLUME_RATIO)

        level_counts.append({
            "ticker": t, "n_levels_baseline": len(levels_baseline), "n_levels_gated": len(levels_gated),
            "n_births_blocked_by_gate": n_gated_out,
        })

        baseline_trades.extend(simulate(t, close, levels_baseline, hold_days=HOLD_DAYS, near_pct=NEAR_PCT))
        gated_trades.extend(simulate(t, close, levels_gated, hold_days=HOLD_DAYS, near_pct=NEAR_PCT))

        # same two level sets, but entries additionally gated on the level
        # being younger than MAX_AGE_DAYS -- the rule actually live in
        # tech_level_continuation_live.py -- to see whether the volume gate
        # adds anything on top of "young levels" rather than the looser
        # any-age naive rule above.
        age_baseline_trades.extend(simulate_age_gated(t, close, levels_baseline, hold_days=HOLD_DAYS,
                                                        near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS))
        age_gated_trades.extend(simulate_age_gated(t, close, levels_gated, hold_days=HOLD_DAYS,
                                                     near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS))

    os.makedirs(OUT_DIR, exist_ok=True)
    counts_df = pd.DataFrame(level_counts)
    counts_df.to_csv(os.path.join(OUT_DIR, "full_level_counts.csv"), index=False)
    total_baseline_levels = counts_df.n_levels_baseline.sum()
    total_gated_levels = counts_df.n_levels_gated.sum()
    total_blocked = counts_df.n_births_blocked_by_gate.sum()
    print(f"\nlevels: {total_baseline_levels} baseline -> {total_gated_levels} gated "
          f"({total_blocked} births blocked by the volume gate, "
          f"{total_blocked / (total_baseline_levels + total_blocked):.1%} of all would-be births)")

    ew = equal_weight_curve(close_series)

    print(f"\n=== ARM SET 1: any-age (naive-strategy rule, mark_broken only), hold={HOLD_DAYS}d, "
          f"near_pct={NEAR_PCT:.0%} ===")
    base_df = report_arm("baseline", baseline_trades, ew, os.path.join(OUT_DIR, "full_baseline_trades.csv"))
    gate_df = report_arm("volume-gated", gated_trades, ew, os.path.join(OUT_DIR, "full_gated_trades.csv"))

    by_stock = summarise(gate_df, by="stock")
    n_positive = (by_stock.mean_excess > 0).sum()
    print(f"\nvolume-gated by stock: {n_positive}/{len(by_stock)} individually positive on excess")
    print(pd.concat([by_stock.head(5), by_stock.tail(5)]).to_string(index=False))

    lift_vs_placebo("volume-gated (any age)", gate_df, close_series, ew,
                     os.path.join(OUT_DIR, "full_gated_placebo_trades.csv"))

    print(f"\n\n=== ARM SET 2: young levels only (age < {MAX_AGE_DAYS}d, the rule actually live in "
          f"tech_level_continuation_live.py), hold={HOLD_DAYS}d, near_pct={NEAR_PCT:.0%} ===")
    age_base_df = report_arm("age-gated baseline (young levels, any volume)", age_baseline_trades, ew,
                              os.path.join(OUT_DIR, "full_age_baseline_trades.csv"))
    age_gate_df = report_arm("age-gated + volume-gated (young AND high-volume-birth levels)", age_gated_trades, ew,
                              os.path.join(OUT_DIR, "full_age_gated_trades.csv"))

    if not age_gate_df.empty:
        by_stock_age = summarise(age_gate_df, by="stock")
        n_positive_age = (by_stock_age.mean_excess > 0).sum()
        print(f"\nage+volume-gated by stock: {n_positive_age}/{len(by_stock_age)} individually positive on excess")
        print(pd.concat([by_stock_age.head(5), by_stock_age.tail(5)]).to_string(index=False))

        lift_vs_placebo("age+volume-gated", age_gate_df, close_series, ew,
                         os.path.join(OUT_DIR, "full_age_gated_placebo_trades.csv"))

        print(f"\n\n=== does the volume gate add anything ON TOP OF the age gate? ===")
        print(f"age-only (young levels, any volume):      n={len(age_base_df)}  "
              f"mean_excess={age_base_df.excess_ret.mean():.4%}" if not age_base_df.empty else
              "age-only: no trades")
        print(f"age + volume gate (young AND high volume): n={len(age_gate_df)}  "
              f"mean_excess={age_gate_df.excess_ret.mean():.4%}")
    else:
        print("\nage+volume-gated arm produced 0 trades -- the two filters together leave nothing "
              "in this universe/window; can't say anything about stacking them.")

    print(f"\nsaved full_level_counts.csv and full_{{baseline,gated,age_baseline,age_gated}}_trades.csv "
          f"(+ their placebo files) to {OUT_DIR}")


if __name__ == "__main__":
    main()
