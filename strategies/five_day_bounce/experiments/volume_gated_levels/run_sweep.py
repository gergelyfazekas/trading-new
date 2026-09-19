"""Sweep the volume-ratio birth gate's threshold across a grid, instead of
just the single min_ratio=1.0 point tested in run_experiment_full.py.

Why this matters before trusting that one point: tech_levels_notes.md's
single most-repeated finding in this whole project is that a result which
climbs *monotonically* as some threshold is loosened/tightened (tech_width
being the classic case) is usually an artifact of the threshold itself, not
a real effect at any one setting -- the "width-shape check". This sweep is
that same check applied to the new volume-ratio threshold: if excess climbs
steadily as the threshold rises (rarer, more extreme high-volume births)
with no interior optimum, that shape alone would be reason to distrust
min_ratio=1.0 as a "finding" and treat it as movement along a curve instead.

Threshold 0.0 is "no gate at all" -- included as the reference point (it
should reproduce run_experiment_full.py's ungated arms exactly).

Reuses the exact same level construction, simulate_age_gated, and placebo
machinery as run_experiment_full.py via experiment_lib.py -- still nothing
touched in tech_levels.py / tech_level_naive_strategy.py /
tech_level_continuation_live.py.

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_sweep.py
Saves sweep_results.csv and sweep_chart.png to this experiment's data/ folder.
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
from tech_level_live import pull_all
from tech_level_naive_strategy import load_fixed_combo, simulate, equal_weight_curve, attach_benchmark
from tech_level_continuation_live import MAX_AGE_DAYS
from experiment_lib import build_levels_baseline, build_levels_gated, simulate_age_gated, placebo_trades

HOLD_DAYS = 5
NEAR_PCT = 0.01
THRESHOLDS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
OUT_DIR = os.path.join(_HERE, "data")


def main():
    tickers = config.ticker_list
    print(f"pulling full history for {len(tickers)} tickers (batched)...")
    series, failed = pull_all(tickers)
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}")
    print(f"got {len(series)} tickers")

    combo = load_fixed_combo()

    close_series, volume_series = {}, {}
    for t, (close, volume) in series.items():
        close = close.copy()
        close.index = pd.to_datetime(close.index)
        volume = volume.copy()
        volume.index = pd.to_datetime(volume.index)
        close_series[t] = close
        volume_series[t] = volume

    ew = equal_weight_curve(close_series)

    rows = []
    for min_ratio in THRESHOLDS:
        print(f"\nthreshold = {min_ratio}...")
        any_age_trades, young_trades = [], []
        n_levels_total, n_blocked_total = 0, 0

        for t, close in close_series.items():
            volume = volume_series[t]
            if min_ratio == 0.0:
                levels = build_levels_baseline(close, combo)
                n_gated_out = 0
            else:
                levels, n_gated_out = build_levels_gated(close, volume, combo, min_ratio=min_ratio)
            n_levels_total += len(levels)
            n_blocked_total += n_gated_out

            any_age_trades.extend(simulate(t, close, levels, hold_days=HOLD_DAYS, near_pct=NEAR_PCT))
            young_trades.extend(simulate_age_gated(t, close, levels, hold_days=HOLD_DAYS,
                                                      near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS))

        any_age_df = attach_benchmark(pd.DataFrame(any_age_trades), ew)
        young_df = attach_benchmark(pd.DataFrame(young_trades), ew)

        any_age_counts = any_age_df.groupby("stock").size() if not any_age_df.empty else pd.Series(dtype=int)
        young_counts = young_df.groupby("stock").size() if not young_df.empty else pd.Series(dtype=int)
        any_age_placebo = placebo_trades(close_series, any_age_counts, hold_days=HOLD_DAYS)
        young_placebo = placebo_trades(close_series, young_counts, hold_days=HOLD_DAYS)
        any_age_placebo = attach_benchmark(any_age_placebo, ew) if not any_age_placebo.empty else any_age_placebo
        young_placebo = attach_benchmark(young_placebo, ew) if not young_placebo.empty else young_placebo

        row = {
            "min_ratio": min_ratio,
            "n_levels": n_levels_total,
            "n_births_blocked": n_blocked_total,
            "any_age_n": len(any_age_df),
            "any_age_mean_excess": any_age_df.excess_ret.mean() if not any_age_df.empty else float("nan"),
            "any_age_t_stat": (any_age_df.excess_ret.mean() / any_age_df.excess_ret.std()
                                 * np.sqrt(len(any_age_df)) if len(any_age_df) > 2 else float("nan")),
            "any_age_placebo_excess": any_age_placebo.excess_ret.mean() if not any_age_placebo.empty else float("nan"),
            "young_n": len(young_df),
            "young_mean_excess": young_df.excess_ret.mean() if not young_df.empty else float("nan"),
            "young_t_stat": (young_df.excess_ret.mean() / young_df.excess_ret.std()
                               * np.sqrt(len(young_df)) if len(young_df) > 2 else float("nan")),
            "young_placebo_excess": young_placebo.excess_ret.mean() if not young_placebo.empty else float("nan"),
            "young_n_stocks_positive": (
                young_df.groupby("stock").excess_ret.mean().gt(0).sum() if not young_df.empty else 0
            ),
            "young_n_stocks_total": young_df["stock"].nunique() if not young_df.empty else 0,
        }
        row["any_age_lift"] = row["any_age_mean_excess"] - row["any_age_placebo_excess"]
        row["young_lift"] = row["young_mean_excess"] - row["young_placebo_excess"]
        rows.append(row)

        print(f"  levels={n_levels_total} (blocked={n_blocked_total}) | "
              f"any-age: n={row['any_age_n']}, excess={row['any_age_mean_excess']:.3%}, "
              f"lift={row['any_age_lift']:+.3%} | "
              f"young(<{MAX_AGE_DAYS}d): n={row['young_n']}, excess={row['young_mean_excess']:.3%}, "
              f"lift={row['young_lift']:+.3%}, "
              f"{row['young_n_stocks_positive']}/{row['young_n_stocks_total']} stocks positive")

    os.makedirs(OUT_DIR, exist_ok=True)
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(os.path.join(OUT_DIR, "sweep_results.csv"), index=False)
    print(f"\nsaved sweep_results.csv to {OUT_DIR}")
    print(sweep_df.to_string(index=False))

    plot_sweep(sweep_df, os.path.join(OUT_DIR, "sweep_chart.png"))
    print(f"saved sweep_chart.png to {OUT_DIR}")


def plot_sweep(df, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df.min_ratio, df.young_mean_excess * 100, marker="o", color="#1f5fa8",
              label=f"young levels (<{MAX_AGE_DAYS}d), real")
    ax1.plot(df.min_ratio, df.young_placebo_excess * 100, marker="o", linestyle="--", color="#1f5fa8", alpha=0.5,
              label="young levels, placebo")
    ax1.plot(df.min_ratio, df.any_age_mean_excess * 100, marker="s", color="#d62728",
              label="any age, real")
    ax1.plot(df.min_ratio, df.any_age_placebo_excess * 100, marker="s", linestyle="--", color="#d62728", alpha=0.5,
              label="any age, placebo")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("mean excess return (%)")
    ax1.set_title("Volume-ratio birth-gate threshold sweep -- real vs. placebo excess")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.plot(df.min_ratio, df.young_n, marker="o", color="#1f5fa8", label="young-levels trades")
    ax2.plot(df.min_ratio, df.any_age_n, marker="s", color="#d62728", label="any-age trades")
    ax2.set_xlabel("min_ratio (birth-day volume / trailing 10d avg)")
    ax2.set_ylabel("n trades")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
