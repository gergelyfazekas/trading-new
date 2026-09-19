"""Stacking test: does filtering the already-validated young-level (<5d)
rule on market_depth_10d < 0 (the equal-weight book is AT OR ABOVE its own
10-trading-day high, i.e. no market-wide pullback at all -- run_dip_depth.py's
cleanest, strongest finding: a clean monotonic staircase, placebo-null,
stronger than every other feature tested in this whole investigation)
improve it, and does that improvement survive the same multi-cutoff
stability discipline used for DHR/MMM and for the rel_mom_5d stacking test?

Threshold choice: market_depth_10d < 0 is the natural a-priori zero cutoff
("is the market currently making/at a new 10-day high, or not") -- not
fitted to this data, same convention as the volume ratio's >=1.0 and
rel_mom_5d's <0 cutoffs elsewhere in this package.

Same two checks as run_stacking_test.py:
  1. Pooled: filtered arm vs. unfiltered young-level arm vs. a matched
     random-entry placebo for the filtered arm.
  2. Stability: split all young-level trades into 4 independent,
     roughly-equal-count periods by entry date and require the filtered
     arm to beat the unfiltered arm's excess in the WORST period.

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_stacking_test_market_depth.py
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
from tech_level_naive_strategy import load_fixed_combo, attach_benchmark, summarise
from tech_level_continuation_live import MAX_AGE_DAYS
from experiment_lib import build_young_level_trades, placebo_trades
from run_momentum_mood import HOLD_DAYS, NEAR_PCT
from run_dip_depth import trailing_max_depth

FEATURE = "market_depth_10d"
DEPTH_WINDOW = 10
THRESHOLD = 0.0  # a-priori: "is the market at/above its own 10-day high"
N_PERIODS = 4
OUT_DIR = os.path.join(_HERE, "data")


def lift_vs_placebo(label, real_df, close_series, ew):
    real_counts = real_df.groupby("stock").size()
    placebo_df = placebo_trades(close_series, real_counts, hold_days=HOLD_DAYS)
    placebo_df = attach_benchmark(placebo_df, ew)
    a, b = real_df.excess_ret.dropna(), placebo_df.excess_ret.dropna()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 2 and len(b) > 2 else float("nan")
    t_lift = (a.mean() - b.mean()) / se if se and se > 0 else float("nan")
    print(f"{label}: real={a.mean():.4%}  placebo={b.mean():.4%}  lift={a.mean() - b.mean():+.4%}  t={t_lift:.2f}")
    return placebo_df


def main():
    tickers = config.ticker_list
    combo = load_fixed_combo()
    print("building the young-level trade population...")
    trades_df, close_series, volume_series, ew, failed = build_young_level_trades(
        tickers, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS
    )
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}")

    trades_df[FEATURE] = [trailing_max_depth(ew, r.entry_date, DEPTH_WINDOW) for r in trades_df.itertuples()]
    trades_df = trades_df.dropna(subset=[FEATURE])
    print(f"{len(trades_df)} young-level trades with a usable {FEATURE}")

    filtered = trades_df[trades_df[FEATURE] < THRESHOLD].copy()
    complement = trades_df[trades_df[FEATURE] >= THRESHOLD].copy()

    os.makedirs(OUT_DIR, exist_ok=True)
    trades_df.to_csv(os.path.join(OUT_DIR, "stacking_market_depth_all_trades.csv"), index=False)

    print(f"\n=== POOLED: young-level rule, with vs. without {FEATURE}<{THRESHOLD} filter ===")
    print(f"unfiltered (n={len(trades_df)}): mean_excess={trades_df.excess_ret.mean():.4%}, "
          f"t={trades_df.excess_ret.mean() / trades_df.excess_ret.std() * np.sqrt(len(trades_df)):.2f}")
    print(f"filtered   (n={len(filtered)}): mean_excess={filtered.excess_ret.mean():.4%}, "
          f"t={filtered.excess_ret.mean() / filtered.excess_ret.std() * np.sqrt(len(filtered)):.2f}")
    print(f"complement (n={len(complement)}): mean_excess={complement.excess_ret.mean():.4%}, "
          f"t={complement.excess_ret.mean() / complement.excess_ret.std() * np.sqrt(len(complement)):.2f}")

    by_stock = summarise(filtered, by="stock")
    n_positive = (by_stock.mean_excess > 0).sum()
    print(f"\nfiltered arm by stock: {n_positive}/{len(by_stock)} individually positive on excess")

    print(f"\n=== placebo comparison (filtered arm) ===")
    lift_vs_placebo("filtered", filtered, close_series, ew)

    print(f"\n=== STABILITY: {N_PERIODS} independent periods by entry date "
          f"(worst-case delta must be positive, DHR/MMM-style bar) ===")
    trades_df["period"] = pd.qcut(trades_df.entry_date, N_PERIODS, duplicates="drop")
    rows = []
    for period, g in trades_df.groupby("period", observed=True):
        g_filt = g[g[FEATURE] < THRESHOLD]
        unf_excess = g.excess_ret.mean()
        filt_excess = g_filt.excess_ret.mean() if len(g_filt) else float("nan")
        rows.append({
            "period": str(period), "n_unfiltered": len(g), "n_filtered": len(g_filt),
            "unfiltered_excess": unf_excess, "filtered_excess": filt_excess,
            "delta": filt_excess - unf_excess if pd.notna(filt_excess) else float("nan"),
        })
    period_df = pd.DataFrame(rows)
    period_df.to_csv(os.path.join(OUT_DIR, "stacking_market_depth_period_stability.csv"), index=False)
    print(period_df.to_string(index=False))

    worst_delta = period_df.delta.min()
    n_positive_periods = (period_df.delta > 0).sum()
    print(f"\nworst-case delta across periods: {worst_delta:+.4%}  "
          f"({n_positive_periods}/{len(period_df)} periods show a positive delta)")
    if worst_delta > 0:
        print("PASSES the worst-case bar: filtering improves on the unfiltered young-level arm in every period.")
    else:
        print("FAILS the worst-case bar: at least one period shows the filter making things worse or flat.")

    plot_stability(period_df, os.path.join(OUT_DIR, "stacking_market_depth_stability_chart.png"))
    print(f"\nsaved stacking_market_depth_all_trades.csv / stacking_market_depth_period_stability.csv / "
          f"stacking_market_depth_stability_chart.png to {OUT_DIR}")


def plot_stability(period_df, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(period_df))
    w = 0.35
    ax.bar(x - w / 2, period_df.unfiltered_excess * 100, width=w, color="#9ecae1", label="unfiltered young-level")
    ax.bar(x + w / 2, period_df.filtered_excess * 100, width=w, color="#1f5fa8", label=f"+ {FEATURE}<{THRESHOLD} filter")
    for xi, row in zip(x, period_df.itertuples()):
        ax.annotate(f"n={row.n_unfiltered}", (xi - w / 2, row.unfiltered_excess * 100),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
        ax.annotate(f"n={row.n_filtered}", (xi + w / 2, row.filtered_excess * 100),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"period {i+1}" for i in range(len(period_df))])
    ax.set_ylabel("mean excess return (%)")
    ax.set_title(f"Stability of the {FEATURE}<{THRESHOLD} filter across independent time periods")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
