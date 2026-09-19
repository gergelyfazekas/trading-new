"""The real bar: does market_depth_10d<0 (run_dip_depth.py's cleanest
finding, stacked on the young-level rule) hold up on config.oos_ticker_list
-- 60 large caps disjoint from config.ticker_list, reserved throughout this
project for exactly this purpose ("nothing here may be used to construct a
signal or choose a threshold" -- config.py's own docstring)?

Nothing in this rule was built using OOS data: the fixed a-priori combo
came from ticker_list-wide calibration (tech_levels_notes.md), the young-
level age<5d rule and near_pct/hold_days were already validated/live before
this thread started, and market_depth_10d<0 is an a-priori zero cutoff, not
fitted to any outcome. So this run is a legitimate first look, not a
selection exercise -- but tech_levels_notes.md's own multi-cutoff stability
check on ticker_list (run_stacking_test_market_depth.py) came back mixed
(3 of 4 periods strongly positive, 1 period a real reversal), so this
should be read as "does the pattern generalize," not as a foregone
conclusion.

NVDA excluded (59 of 60 names) -- same convention as
tech_level_oos_strategy.py: NVDA already has a grid-search touch log on
file (dated before the OOS reservation was written down), so it's no
longer an uninspected name even though this rule's combo doesn't depend on
its outcome.

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_oos_validation.py
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
from stock_class import StockList
from tech_level_naive_strategy import load_fixed_combo, attach_benchmark, summarise
from tech_level_continuation_live import MAX_AGE_DAYS
from experiment_lib import build_young_level_trades, placebo_trades
from run_momentum_mood import HOLD_DAYS, NEAR_PCT
from run_dip_depth import trailing_max_depth


def load_oos_series(tickers):
    """Frozen local pull (config.oos), NOT a fresh yfinance call -- same
    convention as tech_level_oos_strategy.py, so this record doesn't depend
    on when the script happens to run. Returns {ticker: (close, volume)},
    matching tech_level_live.pull_all's shape so build_young_level_trades
    can take either.
    """
    sl = StockList(tickers)
    sl.load_data(config.oos)
    series = {}
    for t in tickers:
        try:
            data = sl[t].data
            close = data["close"].dropna()
            if close.empty:
                continue
            volume = data["volume"].reindex(close.index) if "volume" in data.columns else None
            series[t] = (close, volume)
        except Exception:
            continue
    return series

FEATURE = "market_depth_10d"
DEPTH_WINDOW = 10
THRESHOLD = 0.0
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
    tickers = [t for t in config.oos_ticker_list if t != "NVDA"]
    print(f"OOS universe: {len(tickers)} tickers (NVDA excluded -- see module docstring)")
    print(f"loading frozen pull from {config.oos} (not a fresh yfinance call)...")
    series = load_oos_series(tickers)
    print(f"got {len(series)} tickers")

    combo = load_fixed_combo()
    print("building the young-level trade population on the OOS universe...")
    trades_df, close_series, volume_series, ew, failed = build_young_level_trades(
        tickers, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS, series=series
    )
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}")

    trades_df[FEATURE] = [trailing_max_depth(ew, r.entry_date, DEPTH_WINDOW) for r in trades_df.itertuples()]
    trades_df = trades_df.dropna(subset=[FEATURE])
    print(f"{len(trades_df)} young-level (<{MAX_AGE_DAYS}d) OOS trades with a usable {FEATURE}")

    filtered = trades_df[trades_df[FEATURE] < THRESHOLD].copy()
    complement = trades_df[trades_df[FEATURE] >= THRESHOLD].copy()

    os.makedirs(OUT_DIR, exist_ok=True)
    trades_df.to_csv(os.path.join(OUT_DIR, "oos_market_depth_all_trades.csv"), index=False)

    print(f"\n=== POOLED (OOS universe): young-level rule, with vs. without {FEATURE}<{THRESHOLD} filter ===")
    print(f"unfiltered (n={len(trades_df)}): mean_excess={trades_df.excess_ret.mean():.4%}, "
          f"t={trades_df.excess_ret.mean() / trades_df.excess_ret.std() * np.sqrt(len(trades_df)):.2f}")
    print(f"filtered   (n={len(filtered)}): mean_excess={filtered.excess_ret.mean():.4%}, "
          f"t={filtered.excess_ret.mean() / filtered.excess_ret.std() * np.sqrt(len(filtered)):.2f}")
    print(f"complement (n={len(complement)}): mean_excess={complement.excess_ret.mean():.4%}, "
          f"t={complement.excess_ret.mean() / complement.excess_ret.std() * np.sqrt(len(complement)):.2f}")

    by_stock = summarise(filtered, by="stock")
    n_positive = (by_stock.mean_excess > 0).sum()
    print(f"\nfiltered arm by stock: {n_positive}/{len(by_stock)} individually positive on excess")

    print(f"\n=== placebo comparison (filtered arm, OOS universe) ===")
    lift_vs_placebo("filtered", filtered, close_series, ew)

    print(f"\n=== also checking: does the unfiltered young-level rule itself replicate on OOS? ===")
    lift_vs_placebo("unfiltered (sanity check)", trades_df, close_series, ew)

    print(f"\n=== STABILITY: {N_PERIODS} independent periods by entry date (OOS universe) ===")
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
    period_df.to_csv(os.path.join(OUT_DIR, "oos_market_depth_period_stability.csv"), index=False)
    print(period_df.to_string(index=False))

    worst_delta = period_df.delta.min()
    n_positive_periods = (period_df.delta > 0).sum()
    print(f"\nworst-case delta across periods: {worst_delta:+.4%}  "
          f"({n_positive_periods}/{len(period_df)} periods show a positive delta)")

    plot_stability(period_df, os.path.join(OUT_DIR, "oos_market_depth_stability_chart.png"))
    print(f"\nsaved oos_market_depth_all_trades.csv / oos_market_depth_period_stability.csv / "
          f"oos_market_depth_stability_chart.png to {OUT_DIR}")


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
    ax.set_title(f"OOS universe: stability of the {FEATURE}<{THRESHOLD} filter across independent periods")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
