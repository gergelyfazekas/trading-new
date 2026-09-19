"""Symmetry check on the market_depth_10d finding: run_dip_depth.py showed
buying young support levels works BETTER when the broad market is calm and
near its highs. Does the opposite hold too -- genuinely WORSE (not just
weaker) performance when the market is volatile, or sitting near its own
recent LOWS specifically (a longer-horizon, more severe condition than
"a bit off its 10-day high")?

This matters because everything found so far in this thread has been a
graded continuum where the "bad" side of every split stayed solidly
profitable (market_depth_10d's own complement group: +0.94% ticker_list,
+0.93% OOS, both t>8). A genuine two-sided regime effect -- where the worst
bucket is flat or negative, not just smaller -- would be a materially
stronger and more actionable finding than anything confirmed until now.

New candidate features, all causal (N trading days strictly before entry,
never including it), on the equal-weight book:
  - market_dist_from_low_{20,60,120}d: (price - recent N-day low) / low --
    LARGER means further above the recent low (safer); near zero means
    sitting AT/near a real trough. Longer windows than market_depth's 10d,
    since "near its low" is only a meaningful, distinct condition over a
    longer horizon -- every market is trivially near its own 5-day low
    fairly often.
  - market_vol_{5,10,20}d: raw realized volatility (std of daily returns)
    over the window -- tests "volatile" directly, independent of direction.
  - market_vol_ratio_5_60: 5-day realized vol / 60-day realized vol -- a
    volatility SPIKE indicator (same ratio-to-baseline idea as the
    volume-ratio check), not just an absolute level.

Same rigor as every other feature test in this package: Bonferroni bar
across all candidates, equal-sized quantile buckets (no shrinking-n
confound), and a placebo specificity check for whatever survives.

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_market_regime_extremes.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
from run_momentum_mood import HOLD_DAYS, NEAR_PCT

LOW_WINDOWS = [20, 60, 120]
VOL_WINDOWS = [5, 10, 20]
N_BUCKETS = 5
OUT_DIR = os.path.join(_HERE, "data")


def trailing_low_proximity(series, date, n):
    """(price on date - recent n-day low) / low, low computed over the n
    trading days strictly before date (excluding date itself)."""
    ts = pd.Timestamp(date)
    if ts not in series.index:
        return None
    pos = series.index.get_loc(ts)
    if pos - n < 0:
        return None
    window = series.iloc[pos - n:pos]
    if window.empty or window.isna().all():
        return None
    low = window.min()
    price = series.iloc[pos]
    if pd.isna(low) or pd.isna(price) or low == 0:
        return None
    return float((price - low) / low)


def realized_vol(series, date, n):
    """Std of daily pct returns over the n trading days strictly before
    date (excluding date itself)."""
    ts = pd.Timestamp(date)
    if ts not in series.index:
        return None
    pos = series.index.get_loc(ts)
    if pos - n < 0:
        return None
    window = series.iloc[pos - n:pos]
    rets = window.pct_change().dropna()
    if len(rets) < 2:
        return None
    return float(rets.std())


def main():
    tickers = config.ticker_list
    combo = load_fixed_combo()
    print(f"pulling full history for {len(tickers)} tickers and building the young-level trade population...")
    trades_df, close_series, volume_series, ew, failed = build_young_level_trades(
        tickers, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS
    )
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}")
    print(f"{len(trades_df)} young-level (<{MAX_AGE_DAYS}d) trades")

    for n in LOW_WINDOWS:
        trades_df[f"market_dist_from_low_{n}d"] = [
            trailing_low_proximity(ew, r.entry_date, n) for r in trades_df.itertuples()
        ]
    for n in VOL_WINDOWS:
        trades_df[f"market_vol_{n}d"] = [realized_vol(ew, r.entry_date, n) for r in trades_df.itertuples()]
    trades_df["market_vol_ratio_5_60"] = trades_df["market_vol_5d"] / [
        realized_vol(ew, r.entry_date, 60) for r in trades_df.itertuples()
    ]

    feature_cols = ([f"market_dist_from_low_{n}d" for n in LOW_WINDOWS]
                     + [f"market_vol_{n}d" for n in VOL_WINDOWS]
                     + ["market_vol_ratio_5_60"])

    os.makedirs(OUT_DIR, exist_ok=True)
    trades_df.to_csv(os.path.join(OUT_DIR, "regime_extremes_trades.csv"), index=False)

    bonferroni_bar = 0.05 / len(feature_cols)
    print(f"\n=== correlation of each candidate feature with trade excess_ret (n={len(trades_df)}) ===")
    print(f"({len(feature_cols)} features -- Bonferroni bar p<{bonferroni_bar:.4f})")
    corr_rows = []
    for col in feature_cols:
        sub = trades_df.dropna(subset=[col])
        rho, p_rho = stats.spearmanr(sub[col], sub.excess_ret)
        corr_rows.append({"feature": col, "n": len(sub), "spearman_rho": rho, "p_value": p_rho})
    corr_df = pd.DataFrame(corr_rows).sort_values("p_value")
    corr_df.to_csv(os.path.join(OUT_DIR, "regime_extremes_correlations.csv"), index=False)
    print(corr_df.to_string(index=False))

    survivors = corr_df[corr_df.p_value < bonferroni_bar]
    if survivors.empty:
        print(f"\nno feature survives the Bonferroni-corrected bar -- "
              f"no evidence of a symmetric 'volatile/near-lows = worse' effect beyond what "
              f"market_depth_10d already showed")
        return

    print(f"\nsurvives Bonferroni bar:")
    print(survivors.to_string(index=False))

    real_counts = trades_df.groupby("stock").size()
    placebo_df = placebo_trades(close_series, real_counts, hold_days=HOLD_DAYS)
    placebo_df = attach_benchmark(placebo_df, ew)
    for n in LOW_WINDOWS:
        placebo_df[f"market_dist_from_low_{n}d"] = [
            trailing_low_proximity(ew, r.entry_date, n) for r in placebo_df.itertuples()
        ]
    for n in VOL_WINDOWS:
        placebo_df[f"market_vol_{n}d"] = [realized_vol(ew, r.entry_date, n) for r in placebo_df.itertuples()]
    placebo_df["market_vol_ratio_5_60"] = placebo_df["market_vol_5d"] / [
        realized_vol(ew, r.entry_date, 60) for r in placebo_df.itertuples()
    ]

    print(f"\n=== specificity check + worst-bucket look, for each survivor ===")
    for _, row in survivors.iterrows():
        feat = row["feature"]
        real_sub = trades_df.dropna(subset=[feat])
        placebo_sub = placebo_df.dropna(subset=[feat])
        r_rho, r_p = stats.spearmanr(real_sub[feat], real_sub.excess_ret)
        p_rho, p_p = stats.spearmanr(placebo_sub[feat], placebo_sub.excess_ret)
        print(f"\n{feat}: real rho={r_rho:+.3f} (p={r_p:.4f})   placebo rho={p_rho:+.3f} (p={p_p:.4f})")

        sub = real_sub.copy()
        sub["bucket"] = pd.qcut(sub[feat], N_BUCKETS, duplicates="drop")
        for b, g in sub.groupby("bucket", observed=True):
            n = len(g)
            t_stat = g.excess_ret.mean() / g.excess_ret.std() * np.sqrt(n) if n > 2 else float("nan")
            print(f"  [{b}]  n={n:4d}  mean_excess={g.excess_ret.mean():+.4%}  t={t_stat:+.2f}")
        plot_feature_buckets(trades_df, feat, os.path.join(OUT_DIR, f"regime_extremes_chart_{feat}.png"))

    print(f"\nsaved regime_extremes_trades.csv / regime_extremes_correlations.csv / "
          f"regime_extremes_chart_<feature>.png to {OUT_DIR}")


def plot_feature_buckets(trades_df, feature, out_path):
    sub = trades_df.dropna(subset=[feature]).copy()
    sub["bucket"] = pd.qcut(sub[feature], N_BUCKETS, duplicates="drop")

    rows = []
    for b, g in sub.groupby("bucket", observed=True):
        n = len(g)
        t_stat = g.excess_ret.mean() / g.excess_ret.std() * np.sqrt(n) if n > 2 else float("nan")
        rows.append({"n": n, "mean_feature": g[feature].mean(), "mean_excess": g.excess_ret.mean(), "t_stat": t_stat})
    bucket_df = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(7, 5.5))
    x = np.arange(len(bucket_df))
    colors = ["#d62728" if v < 0 else "#1f5fa8" for v in bucket_df.mean_excess]
    ax1.bar(x, bucket_df.mean_excess * 100, color=colors, zorder=3)
    for xi, row in zip(x, bucket_df.itertuples()):
        ax1.annotate(f"n={row.n}\nt={row.t_stat:.1f}", (xi, row.mean_excess * 100),
                      textcoords="offset points", xytext=(0, 4 if row.mean_excess >= 0 else -14),
                      ha="center", fontsize=8)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Q{i+1}\n{r:.1%}" for i, r in enumerate(bucket_df.mean_feature)], fontsize=8)
    ax1.set_ylabel("mean excess return (%)")
    ax1.set_title(f"Equal-sized buckets by {feature}")
    ax1.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
