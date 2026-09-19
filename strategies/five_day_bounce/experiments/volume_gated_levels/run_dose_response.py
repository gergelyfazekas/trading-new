"""Direct dose-response test: does a young level's own birth-day volume
ratio (continuous, not thresholded) predict its trade's excess return?

run_sweep.py's threshold sweep left a real ambiguity, raised directly by the
user: as min_ratio rises, mean excess return climbs *and* the sample
shrinks, so a falling t-stat there is exactly what a genuine dose-response
effect would ALSO look like (rarer, more extreme, more informative events,
less power) -- not necessarily overfitting to noise. But it's equally what
pure noise looks like once you keep restricting to a shrinking, more
extreme tail, and the sweep's threshold points are NESTED subsets of each
other (min_ratio=1.25's trades are a subset of min_ratio=1.0's), not
independent draws -- which mechanically smooths the curve regardless of
whether anything real is going on. Smoothness alone can't settle it.

This script removes that confound: bucket the SAME young-level trade
population into independent, roughly-equal-sized quantile groups by each
trade's own continuous birth-day volume ratio (no shrinking-n effect,
since bucket sizes stay flat by construction), plus a rank correlation
across all trades directly. A real dose-response should show a consistent
trend across buckets with STABLE power, and a significant positive
correlation. A pattern that only ever showed up as a shrinking, noisier
tail in the threshold sweep would not survive this.

Uses the ungated (baseline) levels only -- every young-level trade is kept,
each carrying its own level's birth-day volume ratio as a covariate,
computed with the exact same causal definition as everywhere else in this
package (build_levels_volume_gated.volume_ratio: that day's volume over the
mean of the 10 trading days strictly before it).

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_dose_response.py
Saves dose_response_trades.csv, dose_response_buckets.csv, and
dose_response_chart.png to this experiment's data/ folder.
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
from build_levels_volume_gated import volume_ratio

HOLD_DAYS = 5
NEAR_PCT = 0.01
N_BUCKETS = 5
OUT_DIR = os.path.join(_HERE, "data")


def main():
    tickers = config.ticker_list
    combo = load_fixed_combo()
    print(f"pulling full history for {len(tickers)} tickers and building the young-level trade population...")
    trades_df, close_series, volume_series, ew, failed = build_young_level_trades(
        tickers, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS
    )
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}")

    trades_df["birth_volume_ratio"] = [
        volume_ratio(volume_series[row.stock], row.support_birth_date, lookback=10)
        for row in trades_df.itertuples()
    ]
    n_before = len(trades_df)
    trades_df = trades_df.dropna(subset=["birth_volume_ratio"])
    print(f"\n{n_before} young-level (<{MAX_AGE_DAYS}d) trades, {len(trades_df)} with a usable "
          f"birth-day volume ratio (dropped {n_before - len(trades_df)} too close to the start of history)")

    os.makedirs(OUT_DIR, exist_ok=True)
    trades_df.to_csv(os.path.join(OUT_DIR, "dose_response_trades.csv"), index=False)

    # direct correlation, no bucketing/thresholding at all
    rho, p_rho = stats.spearmanr(trades_df.birth_volume_ratio, trades_df.excess_ret)
    r, p_r = stats.pearsonr(trades_df.birth_volume_ratio, trades_df.excess_ret)
    print(f"\nSpearman rank correlation (birth_volume_ratio, excess_ret): rho={rho:.3f}, p={p_rho:.4f}, n={len(trades_df)}")
    print(f"Pearson correlation:                                          r={r:.3f}, p={p_r:.4f}")

    # equal-sized quantile buckets -- no shrinking-n confound
    trades_df["bucket"], edges = pd.qcut(trades_df.birth_volume_ratio, N_BUCKETS, retbins=True, duplicates="drop")

    rows = []
    for b, g in trades_df.groupby("bucket", observed=True):
        n = len(g)
        t_stat = g.excess_ret.mean() / g.excess_ret.std() * np.sqrt(n) if n > 2 else float("nan")
        rows.append({
            "ratio_range": str(b), "n": n,
            "mean_birth_ratio": g.birth_volume_ratio.mean(),
            "hit_rate": (g.ret > 0).mean(),
            "mean_excess": g.excess_ret.mean(),
            "t_stat": t_stat,
        })
    bucket_df = pd.DataFrame(rows)
    bucket_df.to_csv(os.path.join(OUT_DIR, "dose_response_buckets.csv"), index=False)
    print(f"\n=== {N_BUCKETS} equal-sized buckets by birth-day volume ratio ===")
    print(bucket_df.to_string(index=False))

    # single overall placebo reference (random entries, matched to total
    # young-level trade count/frequency) -- flat by construction, since
    # random entries don't know about any level's birth-day volume
    placebo_df = placebo_trades(close_series, trades_df.groupby("stock").size(), hold_days=HOLD_DAYS)
    placebo_df = attach_benchmark(placebo_df, ew)
    placebo_excess = placebo_df.excess_ret.mean()
    print(f"\nreference: placebo mean_excess (matched overall count/frequency) = {placebo_excess:.4%}")

    plot_dose_response(trades_df, bucket_df, placebo_excess, os.path.join(OUT_DIR, "dose_response_chart.png"))
    print(f"\nsaved dose_response_trades.csv / dose_response_buckets.csv / dose_response_chart.png to {OUT_DIR}")


def plot_dose_response(trades_df, bucket_df, placebo_excess, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    x = np.arange(len(bucket_df))
    ax1.bar(x, bucket_df.mean_excess * 100, color="#1f5fa8", zorder=3)
    for xi, row in zip(x, bucket_df.itertuples()):
        ax1.annotate(f"n={row.n}\nt={row.t_stat:.1f}", (xi, row.mean_excess * 100),
                      textcoords="offset points", xytext=(0, 4 if row.mean_excess >= 0 else -14),
                      ha="center", fontsize=8)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(placebo_excess * 100, color="#d62728", linestyle="--", linewidth=1.2,
                 label=f"placebo ref ({placebo_excess:.2%})")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Q{i+1}\n{r:.2f}x avg" for i, r in enumerate(bucket_df.mean_birth_ratio)], fontsize=8)
    ax1.set_ylabel("mean excess return (%)")
    ax1.set_title(f"Equal-sized quantile buckets\nby birth-day volume ratio (n~{bucket_df.n.iloc[0]}/bucket)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25, axis="y")

    ax2.scatter(trades_df.birth_volume_ratio, trades_df.excess_ret * 100, s=10, alpha=0.35, color="#1f5fa8")
    z = np.polyfit(trades_df.birth_volume_ratio, trades_df.excess_ret * 100, 1)
    xs = np.linspace(trades_df.birth_volume_ratio.min(), trades_df.birth_volume_ratio.quantile(0.99), 50)
    ax2.plot(xs, np.poly1d(z)(xs), color="#d62728", linewidth=2, label="linear fit")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlim(0, trades_df.birth_volume_ratio.quantile(0.99))
    ax2.set_xlabel("birth-day volume ratio")
    ax2.set_ylabel("trade excess return (%)")
    ax2.set_title("Per-trade scatter (99th pct trimmed on x-axis for readability)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
