"""Does the DEPTH of the decline into a young support level -- how far price
fell from its recent high to reach the entry price -- predict which
five-day-bounce trades win vs. lose?

Same rigor as run_momentum_mood.py, on the identical young-level (<5d)
trade population: equal-sized quantile buckets (no shrinking-n confound),
direct rank correlation with a Bonferroni bar across the candidate windows,
and -- for whatever survives -- a specificity check against a matched
random-entry placebo, since run_momentum_specificity_check.py showed that
step is what actually separates "this is doing something specific to the
support-level setup" from "this is a generic effect that exists on any
random day."

Features, all causal (computed from the N trading days STRICTLY BEFORE the
entry day, never including it) and expressed as a fraction of the recent
high (positive = price has fallen that much below its recent high; 0 or
negative = today's close is at/above the recent high, i.e. no dip at all):
  - stock_depth_{5,10,20,60}d: (recent N-day high - entry price) / high,
    on the stock's own close series
  - market_depth_{5,10,20,60}d: the identical measure on the equal-weight
    book (was the whole market also down from its recent high, or just
    this stock?)
  - rel_depth_{5,10,20,60}d: stock_depth - market_depth (an idiosyncratic
    dip net of whatever the market itself was doing)

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_dip_depth.py
Saves dip_depth_trades.csv, dip_depth_correlations.csv, and
dip_depth_chart_<feature>.png (every Bonferroni survivor) to this
experiment's data/ folder.
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

HOLD_DAYS = 5
NEAR_PCT = 0.01
N_BUCKETS = 5
WINDOWS = [5, 10, 20, 60]
OUT_DIR = os.path.join(_HERE, "data")


def trailing_max_depth(series, date, n):
    """(recent n-day high - price on `date`) / recent high. The high is
    computed over the n trading days STRICTLY BEFORE date (excluding date
    itself -- causal, no lookahead); price is the series' own value ON
    date, i.e. the actual entry/trigger price. Positive = price has fallen
    that fraction below its recent high; near 0 or negative = today's
    close is at/above the recent high (no meaningful dip happened).
    """
    ts = pd.Timestamp(date)
    if ts not in series.index:
        return None
    pos = series.index.get_loc(ts)
    if pos - n < 0:
        return None
    window = series.iloc[pos - n:pos]
    if window.empty or window.isna().all():
        return None
    high = window.max()
    price = series.iloc[pos]
    if pd.isna(high) or pd.isna(price) or high == 0:
        return None
    return float((high - price) / high)


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

    for n in WINDOWS:
        trades_df[f"stock_depth_{n}d"] = [
            trailing_max_depth(close_series[row.stock], row.entry_date, n) for row in trades_df.itertuples()
        ]
        trades_df[f"market_depth_{n}d"] = [
            trailing_max_depth(ew, row.entry_date, n) for row in trades_df.itertuples()
        ]
        trades_df[f"rel_depth_{n}d"] = trades_df[f"stock_depth_{n}d"] - trades_df[f"market_depth_{n}d"]

    feature_cols = [f"{kind}_depth_{n}d" for kind in ("stock", "market", "rel") for n in WINDOWS]

    os.makedirs(OUT_DIR, exist_ok=True)
    trades_df.to_csv(os.path.join(OUT_DIR, "dip_depth_trades.csv"), index=False)

    bonferroni_bar = 0.05 / len(feature_cols)
    print(f"\n=== correlation of each candidate feature with trade excess_ret (n={len(trades_df)}) ===")
    print(f"({len(feature_cols)} features tested at once -- Bonferroni bar for overall p<0.05 is "
          f"p<{bonferroni_bar:.4f} per feature)")
    corr_rows = []
    for col in feature_cols:
        sub = trades_df.dropna(subset=[col])
        rho, p_rho = stats.spearmanr(sub[col], sub.excess_ret)
        corr_rows.append({"feature": col, "n": len(sub), "spearman_rho": rho, "p_value": p_rho})
    corr_df = pd.DataFrame(corr_rows).sort_values("p_value")
    corr_df.to_csv(os.path.join(OUT_DIR, "dip_depth_correlations.csv"), index=False)
    print(corr_df.to_string(index=False))

    survivors = corr_df[corr_df.p_value < bonferroni_bar]
    if survivors.empty:
        print(f"\nno feature survives the Bonferroni-corrected bar (p<{bonferroni_bar:.4f}) -- "
              f"treat any single p<0.05 above as noise, not a finding")
        return

    print(f"\nsurvives Bonferroni bar (p<{bonferroni_bar:.4f}):")
    print(survivors.to_string(index=False))

    real_counts = trades_df.groupby("stock").size()
    placebo_df = placebo_trades(close_series, real_counts, hold_days=HOLD_DAYS)
    placebo_df = attach_benchmark(placebo_df, ew)
    for n in WINDOWS:
        placebo_df[f"stock_depth_{n}d"] = [
            trailing_max_depth(close_series[row.stock], row.entry_date, n) for row in placebo_df.itertuples()
        ]
        placebo_df[f"market_depth_{n}d"] = [
            trailing_max_depth(ew, row.entry_date, n) for row in placebo_df.itertuples()
        ]
        placebo_df[f"rel_depth_{n}d"] = placebo_df[f"stock_depth_{n}d"] - placebo_df[f"market_depth_{n}d"]

    print(f"\n=== specificity check: real (young-level) vs. matched placebo, for each survivor ===")
    for _, row in survivors.iterrows():
        feat = row["feature"]
        real_sub = trades_df.dropna(subset=[feat])
        placebo_sub = placebo_df.dropna(subset=[feat])
        r_rho, r_p = stats.spearmanr(real_sub[feat], real_sub.excess_ret)
        p_rho, p_p = stats.spearmanr(placebo_sub[feat], placebo_sub.excess_ret)
        print(f"{feat:16s}: real rho={r_rho:+.3f} (p={r_p:.4f}, n={len(real_sub)})   "
              f"placebo rho={p_rho:+.3f} (p={p_p:.4f}, n={len(placebo_sub)})")
        plot_feature_buckets(trades_df, feat, os.path.join(OUT_DIR, f"dip_depth_chart_{feat}.png"))

    print(f"\nsaved dip_depth_trades.csv / dip_depth_correlations.csv / dip_depth_chart_<feature>.png to {OUT_DIR}")


def plot_feature_buckets(trades_df, feature, out_path):
    sub = trades_df.dropna(subset=[feature]).copy()
    sub["bucket"] = pd.qcut(sub[feature], N_BUCKETS, duplicates="drop")

    rows = []
    for b, g in sub.groupby("bucket", observed=True):
        n = len(g)
        t_stat = g.excess_ret.mean() / g.excess_ret.std() * np.sqrt(n) if n > 2 else float("nan")
        rows.append({"n": n, "mean_feature": g[feature].mean(), "mean_excess": g.excess_ret.mean(), "t_stat": t_stat})
    bucket_df = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(bucket_df))
    ax1.bar(x, bucket_df.mean_excess * 100, color="#1f5fa8", zorder=3)
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

    ax2.scatter(sub[feature] * 100, sub.excess_ret * 100, s=10, alpha=0.35, color="#1f5fa8")
    z = np.polyfit(sub[feature], sub.excess_ret * 100, 1)
    xs = np.linspace(sub[feature].quantile(0.01), sub[feature].quantile(0.99), 50)
    ax2.plot(xs * 100, np.poly1d(z)(xs), color="#d62728", linewidth=2, label="linear fit")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlim(sub[feature].quantile(0.01) * 100, sub[feature].quantile(0.99) * 100)
    ax2.set_xlabel(f"{feature} (%)")
    ax2.set_ylabel("trade excess return (%)")
    ax2.set_title("Per-trade scatter (1st/99th pct trimmed)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
