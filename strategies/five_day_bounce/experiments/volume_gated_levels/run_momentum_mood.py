"""Tests two "market psychology" hypotheses on the same young-level (<5d)
trade population used by run_dose_response.py: does short-term price
momentum leading INTO a buy signal -- either the stock's own trend
("company mood") or the broader equal-weight book's trend ("market mood")
-- predict which five-day-bounce trades turn out to be winners vs losers?

Motivated directly by the user's own instinct after looking at the AMZN
walk-forward charts: the technical level itself looks reasonable in every
case, but some trades bounce for a genuine profit and some just keep
sliding through, and birth-day volume (tested and rejected in
run_dose_response.py) doesn't explain the difference. This tests the next
candidate.

Same rigor as run_dose_response.py's fix, applied from the start rather
than as a follow-up correction: independent, equal-sized quantile buckets
(no shrinking-n confound) plus a direct rank correlation, for every
candidate feature. With 12 candidate windows/features tested at once, also
reports the Bonferroni bar explicitly rather than reading any single
p<0.05 as meaningful on its own.

Features, all trailing and causal -- computed over the N trading days
STRICTLY BEFORE the entry day (excluding the entry day itself), so they
don't mechanically overlap with the entry trigger's own required dip toward
the support band:
  - stock_mom_{5,10,20,60}d: the stock's own cumulative return
  - market_mom_{5,10,20,60}d: the equal-weight book's cumulative return
    over the same dates (broad "market mood"/regime)
  - rel_mom_{5,10,20,60}d: stock_mom - market_mom ("company mood" isolated
    from the market's own mood)

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_momentum_mood.py
Saves momentum_trades.csv, momentum_correlations.csv, and
momentum_chart.png (best-ranked feature) to this experiment's data/ folder.
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
from tech_level_naive_strategy import load_fixed_combo
from tech_level_continuation_live import MAX_AGE_DAYS
from experiment_lib import build_young_level_trades

HOLD_DAYS = 5
NEAR_PCT = 0.01
N_BUCKETS = 5
WINDOWS = [5, 10, 20, 60]
OUT_DIR = os.path.join(_HERE, "data")


def trailing_return(series, date, n):
    """Cumulative return over the n trading days strictly before `date`
    (ending the day before date, not including date itself) -- causal, and
    deliberately excludes the entry day so this doesn't just re-measure the
    mechanical dip toward the support band the entry rule itself requires.
    """
    ts = pd.Timestamp(date)
    if ts not in series.index:
        return None
    pos = series.index.get_loc(ts)
    if pos - 1 - n < 0:
        return None
    start, end = series.iloc[pos - 1 - n], series.iloc[pos - 1]
    if pd.isna(start) or pd.isna(end) or start == 0:
        return None
    return float(end / start - 1.0)


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
        trades_df[f"stock_mom_{n}d"] = [
            trailing_return(close_series[row.stock], row.entry_date, n) for row in trades_df.itertuples()
        ]
        trades_df[f"market_mom_{n}d"] = [
            trailing_return(ew, row.entry_date, n) for row in trades_df.itertuples()
        ]
        trades_df[f"rel_mom_{n}d"] = trades_df[f"stock_mom_{n}d"] - trades_df[f"market_mom_{n}d"]

    feature_cols = [f"{kind}_mom_{n}d" for kind in ("stock", "market", "rel") for n in WINDOWS]

    os.makedirs(OUT_DIR, exist_ok=True)
    trades_df.to_csv(os.path.join(OUT_DIR, "momentum_trades.csv"), index=False)

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
    corr_df.to_csv(os.path.join(OUT_DIR, "momentum_correlations.csv"), index=False)
    print(corr_df.to_string(index=False))

    survivors = corr_df[corr_df.p_value < bonferroni_bar]
    if survivors.empty:
        print(f"\nno feature survives the Bonferroni-corrected bar (p<{bonferroni_bar:.4f}) -- "
              f"treat any single p<0.05 above as noise, not a finding")
    else:
        print(f"\nsurvives Bonferroni bar (p<{bonferroni_bar:.4f}):")
        print(survivors.to_string(index=False))

    best_feature = corr_df.iloc[0]["feature"]
    print(f"\nplotting equal-sized quantile buckets for the strongest candidate: {best_feature}")
    plot_feature_buckets(trades_df, best_feature, os.path.join(OUT_DIR, "momentum_chart.png"))
    print(f"\nsaved momentum_trades.csv / momentum_correlations.csv / momentum_chart.png to {OUT_DIR}")


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
