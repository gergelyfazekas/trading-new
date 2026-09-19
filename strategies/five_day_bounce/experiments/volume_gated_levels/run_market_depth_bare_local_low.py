"""Re-tests market_depth_10d (run_dip_depth.py's strongest, cleanest
finding) against the BARE-LOCAL-LOW population instead of the two-touch
technical-level population every other script in this folder uses.

Why this matters: tech_level_causal_check.py's Step C (2026-09-15) and
tech_level_confluence_check.py (2026-09-19) both independently found that
requiring a two-touch technical level adds nothing over a bare, unmatched
local low -- the real mechanism behind "the five day bounce" is a generic
short-horizon reversal off a recently confirmed trough, not anything
specific to technical levels. Every feature test run so far in this
experiments/ folder (volume, momentum, dip depth, market regime, sizing)
was built on `build_levels_baseline`/`simulate_age_gated`, i.e. the
two-touch-level version -- so none of it has actually been checked against
what this project's own other sections concluded is the *real* underlying
population. This script closes that gap for the single strongest finding.

Reuses existing, already-validated machinery read-only -- no new detector:
  - tech_level_causal_check.load_universe / causal_entry_and_resistance /
    causal_exit (fully causal entry+exit, same as Step A/C)
  - tech_level_confluence_check.build_full_placebo_trades (the FULL,
    unsampled bare-local-low population, not Step C's per-ticker-count-
    matched subsample -- matches how Framing 2 was built)
  - this folder's own trailing_max_depth / plot_feature_buckets
    (run_dip_depth.py) and placebo_trades (experiment_lib.py) for the
    dose-response + specificity-check methodology

Run on BOTH config.ticker_list and config.oos_ticker_list (-NVDA), same as
every other confirmatory test in this folder.

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_market_depth_bare_local_low.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIVE_DAY_BOUNCE = os.path.join(_HERE, "..", "..")
_REPO_ROOT = os.path.join(_FIVE_DAY_BOUNCE, "..", "..")
sys.path.insert(0, os.path.abspath(_REPO_ROOT))
sys.path.insert(0, os.path.abspath(_FIVE_DAY_BOUNCE))
sys.path.insert(0, _HERE)

from tech_level_naive_strategy import load_fixed_combo, equal_weight_curve, attach_benchmark
from tech_level_causal_check import load_universe
from tech_level_confluence_check import build_full_placebo_trades
from experiment_lib import placebo_trades
from run_dip_depth import trailing_max_depth, plot_feature_buckets, N_BUCKETS

FEATURE = "market_depth_10d"
DEPTH_WINDOW = 10
HOLD_DAYS = 5
OUT_DIR = os.path.join(_HERE, "data")


def run_universe(which):
    print(f"\n{'='*70}\n{which}: bare-local-low population\n{'='*70}")
    combo = load_fixed_combo()
    print(f"loading {which} universe...")
    series = load_universe(which)
    print(f"{len(series)} tickers")

    print("building the FULL (unsampled) bare-local-low candidate population "
          "(causal entry+exit, no level/band machinery)...")
    full_placebo = build_full_placebo_trades(series, combo)
    if full_placebo.empty:
        print("no candidates -- aborting this universe")
        return None
    print(f"{len(full_placebo)} bare-local-low trades "
          f"({full_placebo.real_confirms.mean():.1%} also confirmed by a real two-touch level)")

    ew = equal_weight_curve(series)
    full_placebo[FEATURE] = [
        trailing_max_depth(ew, r.entry_date, DEPTH_WINDOW) for r in full_placebo.itertuples()
    ]
    full_placebo = full_placebo.dropna(subset=[FEATURE])
    print(f"{len(full_placebo)} with a usable {FEATURE}")

    rho, p = stats.spearmanr(full_placebo[FEATURE], full_placebo.excess_ret)
    print(f"\nSpearman rank correlation ({FEATURE}, excess_ret): rho={rho:+.3f}, p={p:.4f}, n={len(full_placebo)}")

    # equal-sized bucket table
    sub = full_placebo.copy()
    sub["bucket"] = pd.qcut(sub[FEATURE], N_BUCKETS, duplicates="drop")
    print(f"\n{N_BUCKETS} equal-sized buckets by {FEATURE}:")
    for b, g in sub.groupby("bucket", observed=True):
        n = len(g)
        t_stat = g.excess_ret.mean() / g.excess_ret.std() * np.sqrt(n) if n > 2 else float("nan")
        print(f"  [{b}]  n={n:4d}  mean_excess={g.excess_ret.mean():+.4%}  t={t_stat:+.2f}")

    # specificity check: matched random-entry placebo (of THIS population's own frequency)
    real_counts = full_placebo.groupby("stock").size()
    close_series = {t: c for t, c in series.items()}
    rand_df = placebo_trades(close_series, real_counts, hold_days=HOLD_DAYS)
    rand_df = attach_benchmark(rand_df, ew)
    rand_df[FEATURE] = [trailing_max_depth(ew, r.entry_date, DEPTH_WINDOW) for r in rand_df.itertuples()]
    rand_df = rand_df.dropna(subset=[FEATURE])
    r_rho, r_p = stats.spearmanr(rand_df[FEATURE], rand_df.excess_ret)
    print(f"\nspecificity check -- random-entry placebo (matched frequency): rho={r_rho:+.3f}, "
          f"p={r_p:.4f}, n={len(rand_df)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"bare_local_low_{which}_trades.csv")
    full_placebo.to_csv(out_path, index=False)
    chart_path = os.path.join(OUT_DIR, f"bare_local_low_{which}_chart.png")
    plot_feature_buckets(full_placebo, FEATURE, chart_path)
    print(f"\nsaved {out_path} and {chart_path}")

    return {"which": which, "n": len(full_placebo), "rho": rho, "p": p,
            "placebo_rho": r_rho, "placebo_p": r_p}


def main():
    results = [run_universe("ticker_list"), run_universe("oos")]
    results = [r for r in results if r is not None]

    print(f"\n{'='*70}\nSUMMARY: {FEATURE} vs. bare-local-low excess_ret\n{'='*70}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
