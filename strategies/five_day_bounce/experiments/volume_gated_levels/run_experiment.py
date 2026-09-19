"""Experiment: does gating level birth on volume (birth-day volume ratio >=
min_ratio, see build_levels_volume_gated.py) change the five-day-bounce
continuation strategy's trades for the better?

Motivated by a suspicious pattern spotted on the walk-forward charts
(strategies/five_day_bounce/tech_level_walkforward_demo.py) -- the hunch
tested here is the opposite of the plain "does average volume rise on
signal days" check already run and found weak/negative on AMZN: maybe
levels born on *already* high-volume days specifically are the informative
subset, even if the average isn't elevated.

Deliberately small and fast, not a rigorous validation: a handful of
tickers, a recent window only (not full 2015-2026 history), fixed a-priori
combo (data/fixed_combo.json), one gate threshold. Meant to answer "is this
worth a real backtest" before spending that effort -- see
tech_levels_notes.md's validation-first convention (state negatives
plainly, don't oversell a small sample).

Does NOT modify tech_levels.py, tech_level_naive_strategy.py, or
tech_level_continuation_live.py -- only imports them read-only. The gated
level construction lives entirely in build_levels_volume_gated.py, next to
this file.

Usage:
  ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_experiment.py [TICKER ...]
  (omit tickers to sample 5 at random from config.ticker_list)

Results saved under this experiment's own data/ folder, not the shared
strategies/five_day_bounce/data/ directory.
"""
import datetime
import os
import random
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIVE_DAY_BOUNCE = os.path.join(_HERE, "..", "..")
_REPO_ROOT = os.path.join(_FIVE_DAY_BOUNCE, "..", "..")
sys.path.insert(0, os.path.abspath(_REPO_ROOT))
sys.path.insert(0, os.path.abspath(_FIVE_DAY_BOUNCE))
sys.path.insert(0, _HERE)

import config
from tech_level_live import pull_series
from tech_level_naive_strategy import load_fixed_combo, simulate, equal_weight_curve, attach_benchmark, summarise
from experiment_lib import build_levels_baseline, build_levels_gated

N_TICKERS = 5
YEARS_BACK = 2          # test window -- NOT the full 2015-2026 history
HOLD_DAYS = 5
NEAR_PCT = 0.01
MIN_VOLUME_RATIO = 1.0
OUT_DIR = os.path.join(_HERE, "data")


def main():
    args = sys.argv[1:]
    tickers = args if args else random.sample(config.ticker_list, N_TICKERS)
    print(f"tickers: {tickers}")
    print(f"test window: last {YEARS_BACK} years | hold_days={HOLD_DAYS} | near_pct={NEAR_PCT:.0%} | "
          f"min_volume_ratio={MIN_VOLUME_RATIO}")

    combo = load_fixed_combo()
    cutoff = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=365 * YEARS_BACK))

    recent_series = {}
    level_counts = []
    baseline_trades, gated_trades = [], []

    for t in tickers:
        try:
            close, volume = pull_series(t)
        except Exception as exc:
            print(f"  {t}: SKIPPED ({exc.__class__.__name__}: {exc})")
            continue
        close = close.copy()
        close.index = pd.to_datetime(close.index)
        volume = volume.copy()
        volume.index = pd.to_datetime(volume.index)

        levels_baseline = build_levels_baseline(close, combo)
        levels_gated, n_gated_out = build_levels_gated(close, volume, combo)

        close_recent = close.loc[close.index >= cutoff]
        if close_recent.empty:
            print(f"  {t}: no data in the recent window, skipped")
            continue
        recent_series[t] = close_recent

        level_counts.append({
            "ticker": t, "n_levels_baseline": len(levels_baseline), "n_levels_gated": len(levels_gated),
            "n_births_blocked_by_gate": n_gated_out,
        })

        baseline_trades.extend(simulate(t, close_recent, levels_baseline, hold_days=HOLD_DAYS, near_pct=NEAR_PCT))
        gated_trades.extend(simulate(t, close_recent, levels_gated, hold_days=HOLD_DAYS, near_pct=NEAR_PCT))

        print(f"  {t}: {len(levels_baseline)} baseline levels, {len(levels_gated)} gated levels "
              f"({n_gated_out} births blocked)")

    if not recent_series:
        print("no usable data -- aborting")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    counts_df = pd.DataFrame(level_counts)
    counts_df.to_csv(os.path.join(OUT_DIR, "level_counts.csv"), index=False)

    ew = equal_weight_curve(recent_series)

    base_df = pd.DataFrame(baseline_trades)
    gate_df = pd.DataFrame(gated_trades)

    print(f"\n=== level counts ===\n{counts_df.to_string(index=False)}")

    print("\n=== trades ===")
    if base_df.empty:
        print("baseline: 0 trades in this window")
    else:
        base_df = attach_benchmark(base_df, ew)
        base_df.to_csv(os.path.join(OUT_DIR, "baseline_trades.csv"), index=False)
        print(f"baseline (n={len(base_df)}):")
        print(summarise(base_df).to_string(index=False))

    if gate_df.empty:
        print("volume-gated: 0 trades in this window")
    else:
        gate_df = attach_benchmark(gate_df, ew)
        gate_df.to_csv(os.path.join(OUT_DIR, "gated_trades.csv"), index=False)
        print(f"volume-gated (n={len(gate_df)}):")
        print(summarise(gate_df).to_string(index=False))

    print(f"\nsaved level_counts.csv / baseline_trades.csv / gated_trades.csv to {OUT_DIR}")
    print("\nsample size is small and the window is short by design -- this is a directional look, "
          "not a validated result. If it looks promising, re-run on the full ticker_list/date range "
          "before trusting it.")


if __name__ == "__main__":
    main()
