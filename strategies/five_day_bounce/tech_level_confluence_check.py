"""Confluence check, follow-up to tech_level_causal_check.py's Step C
placebo (see tech_levels_notes.md, 2026-09-15 "Fully causal entry+exit,
real inference, and a reversal placebo"). Step C found that requiring a
two-touch technical level adds no measurable edge over a bare, unmatched
local-low condition -- the two entry rules produce statistically
indistinguishable trades when tested as separate populations.

This asks a different question: when the two conditions *agree* -- the
same (ticker, date) satisfies both the real technical-level entry AND the
bare-local-low placebo entry on the same day -- is that a stronger signal
than either alone? Two framings of the same underlying overlap, run from
each side so the comparison is symmetric:

  Framing 1 (from the real side): among tech_level_causal_check.py's
  causally-confirmed REAL trades (data/causal_full_{which}.csv), does an
  independently-confirmed bare local low (placebo_raw_days) ALSO exist on
  the same entry date? Split BOTH vs REAL-ONLY.

  Framing 2 (from the placebo side): build the FULL (unsampled) population
  of bare-local-low candidates -- every day placebo_raw_days flags, not the
  per-ticker-count-matched subsample Step C used for its aggregate
  comparison -- and label each with whether the real technical-level
  condition ALSO holds that day. Split BOTH vs PLACEBO-ONLY.

  Both framings share the same BOTH set in principle (same underlying
  overlap), but are built from different starting populations (real trades
  already have one-position-at-a-time blocking baked in from the original
  backtest; placebo candidates are blocked independently by
  build_placebo_candidates) -- reporting both sides shows whether that
  blocking difference matters, rather than assuming it doesn't.

Uses causal_entry_and_resistance/causal_exit for entry+exit exactly as
Step A/C do (no truncation cheats: every condition check and every exit is
computed from data truncated to the date being decided). Not a new
detector -- reuses tech_level_causal_check.py's placebo_raw_days,
causal_entry_and_resistance, causal_exit, and the same fixed a-priori combo.

Run (from repo root): ./venv/bin/python strategies/five_day_bounce/tech_level_confluence_check.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tech_level_naive_strategy import load_fixed_combo, equal_weight_curve, attach_benchmark
from tech_level_causal_check import (
    load_universe, pooled, block_bootstrap_ci,
    causal_entry_and_resistance, causal_exit, placebo_raw_days, build_placebo_candidates,
    DATA_DIR, HOLD_DAYS,
)

CAUSAL_FULL_PATH = os.path.join(DATA_DIR, "causal_full_{which}.csv")


def label_real_trades_by_confluence(series, causal_df, combo):
    """Framing 1: tag each already-causally-confirmed REAL trade with
    whether placebo_raw_days also flags its exact entry date -- i.e.
    whether an independently-confirmed bare local low backs the same entry,
    not just the two-touch level the real rule matched on.
    """
    raw_cache = {}
    flags = []
    for row in causal_df.itertuples():
        t = row.stock
        close = series[t]
        if t not in raw_cache:
            raw_cache[t] = placebo_raw_days(close, combo)
        i = close.index.get_loc(pd.Timestamp(row.entry_date))
        flags.append(i in raw_cache[t])
    out = causal_df.copy()
    out["placebo_confirms"] = flags
    return out


def build_full_placebo_trades(series, combo, cost_bps=10.0):
    """Framing 2: the FULL (unsampled) placebo population -- every raw
    candidate day build_placebo_candidates would accept, not Step C's
    per-ticker-count-matched subsample -- with causal exits, each tagged
    with whether the real technical-level condition ALSO holds on its entry
    date. Step C's subsample exists to make an apples-to-apples aggregate
    comparison against the real trade count; using it here would silently
    understate how many placebo-only trades exist and bias the
    BOTH-vs-PLACEBO-ONLY split.
    """
    rows = []
    for t, close_full in series.items():
        for i in build_placebo_candidates(close_full, combo):
            entry_date = close_full.index[i]
            entry_price = float(close_full.iloc[i])
            try:
                real_ok, resistance = causal_entry_and_resistance(close_full, entry_date, entry_price, combo)
            except Exception:
                real_ok, resistance = False, None
            try:
                exit_info = causal_exit(close_full, entry_date, entry_price, resistance, HOLD_DAYS)
            except Exception:
                exit_info = None
            if exit_info is None:
                continue
            rows.append({
                "stock": t, "entry_date": entry_date, "entry_price": entry_price,
                "exit_date": exit_info["exit_date"], "exit_price": exit_info["exit_price"],
                "exit_reason": exit_info["exit_reason"], "ret": exit_info["ret"],
                "ret_net": exit_info["ret"] - cost_bps / 1e4,
                "real_confirms": real_ok,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ew = equal_weight_curve(series)
    return attach_benchmark(df, ew)


def report_bucket(label, df):
    stats = pooled(df)
    if stats["n"] < 3:
        print(f"  {label:12s} n={stats['n']:4d}  (too few trades for a t-stat)")
        return stats, None
    boot = block_bootstrap_ci(df)
    line = f"  {label:12s} n={stats['n']:4d}  mean_excess={stats['mean_excess']:+.4%}  t={stats['t']:.2f}"
    if boot is not None:
        line += f"  block-boot 95% CI=[{boot['ci_lo']:+.4%}, {boot['ci_hi']:+.4%}]  P(mean<=0)={boot['p_le_0']:.1%}"
    print(line)
    return stats, boot


def run_universe(which):
    print(f"\n########## Confluence check: {which} ##########")
    combo = load_fixed_combo()
    series = load_universe(which)
    causal_path = CAUSAL_FULL_PATH.format(which=which)
    if not os.path.exists(causal_path):
        print(f"skipping {which}: {causal_path} not found -- run tech_level_causal_check.py first")
        return None

    causal_df = pd.read_csv(causal_path, parse_dates=["entry_date", "exit_date"])
    causal_df = causal_df[causal_df.stock.isin(series.keys())]

    print(f"\n--- Framing 1: real trades, split by whether a bare local low also confirms ---")
    labeled_real = label_real_trades_by_confluence(series, causal_df, combo)
    both_from_real = labeled_real[labeled_real.placebo_confirms]
    real_only = labeled_real[~labeled_real.placebo_confirms]
    print(f"{len(both_from_real)}/{len(labeled_real)} ({len(both_from_real) / len(labeled_real):.1%}) "
          f"of real trades are also confirmed by an independent local low")
    report_bucket("BOTH", both_from_real)
    report_bucket("REAL-ONLY", real_only)

    print(f"\n--- Framing 2: full placebo population, split by whether a real level also confirms ---")
    full_placebo = build_full_placebo_trades(series, combo)
    if full_placebo.empty:
        print("no placebo candidates generated")
        both_from_placebo = placebo_only = full_placebo
    else:
        both_from_placebo = full_placebo[full_placebo.real_confirms]
        placebo_only = full_placebo[~full_placebo.real_confirms]
        print(f"{len(both_from_placebo)}/{len(full_placebo)} ({len(both_from_placebo) / len(full_placebo):.1%}) "
              f"of the full placebo population is also confirmed by a real technical level")
        report_bucket("BOTH", both_from_placebo)
        report_bucket("PLACEBO-ONLY", placebo_only)

    print(f"\n--- reference: unsplit populations (for context) ---")
    report_bucket("ALL REAL", causal_df)
    if not full_placebo.empty:
        report_bucket("ALL PLACEBO", full_placebo)

    labeled_real.to_csv(os.path.join(DATA_DIR, f"confluence_real_labeled_{which}.csv"), index=False)
    if not full_placebo.empty:
        full_placebo.to_csv(os.path.join(DATA_DIR, f"confluence_full_placebo_{which}.csv"), index=False)

    return {
        "which": which, "both_from_real": both_from_real, "real_only": real_only,
        "both_from_placebo": both_from_placebo, "placebo_only": placebo_only,
        "all_real": causal_df, "all_placebo": full_placebo,
    }


def combined_report(results):
    print("\n########## Combined universe (ticker_list + oos) ##########")
    keys = ["both_from_real", "real_only", "both_from_placebo", "placebo_only", "all_real", "all_placebo"]
    combined = {}
    for k in keys:
        parts = [r[k] for r in results if r is not None and not r[k].empty]
        combined[k] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    print("\nFraming 1 (real trades):")
    report_bucket("BOTH", combined["both_from_real"])
    report_bucket("REAL-ONLY", combined["real_only"])
    print("\nFraming 2 (full placebo population):")
    report_bucket("BOTH", combined["both_from_placebo"])
    report_bucket("PLACEBO-ONLY", combined["placebo_only"])
    print("\nReference (unsplit):")
    report_bucket("ALL REAL", combined["all_real"])
    report_bucket("ALL PLACEBO", combined["all_placebo"])

    combined["both_from_placebo"].to_csv(os.path.join(DATA_DIR, "confluence_both_combined.csv"), index=False)
    return combined


def main():
    results = [run_universe(which) for which in ("ticker_list", "oos")]
    results = [r for r in results if r is not None]
    if len(results) == 2:
        combined_report(results)


if __name__ == "__main__":
    main()
