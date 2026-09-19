"""Parameter-neighborhood and cost sensitivity checks for the bare-local-low
placebo rule (tech_level_causal_check.py, Step C), matching the scrutiny
tech_level_sensitivity.py already gave the real technical-level rule --
except this one is causal from the start, not retrofitted later.

WHY THIS FILE EXISTS, AND THE LOOKAHEAD RISK IT WAS BUILT TO CATCH
-------------------------------------------------------------------
tech_level_sensitivity.py's near_pct x hold_days grid for the *real* rule
was built on tech_level_naive_strategy.simulate() -- the whole-history,
non-causal backtest. It was never causally rechecked cell-by-cell; only the
single (near_pct=1%, hold_days=5) live cell got that treatment later, in
tech_level_causal_check.py's Step A. This file holds the placebo grid to a
higher bar: every cell here is causal from the start.

That surfaced a real, specific risk on the exit side, caught before running
the grid rather than after: causal_entry_and_resistance/causal_exit
(tech_level_causal_check.py) freeze the resistance level at entry and never
rebuild it during the hold. Their own module docstring calls this "provably
exact, not approximate" -- but only because hold_days=5 is strictly less
than the combo's distance=10-session confirmation window, so no touch
unconfirmed at entry can become confirmable before the hold ends. A
parameter sweep that reaches hold_days=10 (matching tech_level_sensitivity
.py's original grid) or that sweeps distance down to 3-5 breaks that
guarantee outright. Reusing causal_exit unmodified there would silently
launder the gap back in.

Fix: causal_exit_full() rebuilds the resistance target from data truncated
to *that day* on every day of the hold, never frozen at entry.

TWO THINGS THIS CAUGHT, NEITHER OF WHICH IS A LOOKAHEAD LEAK
--------------------------------------------------------------
1. Validating causal_exit_full against the frozen causal_exit at hold_days=5
   (where the "exact" proof is supposed to hold) found a real, ~5%
   disagreement rate -- NOT because either function sees data it shouldn't,
   but because of a mechanism the original "exact" proof never considered:
   tech_levels.mark_broken() can retire a resistance level *during* the
   hold (price fully crosses it) even when no *new* level gets confirmed.
   The frozen shortcut keeps checking the entry-day resistance object
   forever, blind to it having since become broken; causal_exit_full
   correctly drops it and looks for the next-nearest still-valid one. Both
   versions only ever use data available as of the day being decided --
   this is a small fidelity gap in the frozen shortcut, not a leak in
   either direction. Documented, not silently patched into the "exact"
   claim, since it also quietly affects the already-published real-rule
   Step A numbers at hold_days=5, not just this file's placebo work.
2. A full day-by-day rebuild is expensive -- a single (near_pct=3%,
   hold_days=10) cell on the 41-name ticker_list universe alone did not
   finish in 8+ minutes before being killed. Running that cost across a
   full grid would take hours. Fixed by (a) restricting exactly which
   cells need the expensive treatment (see SAFE_HOLD_DAYS / cells needing
   causal_exit_full below -- most of the grid provably doesn't), and (b)
   capping the number of candidates any single (ticker, cell) sends through
   the expensive path via FULL_EXIT_SAMPLE_CAP, sampled with a fixed seed
   and reported as what it is (a bounded sample, not the exhaustive
   population) rather than silently truncating without saying so.

WHICH CELLS NEED causal_exit_full, AND WHY
--------------------------------------------
The "exact" shortcut only holds when hold_days < distance:
  - main near_pct x hold_days grid: hold_days in {1,2,3,5,7}, distance=10
    fixed -> all safe, cheap causal_exit used (with the ~5% mark_broken
    caveat above, present at every hold_days, not just the boundary).
  - hold_days=10 (== distance, the guarantee's edge): computed both ways
    at the base near_pct=1% cell only (not the full near_pct sweep -- the
    point is to measure whether the *new-confirmation* risk the user
    flagged is material, not to re-run every combination twice), divergence
    reported directly rather than assumed negligible.
  - distance sensitivity (near_pct=1%, hold_days=5 fixed): distance in
    {3, 5} violate hold_days < distance and get causal_exit_full;
    distance in {10, 15, 20} satisfy it and use the cheap exit.

Entry-side note: the trough-confirmation check itself (placebo_raw_days /
_trough_confirmed_by) was already fixed to be genuinely causal on
2026-09-15 -- it truncates-and-reruns find_peaks rather than assuming a
fixed confirmation lag -- so it doesn't need revisiting here.

Run (from repo root): ./venv/bin/python strategies/five_day_bounce/tech_level_placebo_sensitivity.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tech_level_naive_strategy import load_fixed_combo, build_levels, equal_weight_curve, attach_benchmark
from tech_level_causal_check import (
    load_universe, pooled, causal_entry_and_resistance, causal_exit,
    build_placebo_candidates, placebo_trough_confirmations, DATA_DIR, NEAR_PCT, AGE_CUTOFF_DAYS,
)

NEAR_PCTS = (0.005, 0.01, 0.015, 0.02, 0.03)
HOLD_DAYS_SAFE = (1, 2, 3, 5, 7)   # all strictly < the combo's distance=10
HOLD_DAYS_BOUNDARY = 10             # == distance -- the frozen-exit guarantee's edge
DISTANCES_UNSAFE = (3, 5)           # hold_days=5 not strictly < these -- need full exit
DISTANCES_SAFE = (10, 15, 20)       # hold_days=5 < these -- cheap exit is exact
COST_GRID_BPS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
FULL_EXIT_SAMPLE_CAP = 25  # per (ticker, cell) -- bounds worst-case cost of causal_exit_full


# ---------------------------------------------------------------------------
# The fully causal exit -- rebuilds the resistance target from truncated
# data on every day of the hold, instead of freezing it at entry.
# ---------------------------------------------------------------------------

def causal_exit_full(close_full, entry_date, entry_price, combo, hold_days):
    """Day-by-day causal exit: on each day j of the hold, truncate to j,
    rebuild levels, and take the nearest not-yet-broken level confirmed as of
    j whose band sits above entry_price (the reference point defining
    "resistance for this trade" is where we bought, same convention
    active_support_resistance uses at entry -- just re-derived with each
    day's fuller information instead of frozen once). Exit if today's price
    falls inside that band.
    """
    idx = close_full.index
    entry_pos = idx.get_loc(pd.Timestamp(entry_date))
    n = len(idx)
    if entry_pos + hold_days >= n:
        return None

    exit_pos = min(entry_pos + hold_days, n - 1)
    exit_reason = "hold_days"
    for j in range(entry_pos + 1, min(entry_pos + hold_days, n - 1) + 1):
        as_of_j = idx[j]
        price_j = float(close_full.iloc[j])
        trunc = close_full.loc[:as_of_j]
        levels_j = build_levels(trunc, combo)
        candidates = [
            lvl for lvl in levels_j
            if pd.Timestamp(lvl.birth_date) <= as_of_j
            and (lvl.broken_at is None or as_of_j < pd.Timestamp(lvl.broken_at))
            and lvl.band[0] > entry_price
        ]
        resistance_j = min(candidates, key=lambda lvl: lvl.band[0]) if candidates else None
        if resistance_j is not None and resistance_j.band[0] <= price_j <= resistance_j.band[1]:
            exit_pos = j
            exit_reason = "resistance"
            break

    exit_date = idx[exit_pos]
    exit_price = float(close_full.iloc[exit_pos])
    return {"exit_date": exit_date, "exit_price": exit_price, "exit_reason": exit_reason,
            "ret": exit_price / entry_price - 1.0}


def validate_full_exit_matches_frozen(series, combo, hold_days_list=(5, 7), sample_per_ticker=6, seed=0):
    """At hold_days values where causal_exit's frozen shortcut is claimed
    exact (< distance=10), check how often causal_exit_full agrees. This is
    a sanity check on causal_exit_full's own correctness (they operate on
    the same truncated-data discipline, so wholesale disagreement would mean
    a bug in the new function) -- but any *remaining* disagreement after
    ruling that out points at the mark_broken timing gap documented in the
    module docstring, not at a leak in either function.
    """
    rng = np.random.default_rng(seed)
    print(f"\n--- validating causal_exit_full against the frozen causal_exit (hold_days={hold_days_list}) ---")
    for hold_days in hold_days_list:
        checked = agree = 0
        mismatches = []
        for t, close_full in series.items():
            candidates = build_placebo_candidates(close_full, combo, near_pct=NEAR_PCT, hold_days=hold_days)
            if not candidates:
                continue
            sample = rng.choice(candidates, size=min(sample_per_ticker, len(candidates)), replace=False)
            for i in sample:
                entry_date = close_full.index[i]
                entry_price = float(close_full.iloc[i])
                try:
                    _, resistance = causal_entry_and_resistance(close_full, entry_date, entry_price, combo)
                    frozen = causal_exit(close_full, entry_date, entry_price, resistance, hold_days)
                    full = causal_exit_full(close_full, entry_date, entry_price, combo, hold_days)
                except Exception:
                    continue
                if frozen is None or full is None:
                    continue
                checked += 1
                if (frozen["exit_date"] == full["exit_date"] and frozen["exit_reason"] == full["exit_reason"]
                        and abs(frozen["ret"] - full["ret"]) < 1e-9):
                    agree += 1
                else:
                    mismatches.append((t, str(entry_date.date()), frozen["exit_reason"], full["exit_reason"]))
        rate = agree / checked if checked else float("nan")
        print(f"  hold_days={hold_days}: {agree}/{checked} sampled trades agree exactly ({rate:.1%})")
        if mismatches:
            print(f"  sample mismatches (ticker, entry_date, frozen_reason, full_reason): {mismatches[:5]}")


# ---------------------------------------------------------------------------
# Trade construction for one (near_pct, hold_days) cell
# ---------------------------------------------------------------------------

def build_cell_trades(series, combo, near_pct, hold_days, use_full_exit, cost_bps=10.0,
                       sample_cap=None, seed=0, confirmations_by_ticker=None):
    """sample_cap, when use_full_exit=True, bounds how many of a ticker's
    candidates get the expensive per-day rebuild -- a fixed-seed random
    subsample, reported via the returned df's row count (never silently
    treated as exhaustive). confirmations_by_ticker, when given, is a
    {ticker: (trough_idx, confirm_pos)} map from placebo_trough_confirmations,
    precomputed once per ticker outside any near_pct/hold_days loop -- that
    walk doesn't depend on either, so recomputing it on every grid cell (the
    original, naive version of this function) made a 25-cell grid ~25x
    slower than necessary; this is the fix.
    """
    rng = np.random.default_rng(seed) if sample_cap else None
    rows = []
    for t, close_full in series.items():
        confirmations = confirmations_by_ticker.get(t) if confirmations_by_ticker else None
        candidates = build_placebo_candidates(close_full, combo, near_pct=near_pct, hold_days=hold_days,
                                               confirmations=confirmations)
        if use_full_exit and sample_cap and len(candidates) > sample_cap:
            candidates = sorted(rng.choice(candidates, size=sample_cap, replace=False).tolist())
        for i in candidates:
            entry_date = close_full.index[i]
            entry_price = float(close_full.iloc[i])
            try:
                if use_full_exit:
                    exit_info = causal_exit_full(close_full, entry_date, entry_price, combo, hold_days)
                else:
                    _, resistance = causal_entry_and_resistance(close_full, entry_date, entry_price, combo)
                    exit_info = causal_exit(close_full, entry_date, entry_price, resistance, hold_days)
            except Exception:
                exit_info = None
            if exit_info is None:
                continue
            rows.append({
                "stock": t, "near_pct": near_pct, "hold_days": hold_days,
                "entry_date": entry_date, "entry_price": entry_price,
                "exit_date": exit_info["exit_date"], "exit_reason": exit_info["exit_reason"],
                "ret": exit_info["ret"], "ret_net": exit_info["ret"] - cost_bps / 1e4,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ew = equal_weight_curve(series)
    return attach_benchmark(df, ew)


# ---------------------------------------------------------------------------
# Grid, boundary check, distance sweep, cost sweep
# ---------------------------------------------------------------------------

def parameter_grid(which, series, combo):
    print(f"\n########## Placebo parameter-neighborhood grid: {which} ##########")
    print("(hold_days in {1,2,3,5,7}: cheap frozen exit -- exact modulo the ~5% "
          "mark_broken-timing gap documented above, not a lookahead issue)")

    print("precomputing trough confirmations once per ticker (near_pct/hold_days-independent)...")
    confirmations_by_ticker = {t: placebo_trough_confirmations(c, combo) for t, c in series.items()}

    rows = []
    base_cell_df = None
    for hold_days in HOLD_DAYS_SAFE:
        for near_pct in NEAR_PCTS:
            df = build_cell_trades(series, combo, near_pct, hold_days, use_full_exit=False,
                                    confirmations_by_ticker=confirmations_by_ticker)
            stats = pooled(df)
            rows.append({"near_pct": near_pct, "hold_days": hold_days, **stats})
            if near_pct == NEAR_PCT and hold_days == 5:
                base_cell_df = df

    grid = pd.DataFrame(rows)
    table = grid.pivot(index="near_pct", columns="hold_days", values="mean_excess")
    tstat = grid.pivot(index="near_pct", columns="hold_days", values="t")
    print("\nmean excess % (t-stat):")
    for near_pct in NEAR_PCTS:
        cells = "  ".join(f"{table.loc[near_pct, h]*100:+.2f} ({tstat.loc[near_pct, h]:.1f})"
                           for h in HOLD_DAYS_SAFE)
        print(f"  near_pct={near_pct:.1%}: {cells}")

    print(f"\n--- hold_days={HOLD_DAYS_BOUNDARY} (== distance) at base near_pct={NEAR_PCT:.0%}: "
          f"frozen vs. fully-causal (sample-capped at {FULL_EXIT_SAMPLE_CAP}/ticker), not assumed safe ---")
    frozen_df = build_cell_trades(series, combo, NEAR_PCT, HOLD_DAYS_BOUNDARY, use_full_exit=False,
                                   confirmations_by_ticker=confirmations_by_ticker)
    full_df = build_cell_trades(series, combo, NEAR_PCT, HOLD_DAYS_BOUNDARY, use_full_exit=True,
                                 sample_cap=FULL_EXIT_SAMPLE_CAP, confirmations_by_ticker=confirmations_by_ticker)
    f_stats, u_stats = pooled(frozen_df), pooled(full_df)
    print(f"  frozen (full population): n={f_stats['n']:5d}  mean_excess={f_stats['mean_excess']:+.4%}  "
          f"t={f_stats['t']:.2f}")
    print(f"  fully-causal (capped sample): n={u_stats['n']:5d}  mean_excess={u_stats['mean_excess']:+.4%}  "
          f"t={u_stats['t']:.2f}")
    exit_reason_diff = np.nan
    if not frozen_df.empty and not full_df.empty:
        merged = frozen_df[["stock", "entry_date", "exit_reason"]].merge(
            full_df[["stock", "entry_date", "exit_reason"]], on=["stock", "entry_date"],
            suffixes=("_frozen", "_full"))
        if len(merged):
            exit_reason_diff = (merged.exit_reason_frozen != merged.exit_reason_full).mean()
            print(f"  of the {len(merged)} trades present in both (the capped sample's entries): "
                  f"exit reason differs in {exit_reason_diff:.1%}")

    grid.to_csv(os.path.join(DATA_DIR, f"placebo_sensitivity_grid_{which}.csv"), index=False)
    boundary_out = pd.DataFrame([{"near_pct": NEAR_PCT, **{f"frozen_{k}": v for k, v in f_stats.items()},
                                   **{f"full_{k}": v for k, v in u_stats.items()},
                                   "exit_reason_diff": exit_reason_diff}])
    boundary_out.to_csv(os.path.join(DATA_DIR, f"placebo_sensitivity_boundary_{which}.csv"), index=False)
    return grid, base_cell_df


def distance_sensitivity(which, series):
    print(f"\n########## Placebo distance sensitivity: {which} (near_pct=1%, hold_days=5) ##########")
    print(f"distance in {DISTANCES_UNSAFE}: hold_days=5 not < distance -- fully-causal exit "
          f"(sample-capped at {FULL_EXIT_SAMPLE_CAP}/ticker); distance in {DISTANCES_SAFE}: "
          f"hold_days=5 < distance -- cheap frozen exit is exact (modulo the mark_broken caveat)")
    base_combo = load_fixed_combo()
    rows = []
    for d in DISTANCES_UNSAFE + DISTANCES_SAFE:
        combo = dict(base_combo)
        combo["distance"] = d
        use_full = d in DISTANCES_UNSAFE
        df = build_cell_trades(series, combo, NEAR_PCT, 5, use_full_exit=use_full,
                                sample_cap=FULL_EXIT_SAMPLE_CAP if use_full else None)
        stats = pooled(df)
        tag = "fully-causal, capped" if use_full else "frozen exit"
        print(f"  distance={d:3d} ({tag:20s})  n={stats['n']:5d}  mean_excess={stats['mean_excess']:+.4%}  "
              f"t={stats['t']:.2f}")
        rows.append({"distance": d, "method": tag, **stats})
    out = pd.DataFrame(rows).sort_values("distance")
    out.to_csv(os.path.join(DATA_DIR, f"placebo_distance_sensitivity_{which}.csv"), index=False)
    return out


def cost_sensitivity(which, base_cell_df):
    print(f"\n########## Placebo cost sensitivity: {which} (near_pct=1%, hold_days=5) ##########")
    if base_cell_df is None or base_cell_df.empty:
        print("  no base-cell trades available")
        return None
    ew_bench = base_cell_df["bench_ret"]
    rows = []
    for cost_bps in COST_GRID_BPS:
        ret_net = base_cell_df["ret"] - cost_bps / 1e4
        excess = ret_net - ew_bench
        n = len(excess)
        mean = excess.mean()
        t = mean / excess.std() * np.sqrt(n) if n > 2 else np.nan
        print(f"  cost={cost_bps:5.1f}bp  mean_excess={mean:+.4%}  t={t:.2f}")
        rows.append({"cost_bps": cost_bps, "n": n, "mean_excess": mean, "t": t})
    out = pd.DataFrame(rows)
    above = out[out.mean_excess > 0]
    below = out[out.mean_excess <= 0]
    if not above.empty and not below.empty:
        c0, m0 = above.iloc[-1][["cost_bps", "mean_excess"]]
        c1, m1 = below.iloc[0][["cost_bps", "mean_excess"]]
        breakeven = c0 + (0 - m0) * (c1 - c0) / (m1 - m0) if m1 != m0 else np.nan
        print(f"  approx. breakeven cost: ~{breakeven:.0f}bp")
    out.to_csv(os.path.join(DATA_DIR, f"placebo_cost_sensitivity_{which}.csv"), index=False)
    return out


def main():
    combo = load_fixed_combo()
    for which in ("ticker_list", "oos"):
        series = load_universe(which)
        print(f"\n{len(series)} tickers loaded for {which}")

        validate_full_exit_matches_frozen(series, combo)
        grid, base_cell_df = parameter_grid(which, series, combo)
        cost_sensitivity(which, base_cell_df)
        distance_sensitivity(which, series)


if __name__ == "__main__":
    main()
