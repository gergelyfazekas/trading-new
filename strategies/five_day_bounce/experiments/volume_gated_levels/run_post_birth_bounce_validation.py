"""Two validation checks on run_post_birth_bounce.py's finding (a SMALLER
initial bounce right after birth/trough -- bounce_1d and bounce_to_entry_high
-- predicts a BETTER subsequent trade, replicated on both ticker_list and oos
after fixing the lookahead bug in the first pass):

  1. Does it survive against the bare-local-low population? Unlike every
     other feature in this folder, the bare-local-low population's trades
     (tech_level_confluence_check.build_full_placebo_trades) don't carry a
     "birth/trough date" field at all -- build_placebo_candidates only
     returns qualifying entry-day positions, not which trough backed them.
     Extended here with build_placebo_candidates_with_trough /
     build_full_placebo_trades_with_trough (standalone copies, same pattern
     as every other "don't touch the live/validated files" extension in this
     folder) to recover that trough date so bounce_1d/bounce_to_entry_high
     can be computed the same way as on the two-touch population.
  2. Does a bounce_1d<median gate on the two-touch population clear the
     DHR/MMM-style multi-period stability bar (worst-case delta across 4
     independent periods must be positive), the same test market_depth_10d
     and rel_mom_5d were held to?

Threshold: each universe's own median of bounce_1d (an unsupervised split on
the covariate's own distribution, not fitted to the outcome) -- there's no
natural zero-point for "size of a post-trough bounce" the way there was for
"volume ratio >= 1x" or "underperformed the market at all".

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_post_birth_bounce_validation.py
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
from tech_level_naive_strategy import load_fixed_combo, equal_weight_curve, attach_benchmark, summarise
from tech_level_continuation_live import MAX_AGE_DAYS
from tech_level_causal_check import placebo_trough_confirmations, causal_entry_and_resistance, causal_exit
from tech_level_causal_check import load_universe as load_causal_universe
from experiment_lib import build_young_level_trades, placebo_trades
from run_momentum_mood import HOLD_DAYS, NEAR_PCT
from run_oos_validation import load_oos_series
from run_post_birth_bounce import bounce_after_birth, bounce_to_entry_high

OUT_DIR = os.path.join(_HERE, "data")
N_PERIODS = 4


# ---------------------------------------------------------------------------
# Part 1: bare-local-low population, with trough date recovered
# ---------------------------------------------------------------------------

def placebo_raw_days_with_trough(close, trough_idx, confirm_pos, near_pct=NEAR_PCT, age_cutoff_days=MAX_AGE_DAYS):
    """Same day-level logic as tech_level_causal_check.placebo_raw_days_from_confirmations,
    but returns {day_position: trough_position} instead of just a set of
    qualifying days -- needed to know which trough backed a given bare-local-low
    entry, which build_placebo_candidates otherwise discards.
    """
    idx = close.index
    n = len(idx)
    if len(trough_idx) == 0:
        return {}
    qualifying = {}
    ti = 0
    for i in range(n):
        while ti < len(trough_idx) - 1 and trough_idx[ti + 1] <= i:
            ti += 1
        t_pos = trough_idx[ti]
        if t_pos > i:
            continue
        cpos = confirm_pos.get(t_pos)
        if cpos is None or cpos > i:
            continue
        age_days = (idx[i] - idx[t_pos]).days
        if age_days >= age_cutoff_days:
            continue
        price = close.iloc[i]
        trough_price = close.iloc[t_pos]
        dist = (price - trough_price) / price
        if 0 <= dist <= near_pct:
            qualifying[i] = t_pos
    return qualifying


def build_placebo_candidates_with_trough(close, combo, near_pct=NEAR_PCT, age_cutoff_days=MAX_AGE_DAYS,
                                           max_check_ahead=15, hold_days=HOLD_DAYS):
    trough_idx, confirm_pos = placebo_trough_confirmations(close, combo, max_check_ahead)
    qualifying = placebo_raw_days_with_trough(close, trough_idx, confirm_pos, near_pct, age_cutoff_days)
    n = len(close)
    candidates = []  # [(entry_pos, trough_pos), ...]
    in_position_until = -1
    for i in sorted(qualifying):
        if i <= in_position_until:
            continue
        candidates.append((i, qualifying[i]))
        in_position_until = min(i + hold_days, n - 1)
    return candidates


def build_full_placebo_trades_with_trough(series, combo, cost_bps=10.0):
    rows = []
    for t, close_full in series.items():
        for i, t_pos in build_placebo_candidates_with_trough(close_full, combo):
            entry_date = close_full.index[i]
            entry_price = float(close_full.iloc[i])
            trough_date = close_full.index[t_pos]
            try:
                real_ok, resistance = causal_entry_and_resistance(close_full, entry_date, entry_price, combo)
            except Exception:
                resistance = None
            try:
                exit_info = causal_exit(close_full, entry_date, entry_price, resistance, HOLD_DAYS)
            except Exception:
                exit_info = None
            if exit_info is None:
                continue
            rows.append({
                "stock": t, "entry_date": entry_date, "entry_price": entry_price,
                "trough_date": trough_date,
                "exit_date": exit_info["exit_date"], "exit_price": exit_info["exit_price"],
                "exit_reason": exit_info["exit_reason"], "ret": exit_info["ret"],
                "ret_net": exit_info["ret"] - cost_bps / 1e4,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ew = equal_weight_curve(series)
    return attach_benchmark(df, ew)


def check_bare_local_low(which, series, combo):
    print(f"\n{'='*70}\nbare-local-low / {which}: does bounce_1d / bounce_to_entry_high survive?\n{'='*70}")
    bare_df = build_full_placebo_trades_with_trough(series, combo)
    if bare_df.empty:
        print("no bare-local-low trades -- skipping")
        return None
    print(f"{len(bare_df)} bare-local-low trades")

    bare_df["bounce_1d"] = [
        bounce_after_birth(series[r.stock], r.trough_date, r.entry_date, 1) for r in bare_df.itertuples()
    ]
    bare_df["bounce_to_entry_high"] = [
        bounce_to_entry_high(series[r.stock], r.trough_date, r.entry_date) for r in bare_df.itertuples()
    ]

    bonferroni_bar = 0.05 / 2
    rows = []
    for col in ["bounce_1d", "bounce_to_entry_high"]:
        sub = bare_df.dropna(subset=[col])
        rho, p = stats.spearmanr(sub[col], sub.excess_ret)
        rows.append({"feature": col, "n": len(sub), "spearman_rho": rho, "p_value": p})
        print(f"{col}: n={len(sub)}  rho={rho:+.3f}  p={p:.4f}  "
              f"{'SURVIVES' if p < bonferroni_bar else 'does not survive'} Bonferroni (p<{bonferroni_bar})")

    bare_df.to_csv(os.path.join(OUT_DIR, f"post_birth_bounce_bare_local_low_{which}_trades.csv"), index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2: multi-period stability of a bounce_1d<median gate, two-touch population
# ---------------------------------------------------------------------------

def lift_vs_placebo(label, real_df, close_series, ew):
    real_counts = real_df.groupby("stock").size()
    placebo_df = placebo_trades(close_series, real_counts, hold_days=HOLD_DAYS)
    placebo_df = attach_benchmark(placebo_df, ew)
    a, b = real_df.excess_ret.dropna(), placebo_df.excess_ret.dropna()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 2 and len(b) > 2 else float("nan")
    t_lift = (a.mean() - b.mean()) / se if se and se > 0 else float("nan")
    print(f"{label}: real={a.mean():.4%}  placebo={b.mean():.4%}  lift={a.mean() - b.mean():+.4%}  t={t_lift:.2f}")


def check_stability(which, trades_df, close_series, ew):
    print(f"\n{'='*70}\ntwo-touch / {which}: bounce_1d<median stacking + stability\n{'='*70}")
    trades_df = trades_df.dropna(subset=["bounce_1d"]).copy()
    threshold = trades_df.bounce_1d.median()
    print(f"median bounce_1d = {threshold:.4%} (this universe's own split point)")

    filtered = trades_df[trades_df.bounce_1d < threshold]
    complement = trades_df[trades_df.bounce_1d >= threshold]
    print(f"unfiltered (n={len(trades_df)}): mean_excess={trades_df.excess_ret.mean():.4%}")
    print(f"filtered   (n={len(filtered)}): mean_excess={filtered.excess_ret.mean():.4%}")
    print(f"complement (n={len(complement)}): mean_excess={complement.excess_ret.mean():.4%}")

    by_stock = summarise(filtered, by="stock")
    n_positive = (by_stock.mean_excess > 0).sum()
    print(f"filtered arm by stock: {n_positive}/{len(by_stock)} individually positive on excess")

    lift_vs_placebo("filtered", filtered, close_series, ew)

    trades_df["period"] = pd.qcut(trades_df.entry_date, N_PERIODS, duplicates="drop")
    rows = []
    for period, g in trades_df.groupby("period", observed=True):
        g_filt = g[g.bounce_1d < threshold]
        unf_excess = g.excess_ret.mean()
        filt_excess = g_filt.excess_ret.mean() if len(g_filt) else float("nan")
        rows.append({
            "period": str(period), "n_unfiltered": len(g), "n_filtered": len(g_filt),
            "unfiltered_excess": unf_excess, "filtered_excess": filt_excess,
            "delta": filt_excess - unf_excess if pd.notna(filt_excess) else float("nan"),
        })
    period_df = pd.DataFrame(rows)
    print(period_df.to_string(index=False))
    worst = period_df.delta.min()
    n_pos = (period_df.delta > 0).sum()
    print(f"worst-case delta: {worst:+.4%}  ({n_pos}/{len(period_df)} periods positive)")
    if worst > 0:
        print("PASSES the worst-case bar.")
    else:
        print("FAILS the worst-case bar.")

    period_df.to_csv(os.path.join(OUT_DIR, f"post_birth_bounce_stability_{which}.csv"), index=False)
    return period_df


def main():
    combo = load_fixed_combo()
    os.makedirs(OUT_DIR, exist_ok=True)

    for which in ("ticker_list", "oos"):
        print(f"\n\n########## universe: {which} (loaded once, reused for both checks) ##########")
        close_series = load_causal_universe(which)
        print(f"{len(close_series)} tickers loaded")

        # Part 1: bare-local-low check
        check_bare_local_low(which, close_series, combo)

        # Part 2: two-touch stability check -- reuse the SAME close_series.
        # build_young_level_trades unconditionally calls volume.copy(), even
        # though consider_volume=False means find_touches never reads it --
        # pass a dummy zero series rather than None to satisfy that.
        tickers = list(close_series.keys())
        series_cv = {t: (close_series[t], pd.Series(0, index=close_series[t].index)) for t in tickers}
        trades_df, cs, _v, ew, failed = build_young_level_trades(
            tickers, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS, series=series_cv
        )
        trades_df["bounce_1d"] = [
            bounce_after_birth(cs[r.stock], r.support_birth_date, r.entry_date, 1) for r in trades_df.itertuples()
        ]
        check_stability(which, trades_df, cs, ew)


if __name__ == "__main__":
    main()
