"""Fully causal (entry *and* exit) recheck of the "provisional levels" caveat,
plus the three follow-ups its own 2026-09-15 writeup flagged as not yet done:
non-i.i.d.-aware inference, a level-vs-generic-reversal placebo, and a
concurrency-capped portfolio simulation. See tech_levels_notes.md for the
full history and the plan this file implements.

The original problem (unchanged from the first version of this file): the
naive-strategy backtest runs find_touches once over each ticker's *whole*
history, so a touch born <5 calendar days before a trade may have only been
confirmed as a real peak/trough because the detector could see up to
`distance`=10 sessions of price action *after* that date -- data a live
trader at entry time would not have had yet.

Step A below closes a gap the original version left open: it rechecked the
*entry* side only (does the support still qualify at truncated data) but
left exit_date/exit_price/ret coming straight from the hindsight trade list
-- a resistance band needs exactly the same distance=10 confirmation as a
support band, so a hindsight "day 2 of 5" exit could rest on a resistance
level the detector could not actually have confirmed by day 2. Fix: a
resistance level is only ever picked from levels built on data truncated to
the *entry* date (never rebuilt mid-hold) -- which is provably exact here,
not an approximation: distance=10 > hold_days=5, so no touch that wasn't
already confirmable at entry can become confirmable before the hold ends.
This also matches what the original (hindsight) simulate() already does --
it evaluates active_support_resistance() once at entry and never re-derives
resistance mid-hold either -- so Step A only changes *what data* that one
evaluation is allowed to see, not the rule being tested.

Five checks, run in this order (each depends on the previous):
  A. per-trade causal recheck of entry AND exit -- the direct test, see above.
  B. inference that matches the data's actual (non-i.i.d.) structure: a
     calendar-year block bootstrap CI + a cluster-robust (by-ticker) t-stat,
     replacing the plain pooled t-stat that assumes independent trades.
  C. a placebo that drops the "matches a two-touch technical level"
     requirement and keeps only "price is within near_pct of a recently
     confirmed local low" -- same exit mechanics, so the only thing being
     tested is whether the *level* concept adds anything over generic
     short-horizon reversal. Trough confirmation date is approximated as
     trough_date + distance trading sessions (a conservative, deliberately
     cheap bound -- see build_placebo_candidates docstring -- rather than a
     full per-date rebuild, since this check is diagnostic, not the number
     being sized).
  D. a concurrency-capped (n_slots) portfolio equity curve on Step A's
     surviving trades, since mean-excess-per-trade isn't what a trader with
     ~7-day holds and up to 24 concurrent signals actually experiences.
  (distance sensitivity, from the original version, kept as-is below the
  main() sequence -- cheap complementary check, not a substitute for A.)

Run (from repo root): ./venv/bin/python strategies/five_day_bounce/tech_level_causal_check.py
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import config
from stock_class import StockList
from tech_level_naive_strategy import (
    load_fixed_combo, build_levels, active_support_resistance,
    simulate, equal_weight_curve, attach_benchmark,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EXCLUDE_OOS = {"NVDA"}  # see tech_level_oos_strategy.py docstring
START_DATE = datetime.date(2015, 5, 28)
NEAR_PCT = 0.01
HOLD_DAYS = 5
AGE_CUTOFF_DAYS = 5


def load_universe(which):
    if which == "ticker_list":
        tickers = list(config.ticker_list)
        sl = StockList(tickers)
        sl.pull_data(start=START_DATE, end=datetime.date.today())
    else:
        tickers = [t for t in config.oos_ticker_list if t not in EXCLUDE_OOS]
        sl = StockList(tickers)
        sl.load_data(config.oos)

    series = {}
    for t in tickers:
        try:
            c = sl[t].data["close"].dropna()
            c.index = pd.to_datetime(c.index)
            if not c.empty:
                series[t] = c
        except Exception:
            continue
    return series


def pooled(trades_df):
    if trades_df.empty:
        return {"n": 0, "mean_excess": np.nan, "t": np.nan}
    n = len(trades_df)
    mean = trades_df.excess_ret.mean()
    t = mean / trades_df.excess_ret.std() * np.sqrt(n) if n > 2 else np.nan
    return {"n": n, "mean_excess": mean, "t": t}


# ---------------------------------------------------------------------------
# Step A: causal entry AND exit
# ---------------------------------------------------------------------------

def causal_entry_and_resistance(close_full, entry_date, entry_price, combo):
    """Rebuild levels from data truncated to entry_date only. Returns
    (entry_ok, resistance) -- resistance is the Level a live trader would be
    watching for the rest of the hold, or None. Shared by the entry check
    and by causal_exit() so the two can't see different data.
    """
    trunc = close_full.loc[:pd.Timestamp(entry_date)]
    levels = build_levels(trunc, combo)
    support, resistance = active_support_resistance(levels, pd.Timestamp(entry_date), entry_price)
    entry_ok = support is not None and 0 <= (entry_price - support.band[1]) / entry_price <= NEAR_PCT
    return entry_ok, resistance


def causal_exit(close_full, entry_date, entry_price, resistance, hold_days):
    """Walk forward from entry using only the entry-time resistance (see
    module docstring for why this is exact, not approximate, at hold_days=5
    < distance=10). Returns None if the trade would need truncation (same
    drop convention as simulate()).
    """
    idx = close_full.index
    entry_pos = idx.get_loc(pd.Timestamp(entry_date))
    n = len(idx)
    if entry_pos + hold_days >= n:
        return None

    exit_pos = min(entry_pos + hold_days, n - 1)
    exit_reason = "hold_days"
    for j in range(entry_pos + 1, min(entry_pos + hold_days, n - 1) + 1):
        price_j = close_full.iloc[j]
        if resistance is not None and resistance.band[0] <= price_j <= resistance.band[1]:
            exit_pos = j
            exit_reason = "resistance"
            break

    exit_date = idx[exit_pos]
    exit_price = float(close_full.iloc[exit_pos])
    return {
        "exit_date": exit_date, "exit_price": exit_price, "exit_reason": exit_reason,
        "ret": exit_price / entry_price - 1.0,
    }


def build_causal_trades(which, series, trades_path, cost_bps=10.0):
    """Step A: full entry+exit causal recheck of every logged young-level
    trade. Returns a DataFrame of only the trades that fire causally on
    entry (exit is always computed -- exit_reason/exit_date/ret can differ
    from the hindsight record even for a causally-confirmed entry, since the
    resistance a live trader would watch may not be the same one the
    whole-history build picked).
    """
    print(f"\n########## Step A: causal entry+exit recheck: {which} ##########")
    combo = load_fixed_combo()
    trades_df = pd.read_csv(trades_path, parse_dates=["entry_date", "exit_date"])
    young = trades_df[(trades_df.hold_days == HOLD_DAYS) & (trades_df.near_pct == NEAR_PCT) &
                       (trades_df.support_age_days < AGE_CUTOFF_DAYS)].copy()
    young = young[young.stock.isin(series.keys())]
    print(f"{len(young)} young-level (<{AGE_CUTOFF_DAYS}d) trades to recheck")

    entry_confirmed = []
    rows = []
    for row in young.itertuples():
        close_full = series[row.stock]
        try:
            ok, resistance = causal_entry_and_resistance(close_full, row.entry_date, row.entry_price, combo)
        except Exception:
            ok, resistance = False, None
        entry_confirmed.append(ok)
        if not ok:
            continue
        try:
            exit_info = causal_exit(close_full, row.entry_date, row.entry_price, resistance, HOLD_DAYS)
        except Exception:
            exit_info = None
        if exit_info is None:
            continue
        rows.append({
            "stock": row.stock, "hold_days": HOLD_DAYS, "near_pct": NEAR_PCT,
            "entry_date": row.entry_date, "entry_price": row.entry_price,
            "exit_date": exit_info["exit_date"], "exit_price": exit_info["exit_price"],
            "exit_reason": exit_info["exit_reason"],
            "ret": exit_info["ret"], "ret_net": exit_info["ret"] - cost_bps / 1e4,
            "support_age_days": row.support_age_days,
            "hindsight_exit_date": row.exit_date, "hindsight_exit_reason": row.exit_reason,
            "hindsight_ret": row.ret,
        })

    young["entry_causally_confirmed"] = entry_confirmed
    print(f"entry survival rate: {young.entry_causally_confirmed.mean():.1%} "
          f"({young.entry_causally_confirmed.sum()}/{len(young)})")

    causal_df = pd.DataFrame(rows)
    if causal_df.empty:
        print("no causally-confirmed trades with a valid exit")
        return causal_df

    ew = equal_weight_curve(series)
    causal_df = attach_benchmark(causal_df, ew)
    exit_changed = (causal_df.exit_date != causal_df.hindsight_exit_date).mean()
    print(f"of entry-confirmed trades with a valid exit: n={len(causal_df)}, "
          f"exit date differs from hindsight record in {exit_changed:.1%} of them")

    full_stats = pooled(attach_benchmark(young.copy(), ew))
    surv_stats = pooled(causal_df)
    print(f"full (hindsight) set        -- n={full_stats['n']:5d}  "
          f"mean_excess={full_stats['mean_excess']:+.4%}  t={full_stats['t']:.2f}")
    print(f"fully causal (entry+exit)   -- n={surv_stats['n']:5d}  "
          f"mean_excess={surv_stats['mean_excess']:+.4%}  t={surv_stats['t']:.2f}")

    causal_df.to_csv(os.path.join(DATA_DIR, f"causal_full_{which}.csv"), index=False)
    return causal_df


def concentration_report(which, causal_df):
    """Do the causally-confirmed survivors cluster in a few tickers/years, or
    spread broadly? A real, general effect should survive being broad; an
    effect propped up by a handful of names/periods is much weaker evidence.
    """
    surv = causal_df.copy()
    surv["year"] = pd.DatetimeIndex(surv.entry_date).year
    n = len(surv)
    print(f"\n--- concentration of causally-confirmed survivors: {which} (n={n}) ---")

    by_stock = surv.groupby("stock").agg(n=("excess_ret", "size"), total_excess=("excess_ret", "sum"))
    by_stock = by_stock.sort_values("total_excess", ascending=False)
    total_excess_all = by_stock["total_excess"].sum()
    by_stock["pct_of_trades"] = by_stock["n"] / n
    by_stock["pct_of_total_excess"] = by_stock["total_excess"] / total_excess_all
    print(f"\n{surv.stock.nunique()} distinct tickers among survivors")
    print("top 10 tickers by total excess return contributed:")
    print(by_stock.head(10).round(4).to_string())
    top5_trade_share = by_stock["n"].head(5).sum() / n
    top5_excess_share = by_stock["total_excess"].head(5).sum() / total_excess_all
    print(f"top 5 tickers: {top5_trade_share:.1%} of trades, {top5_excess_share:.1%} of total excess return")

    by_year = surv.groupby("year").agg(n=("excess_ret", "size"), mean_excess=("excess_ret", "mean"),
                                        total_excess=("excess_ret", "sum"))
    by_year["pct_of_trades"] = by_year["n"] / n
    print("\nby year:")
    print(by_year.round(4).to_string())


def distance_sensitivity(which, series, distances):
    print(f"\n########## distance sensitivity: {which} ##########")
    base_combo = load_fixed_combo()
    ew = equal_weight_curve(series)
    rows = []
    for d in distances:
        combo = dict(base_combo)
        combo["distance"] = d
        levels_by_ticker = {t: build_levels(c, combo) for t, c in series.items()}
        all_trades = []
        for t, c in series.items():
            all_trades.extend(simulate(t, c, levels_by_ticker[t], hold_days=HOLD_DAYS, near_pct=NEAR_PCT))
        trades_df = pd.DataFrame(all_trades)
        if trades_df.empty:
            rows.append({"distance": d, "n_all": 0, "n_young": 0, "mean_excess": np.nan, "t": np.nan})
            continue
        trades_df = attach_benchmark(trades_df, ew)
        young = trades_df[trades_df.support_age_days < AGE_CUTOFF_DAYS]
        stats = pooled(young)
        rows.append({"distance": d, "n_all": len(trades_df), "n_young": stats["n"],
                      "mean_excess": stats["mean_excess"], "t": stats["t"]})
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Step B: inference that matches the data's actual structure
# ---------------------------------------------------------------------------

def block_bootstrap_ci(trades_df, n_boot=2000, seed=0):
    """Resample whole calendar-year blocks (with replacement) so within-year
    correlation across overlapping/clustered trades survives into the
    resampled draws, instead of resampling individual trades as if they were
    independent. Reports the empirical 95% CI of the mean and the fraction
    of bootstrap draws with mean_excess <= 0 -- that fraction is the number
    to quote in place of a t-stat p-value.
    """
    df = trades_df.copy()
    df["year"] = pd.DatetimeIndex(df.entry_date).year
    years = df["year"].unique()
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sampled_years = rng.choice(years, size=len(years), replace=True)
        parts = [df[df.year == y] for y in sampled_years]
        boot = pd.concat(parts)
        if boot.empty:
            continue
        means.append(boot.excess_ret.mean())
    means = np.array(means)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": df.excess_ret.mean(), "ci_lo": lo, "ci_hi": hi,
        "p_le_0": float((means <= 0).mean()), "n_boot": len(means), "n_years": len(years),
    }


def cluster_robust_se(trades_df, cluster_col="stock"):
    """Cheap cluster-robust cross-check: collapse to one mean excess return
    per cluster (ticker), treat those cluster means as the observations, and
    scale by sqrt(n_clusters). Much coarser than a full Liang-Zeger sandwich
    estimator but directionally right and simple to sanity-check by eye --
    if this and the block bootstrap disagree by a lot, that disagreement is
    itself worth reporting rather than picking one.
    """
    g = trades_df.groupby(cluster_col)["excess_ret"]
    cluster_means = g.mean()
    n_clusters = len(cluster_means)
    grand_mean = trades_df.excess_ret.mean()
    se = cluster_means.std(ddof=1) / np.sqrt(n_clusters) if n_clusters > 2 else np.nan
    t = grand_mean / se if se and se > 0 else np.nan
    return {"n_clusters": n_clusters, "grand_mean": grand_mean, "cluster_se": se, "t_cluster": t}


def inference_report(which, causal_df):
    print(f"\n########## Step B: inference, {which} (n={len(causal_df)}) ##########")
    naive = pooled(causal_df)
    print(f"naive pooled (i.i.d. assumption)      -- mean={naive['mean_excess']:+.4%}  t={naive['t']:.2f}")

    boot = block_bootstrap_ci(causal_df)
    print(f"calendar-year block bootstrap         -- mean={boot['mean']:+.4%}  "
          f"95% CI=[{boot['ci_lo']:+.4%}, {boot['ci_hi']:+.4%}]  "
          f"P(mean<=0)={boot['p_le_0']:.1%}  ({boot['n_years']} year-blocks, {boot['n_boot']} draws)")

    clus = cluster_robust_se(causal_df)
    print(f"ticker-clustered SE                   -- mean={clus['grand_mean']:+.4%}  "
          f"se={clus['cluster_se']:.4%}  t={clus['t_cluster']:.2f}  ({clus['n_clusters']} clusters)")
    return {"naive": naive, "bootstrap": boot, "cluster": clus}


# ---------------------------------------------------------------------------
# Step C: placebo -- level vs. generic short-horizon reversal
# ---------------------------------------------------------------------------

def _trough_confirmed_by(log_close, distance, prominence, t_pos, as_of_pos):
    """Would find_peaks flag t_pos as a trough using only data up to and
    including as_of_pos (the real, non-approximated causal check)?"""
    trunc = log_close[: as_of_pos + 1]
    trough_idx, _ = find_peaks(-trunc, distance=distance, prominence=prominence)
    return t_pos in trough_idx


def placebo_trough_confirmations(close, combo, max_check_ahead=15):
    """The expensive half of placebo_raw_days -- find every trough and, for
    each, the day position it first becomes causally confirmable
    (truncate-and-rerun find_peaks, not a fixed lag). Depends only on
    (close, combo's distance/prominence) -- NOT on near_pct, age_cutoff_days,
    or hold_days -- so a caller sweeping those (a parameter grid) should
    compute this once per (ticker, combo) and reuse it, rather than repeat
    an O(n_troughs * max_check_ahead) confirmation walk on every grid cell.
    Split out after a parameter sweep first exposed this as the dominant
    cost -- 25 grid cells each recomputing the same near_pct-independent
    walk from scratch made a full sweep 25x slower than necessary (see
    tech_level_placebo_sensitivity.py).
    """
    distance = combo["distance"]
    prominence = combo["prominence"]
    log_close = np.log(close.to_numpy())
    trough_idx, _ = find_peaks(-log_close, distance=distance, prominence=prominence)
    n = len(close)
    confirm_pos = {}
    for t_pos in trough_idx:
        for as_of in range(t_pos, min(t_pos + max_check_ahead, n - 1) + 1):
            if _trough_confirmed_by(log_close, distance, prominence, t_pos, as_of):
                confirm_pos[t_pos] = as_of
                break
    return trough_idx, confirm_pos


def placebo_raw_days_from_confirmations(close, trough_idx, confirm_pos, near_pct=NEAR_PCT,
                                         age_cutoff_days=AGE_CUTOFF_DAYS):
    """Cheap half of placebo_raw_days: the day-level near_pct/age filter,
    given precomputed (trough_idx, confirm_pos) from placebo_trough_confirmations.
    """
    idx = close.index
    n = len(idx)
    if len(trough_idx) == 0:
        return set()

    qualifying = set()
    ti = 0  # pointer into trough_idx
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
            qualifying.add(i)
    return qualifying


def placebo_raw_days(close, combo, near_pct=NEAR_PCT, age_cutoff_days=AGE_CUTOFF_DAYS,
                      max_check_ahead=15):
    """Day-level bare-local-low condition -- price within near_pct of a
    recently confirmed local trough -- with NO one-position-at-a-time
    blocking applied. Returns the set of every day position that
    raw-qualifies, independent of whether an earlier candidate's hold window
    would still be open.

    Factored out of build_placebo_candidates (which applies the blocking on
    top of this) so a day-level condition is available on its own --
    needed by tech_level_confluence_check.py to test whether this condition
    and the real technical-level condition hold on the *same* day, which
    the blocked, one-trade-per-window candidate list would silently distort
    (a day could fail to appear in the blocked list only because an earlier,
    unrelated candidate's hold window was still open, not because the
    condition itself was false that day).

    Convenience wrapper around placebo_trough_confirmations +
    placebo_raw_days_from_confirmations for a single (near_pct,
    age_cutoff_days) call -- prefer calling those two directly when sweeping
    near_pct/age_cutoff_days for the same (close, combo) repeatedly.
    """
    trough_idx, confirm_pos = placebo_trough_confirmations(close, combo, max_check_ahead)
    return placebo_raw_days_from_confirmations(close, trough_idx, confirm_pos, near_pct, age_cutoff_days)


def build_placebo_candidates(close, combo, near_pct=NEAR_PCT, age_cutoff_days=AGE_CUTOFF_DAYS,
                              max_check_ahead=15, hold_days=HOLD_DAYS, confirmations=None):
    """Causal local-low entries with no level/band machinery: price within
    near_pct of a recently confirmed local trough (placebo_raw_days), with
    one-position-at-a-time blocking applied -- a trade-level candidate list,
    matching how build_placebo_trades and the original naive-strategy
    backtest both treat one open position per ticker at a time. `hold_days`
    controls how long an accepted candidate blocks new ones (defaults to the
    module HOLD_DAYS for existing callers); a hold_days *sweep* must pass the
    value actually being tested here, not rely on the default, or the
    blocking window would silently mismatch the position-duration being
    simulated (see tech_level_placebo_sensitivity.py). `confirmations`, when
    given, is a precomputed (trough_idx, confirm_pos) pair from
    placebo_trough_confirmations -- pass it when sweeping near_pct/hold_days
    for the same (close, combo) to skip recomputing the expensive,
    near_pct-independent confirmation walk on every call. (An earlier version
    of this function approximated confirmation as trough_date + distance
    trading sessions; distance=10 means that bound is >= ~14 calendar days,
    which always exceeds the 5-calendar-day age window and silently made
    this placebo empty by construction -- caught because Step C printed zero
    candidates on every ticker, not by inspection.)
    """
    n = len(close)
    if confirmations is not None:
        trough_idx, confirm_pos = confirmations
        qualifying = placebo_raw_days_from_confirmations(close, trough_idx, confirm_pos, near_pct, age_cutoff_days)
    else:
        qualifying = placebo_raw_days(close, combo, near_pct, age_cutoff_days, max_check_ahead)

    candidates = []
    in_position_until = -1
    for i in sorted(qualifying):
        if i <= in_position_until:
            continue
        candidates.append(i)
        in_position_until = min(i + hold_days, n - 1)
    return candidates


def build_placebo_trades(which, series, target_counts, cost_bps=10.0, seed=0):
    """Placebo trade set matched to target_counts (per-ticker trade counts
    from Step A's causally-confirmed real sample), using the same causal
    entry+exit mechanics as build_causal_trades except the entry condition
    is build_placebo_candidates's raw-trough check instead of a two-touch
    technical level. Exit still uses causal_entry_and_resistance/causal_exit
    (whatever support/resistance framework applies at that date) so entry
    is the only thing that differs from the real trades.
    """
    print(f"\n########## Step C: reversal placebo: {which} ##########")
    combo = load_fixed_combo()
    rng = np.random.default_rng(seed)
    rows = []
    for t, close_full in series.items():
        target_n = int(target_counts.get(t, 0))
        if target_n == 0:
            continue
        candidates = build_placebo_candidates(close_full, combo)
        if not candidates:
            continue
        picked_idx = rng.choice(candidates, size=min(target_n, len(candidates)), replace=False)
        for i in sorted(picked_idx):
            entry_date = close_full.index[i]
            entry_price = float(close_full.iloc[i])
            try:
                _, resistance = causal_entry_and_resistance(close_full, entry_date, entry_price, combo)
                exit_info = causal_exit(close_full, entry_date, entry_price, resistance, HOLD_DAYS)
            except Exception:
                exit_info = None
            if exit_info is None:
                continue
            rows.append({
                "stock": t, "entry_date": entry_date, "entry_price": entry_price,
                "exit_date": exit_info["exit_date"], "exit_price": exit_info["exit_price"],
                "exit_reason": exit_info["exit_reason"],
                "ret": exit_info["ret"], "ret_net": exit_info["ret"] - cost_bps / 1e4,
            })

    placebo_df = pd.DataFrame(rows)
    if placebo_df.empty:
        print("no placebo trades generated")
        return placebo_df
    ew = equal_weight_curve(series)
    placebo_df = attach_benchmark(placebo_df, ew)
    stats = pooled(placebo_df)
    print(f"placebo (local-low, no level) -- n={stats['n']:5d}  "
          f"mean_excess={stats['mean_excess']:+.4%}  t={stats['t']:.2f}")
    placebo_df.to_csv(os.path.join(DATA_DIR, f"placebo_reversal_{which}.csv"), index=False)
    return placebo_df


def placebo_comparison(which, causal_df, placebo_df):
    print(f"\n--- Step C verdict: {which} ---")
    real_stats = pooled(causal_df)
    placebo_stats = pooled(placebo_df)
    lift = real_stats["mean_excess"] - placebo_stats["mean_excess"]
    print(f"real (technical level)  -- n={real_stats['n']:5d}  mean_excess={real_stats['mean_excess']:+.4%}")
    print(f"placebo (local low only) -- n={placebo_stats['n']:5d}  mean_excess={placebo_stats['mean_excess']:+.4%}")
    print(f"lift from requiring a technical level: {lift:+.4%}")


# ---------------------------------------------------------------------------
# Step D: concurrency-capped portfolio equity curve
# ---------------------------------------------------------------------------

def slotted_trades(trades_df, n_slots):
    """Greedily accept trades in entry-date order, skipping any that would
    exceed n_slots concurrently open positions (a position frees its slot
    on its own exit_date, available for same-day reuse -- matches simulate's
    one-position-per-ticker convention of blocking through the exit bar but
    freeing it same-day).
    """
    df = trades_df.sort_values("entry_date").reset_index(drop=True)
    open_exits = []
    accepted = []
    for row in df.itertuples():
        open_exits = [d for d in open_exits if d > row.entry_date]
        if len(open_exits) < n_slots:
            accepted.append(row.Index)
            open_exits.append(row.exit_date)
    return df.loc[accepted]


def slotted_equity_curve(accepted, series, n_slots, cost_bps=10.0):
    """Daily mark-to-market equity curve: each day, average the daily
    returns of currently-open accepted positions, scaled by the fraction of
    slots actually filled (idle capital earns 0%, no cash yield assumed).
    Entry-day cost is applied as a one-time drag on that day's contribution.
    """
    if accepted.empty:
        return pd.Series(dtype=float), {}

    all_dates = sorted(set().union(*[set(c.index) for c in series.values()]))
    all_dates = pd.DatetimeIndex(all_dates)
    daily_rets = pd.Series(0.0, index=all_dates)
    counts = pd.Series(0, index=all_dates)

    for row in accepted.itertuples():
        c = series[row.stock]
        window = c.loc[row.entry_date:row.exit_date]
        if len(window) < 2:
            continue
        r = window.pct_change().dropna()
        daily_rets.loc[r.index] += r.values
        counts.loc[r.index] += 1
        daily_rets.loc[r.index[0]] -= cost_bps / 1e4  # one-time entry cost drag

    with np.errstate(invalid="ignore", divide="ignore"):
        port_ret = (daily_rets / counts.replace(0, np.nan)).fillna(0.0)
    port_ret = port_ret * (counts.clip(upper=n_slots) / n_slots)

    active_start = accepted.entry_date.min()
    port_ret = port_ret.loc[port_ret.index >= active_start]
    equity = (1 + port_ret).cumprod()

    n_years = (port_ret.index[-1] - port_ret.index[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    daily_sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else np.nan
    running_max = equity.cummax()
    max_dd = (equity / running_max - 1).min()

    stats = {
        "n_slots": n_slots, "n_trades_accepted": len(accepted),
        "cagr": cagr, "sharpe": daily_sharpe, "max_drawdown": max_dd,
        "final_equity_multiple": equity.iloc[-1],
    }
    return equity, stats


def portfolio_report(which, causal_df, series, slot_scenarios=(5, 10, 24)):
    print(f"\n########## Step D: concurrency-capped portfolio, {which} ##########")
    rows = []
    equities = {}
    for n_slots in slot_scenarios:
        accepted = slotted_trades(causal_df, n_slots)
        equity, stats = slotted_equity_curve(accepted, series, n_slots)
        if stats:
            print(f"n_slots={n_slots:3d}  accepted={stats['n_trades_accepted']:5d}/{len(causal_df)}  "
                  f"CAGR={stats['cagr']:+.2%}  Sharpe={stats['sharpe']:.2f}  "
                  f"maxDD={stats['max_drawdown']:.2%}  final={stats['final_equity_multiple']:.2f}x")
            rows.append(stats)
            equities[n_slots] = equity
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(os.path.join(DATA_DIR, f"portfolio_equity_stats_{which}.csv"), index=False)
    return out, equities


def combined_report(all_causal, all_series, slot_scenarios=(5, 10, 15, 20), account_sizes=(10_000, 100_000)):
    """The live script trades one book across the full ticker_list + oos
    union (100 names here -- 101 minus the excluded NVDA, see EXCLUDE_OOS),
    not two separate universes -- so the per-universe trades/year and
    per-universe portfolio sims in the sections above understate real
    concurrency and don't answer "how many trades a year / how much money"
    for the account actually being run. Merges both universes' causally-
    confirmed trades and price series and reruns Steps B/D on the union.
    """
    print("\n########## Combined universe (ticker_list + oos, as actually traded live) ##########")
    causal_df = pd.concat(all_causal.values(), ignore_index=True)
    series = {}
    for s in all_series.values():
        series.update(s)
    print(f"{len(series)} distinct tickers, {len(causal_df)} causally-confirmed trades")

    span_years = (pd.Timestamp(causal_df.entry_date.max()) - pd.Timestamp(causal_df.entry_date.min())).days / 365.25
    trades_per_year = len(causal_df) / span_years
    stats = pooled(causal_df)
    print(f"trades/year: {trades_per_year:.1f}  (n={len(causal_df)} over {span_years:.1f} years)")
    print(f"mean net return per trade (own capital, not vs. benchmark): {causal_df.ret_net.mean():+.4%}")
    print(f"mean excess return per trade (vs. equal-weight benchmark):  {stats['mean_excess']:+.4%}  t={stats['t']:.2f}")

    boot = block_bootstrap_ci(causal_df)
    print(f"calendar-year block bootstrap (excess) -- 95% CI=[{boot['ci_lo']:+.4%}, {boot['ci_hi']:+.4%}]  "
          f"P(mean<=0)={boot['p_le_0']:.1%}")

    print()
    rows = []
    for n_slots in slot_scenarios:
        accepted = slotted_trades(causal_df, n_slots)
        equity, stats_d = slotted_equity_curve(accepted, series, n_slots)
        if not stats_d:
            continue
        rows.append(stats_d)
        pct_accepted = stats_d["n_trades_accepted"] / len(causal_df)
        line = (f"n_slots={n_slots:3d}  accepted={stats_d['n_trades_accepted']:5d}/{len(causal_df)} "
                f"({pct_accepted:.0%})  CAGR={stats_d['cagr']:+.2%}  Sharpe={stats_d['sharpe']:.2f}  "
                f"maxDD={stats_d['max_drawdown']:.2%}  final={stats_d['final_equity_multiple']:.2f}x")
        for acct in account_sizes:
            line += f"  | ${acct:,}->${acct * (1 + stats_d['cagr']):,.0f}/yr"
        print(line)

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(os.path.join(DATA_DIR, "portfolio_equity_stats_combined.csv"), index=False)
    causal_df.to_csv(os.path.join(DATA_DIR, "causal_full_combined.csv"), index=False)
    return causal_df, out


# ---------------------------------------------------------------------------
def main():
    all_causal, all_series = {}, {}
    for which, trades_file in (("ticker_list", "naive_strategy_trades.csv"),
                                ("oos", "oos_strategy_trades.csv")):
        series = load_universe(which)
        print(f"\n{len(series)} tickers loaded for {which}")

        causal_df = build_causal_trades(which, series, os.path.join(DATA_DIR, trades_file))
        if causal_df.empty:
            print(f"skipping downstream steps for {which} -- no causally-confirmed trades")
            continue

        concentration_report(which, causal_df)
        distance_sensitivity(which, series, distances=[3, 5, 10, 15, 20])
        inference_report(which, causal_df)

        target_counts = causal_df.groupby("stock").size()
        placebo_df = build_placebo_trades(which, series, target_counts)
        if not placebo_df.empty:
            placebo_comparison(which, causal_df, placebo_df)

        portfolio_report(which, causal_df, series)

        all_causal[which] = causal_df
        all_series[which] = series

    if len(all_causal) == 2:
        combined_report(all_causal, all_series)


if __name__ == "__main__":
    main()
