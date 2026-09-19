"""Does the size of the initial bounce right after a support level is born
predict whether the LATER retest -- the day price comes back near the band
and actually triggers the two-touch buy signal -- turns into a profitable
trade?

Mechanism: a level's birth_date is, by construction, a local trough
(find_peaks only confirms it once price has moved away enough to satisfy
distance/prominence) -- so price almost always rises somewhat right after
birth, then it comes back down toward the band again, and THAT return trip
is what the near_pct=1% / age<5d rule actually enters on. Every feature
tested so far in this folder describes conditions AT the retest/entry; none
of them ask what happened in between. User's hypothesis: a small, tepid
initial uptick vs. a large, sharp one right after birth might carry
information about how much real buying conviction is behind the level --
distinct from, and available earlier than, anything measured at entry time.

Five ways of measuring "the bounce", all using only price action strictly
between birth_date and entry_date (already-observed by the time entry
happens -- no lookahead):
  - bounce_{1,2,3,5}d: fixed-horizon return N trading days after birth,
    regardless of when entry actually happens.
  - bounce_to_entry_high: the full pre-retest rally -- (peak close between
    birth and entry) / birth price -- the most direct operationalization of
    "how big was the uptick before it came back to retest," whatever number
    of days that took. 0.0 when entry fires on the birth day itself (no
    room for a bounce to have happened yet).

Scoped to the two-touch population specifically (matches how the question
was framed -- "the two touch bounce"), checked on both config.ticker_list
and config.oos_ticker_list for replication, same equal-bucket + Bonferroni
discipline as every other feature test in this folder. Not yet extended to
the bare-local-low population -- unlike the other features, "birth date"
isn't a field already carried on that population's trades (it would need
recovering the qualifying trough's position from
tech_level_causal_check.placebo_trough_confirmations), so that's flagged as
a follow-up rather than done here.

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_post_birth_bounce.py
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

import config
from tech_level_naive_strategy import load_fixed_combo, summarise
from tech_level_continuation_live import MAX_AGE_DAYS
from experiment_lib import build_young_level_trades
from run_momentum_mood import HOLD_DAYS, NEAR_PCT
from run_oos_validation import load_oos_series
from run_dip_depth import plot_feature_buckets

FIXED_WINDOWS = [1, 2, 3, 5]
N_BUCKETS = 5
OUT_DIR = os.path.join(_HERE, "data")


def bounce_after_birth(close_s, birth_date, entry_date, n):
    """Return from birth_date to n trading days later -- but ONLY if that
    point is strictly before entry_date. Most entries fire just 1 calendar
    day after birth (median age=1d, 60% of trades), so an n-days-after-birth
    window for n>=2 or 3 frequently lands on or after entry, inside the
    trade's own hold_days=5 window -- correlating that against the trade's
    own excess_ret would just be measuring (part of) the same return twice,
    not a genuine pre-entry signal. Caught exactly this way: an initial pass
    without this guard gave bounce_5d rho=0.60 (p=1e-113), a value far too
    large to be real and diagnostic of leakage on its own.
    """
    ts_b, ts_e = pd.Timestamp(birth_date), pd.Timestamp(entry_date)
    if ts_b not in close_s.index or ts_e not in close_s.index:
        return None
    pos_b, pos_e = close_s.index.get_loc(ts_b), close_s.index.get_loc(ts_e)
    target_pos = pos_b + n
    # target_pos == pos_e (birth+n lands exactly on the entry day) is fine --
    # that price is known at the moment the entry decision is made. Only
    # reject strictly past entry, which would reach into the hold period.
    if target_pos > pos_e or target_pos >= len(close_s):
        return None
    birth_price, later_price = close_s.iloc[pos_b], close_s.iloc[target_pos]
    if birth_price == 0 or pd.isna(birth_price) or pd.isna(later_price):
        return None
    return float(later_price / birth_price - 1.0)


def bounce_to_entry_high(close_s, birth_date, entry_date):
    ts_b, ts_e = pd.Timestamp(birth_date), pd.Timestamp(entry_date)
    if ts_b not in close_s.index or ts_e not in close_s.index:
        return None
    pos_b, pos_e = close_s.index.get_loc(ts_b), close_s.index.get_loc(ts_e)
    birth_price = close_s.iloc[pos_b]
    if birth_price == 0 or pd.isna(birth_price):
        return None
    if pos_e <= pos_b:
        return 0.0
    peak = close_s.iloc[pos_b:pos_e + 1].max()
    if pd.isna(peak):
        return None
    return float(peak / birth_price - 1.0)


def add_bounce_features(trades_df, close_series):
    trades_df = trades_df.copy()
    for n in FIXED_WINDOWS:
        trades_df[f"bounce_{n}d"] = [
            bounce_after_birth(close_series[r.stock], r.support_birth_date, r.entry_date, n)
            for r in trades_df.itertuples()
        ]
    trades_df["bounce_to_entry_high"] = [
        bounce_to_entry_high(close_series[r.stock], r.support_birth_date, r.entry_date)
        for r in trades_df.itertuples()
    ]
    return trades_df


FEATURES = [f"bounce_{n}d" for n in FIXED_WINDOWS] + ["bounce_to_entry_high"]


def analyze(label, trades_df):
    print(f"\n{'='*70}\n{label}  (n={len(trades_df)})\n{'='*70}")
    bonferroni_bar = 0.05 / len(FEATURES)
    rows = []
    for col in FEATURES:
        sub = trades_df.dropna(subset=[col])
        rho, p = stats.spearmanr(sub[col], sub.excess_ret)
        rows.append({"feature": col, "n": len(sub), "spearman_rho": rho, "p_value": p})
    corr_df = pd.DataFrame(rows).sort_values("p_value")
    print(f"(Bonferroni bar: p<{bonferroni_bar:.4f})")
    print(corr_df.to_string(index=False))

    survivors = corr_df[corr_df.p_value < bonferroni_bar]
    if survivors.empty:
        print("no feature survives Bonferroni here")
        return corr_df

    print(f"\nsurvives Bonferroni bar:")
    print(survivors.to_string(index=False))
    for _, row in survivors.iterrows():
        feat = row["feature"]
        sub = trades_df.dropna(subset=[feat]).copy()
        sub["bucket"] = pd.qcut(sub[feat], N_BUCKETS, duplicates="drop")
        print(f"\n{feat} buckets:")
        for b, g in sub.groupby("bucket", observed=True):
            n = len(g)
            t_stat = g.excess_ret.mean() / g.excess_ret.std() * np.sqrt(n) if n > 2 else float("nan")
            print(f"  [{b}]  n={n:4d}  mean_excess={g.excess_ret.mean():+.4%}  t={t_stat:+.2f}")
        by_stock = summarise(trades_df.dropna(subset=[feat]), by="stock")
        chart_path = os.path.join(OUT_DIR, f"post_birth_bounce_{label.replace('/', '_').replace(' ', '_')}_{feat}.png")
        plot_feature_buckets(trades_df, feat, chart_path)
        print(f"chart saved to {chart_path}")

    return corr_df


def main():
    combo = load_fixed_combo()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("building two-touch trade population: ticker_list...")
    tl_trades, tl_close, _v, _ew, failed = build_young_level_trades(
        config.ticker_list, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS
    )
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}")
    tl_trades = add_bounce_features(tl_trades, tl_close)
    tl_trades.to_csv(os.path.join(OUT_DIR, "post_birth_bounce_ticker_list_trades.csv"), index=False)
    tl_corr = analyze("ticker_list", tl_trades)

    print("\nbuilding two-touch trade population: oos (-NVDA, frozen pull)...")
    oos_tickers = [t for t in config.oos_ticker_list if t != "NVDA"]
    oos_series = load_oos_series(oos_tickers)
    oos_trades, oos_close, _v, _ew, failed = build_young_level_trades(
        oos_tickers, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS, series=oos_series
    )
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}")
    oos_trades = add_bounce_features(oos_trades, oos_close)
    oos_trades.to_csv(os.path.join(OUT_DIR, "post_birth_bounce_oos_trades.csv"), index=False)
    oos_corr = analyze("oos", oos_trades)

    print(f"\n\n{'='*70}\nSIDE-BY-SIDE (does anything replicate?)\n{'='*70}")
    merged = tl_corr.merge(oos_corr, on="feature", suffixes=("_ticker_list", "_oos"))
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
