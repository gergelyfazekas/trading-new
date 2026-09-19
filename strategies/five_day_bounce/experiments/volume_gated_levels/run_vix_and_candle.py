"""Two new, genuinely different "market psychology" features, tested with
the lesson from the market_depth_10d episode applied from the start: run
against BOTH the two-touch technical-level population and the bare-local-low
population, on both config.ticker_list and config.oos_ticker_list, not as
an afterthought.

1. VIX (an external fear gauge, NOT derived from the traded universe itself
   -- every prior "market mood" feature was built from the equal-weight book
   of the same stocks being traded, which is exactly why market_depth_10d
   could plausibly be entangled with which stocks the two-touch filter
   happens to select). Candidates: vix_level (yesterday's close), vix_chg_5d/
   vix_chg_10d (cumulative change), vix_ratio_5_60 (5d avg / 60d avg -- a
   spike indicator, same construction as the volume/vol-spike ratios
   elsewhere in this folder).

2. Candle shape on the touch day (High/Low, untouched by every previous
   script in this folder, which only ever used close and volume).
   clv_entry = (close-low)/(high-low) on the entry day itself -- a "hammer"
   reversal candle (plunge intraday, close back near the high) is a much
   more direct behavioral signal than the close-to-close return. Also
   clv_prev_day (did the reversal start a day earlier) and range_ratio_entry
   (today's high-low range vs. its own trailing 10d average -- an
   unusually wide, high-emotion day).

Reuses tech_level_causal_check / tech_level_confluence_check /
experiment_lib exactly as run_market_depth_bare_local_low.py does -- no new
detector, no changes to any live file.

Usage: ./venv/bin/python strategies/five_day_bounce/experiments/volume_gated_levels/run_vix_and_candle.py
"""
import datetime
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
from stock_class import StockList
from tech_level_naive_strategy import load_fixed_combo, equal_weight_curve, attach_benchmark
from tech_level_continuation_live import MAX_AGE_DAYS
from tech_level_confluence_check import build_full_placebo_trades
from experiment_lib import build_young_level_trades, placebo_trades
from run_momentum_mood import HOLD_DAYS, NEAR_PCT, trailing_return
from run_dip_depth import plot_feature_buckets

START_DATE = datetime.date(2015, 5, 28)
EXCLUDE_OOS = {"NVDA"}
FEATURES = ["vix_level", "vix_chg_5d", "vix_chg_10d", "vix_ratio_5_60",
            "clv_entry", "clv_prev_day", "range_ratio_entry"]
OUT_DIR = os.path.join(_HERE, "data")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_universe_ohlc(which):
    if which == "ticker_list":
        tickers = list(config.ticker_list)
        sl = StockList(tickers)
        sl.pull_data(start=START_DATE, end=datetime.date.today() + datetime.timedelta(days=1))
    else:
        tickers = [t for t in config.oos_ticker_list if t not in EXCLUDE_OOS]
        sl = StockList(tickers)
        sl.load_data(config.oos)

    close_series, high_series, low_series, volume_series = {}, {}, {}, {}
    for t in tickers:
        try:
            data = sl[t].data
            close = data["close"].dropna()
            if close.empty:
                continue
            close.index = pd.to_datetime(close.index)
            high = data["high"].reindex(data["close"].dropna().index)
            high.index = pd.to_datetime(high.index)
            low = data["low"].reindex(data["close"].dropna().index)
            low.index = pd.to_datetime(low.index)
            volume = data["volume"].reindex(close.index) if "volume" in data.columns else None
            if volume is not None:
                volume.index = pd.to_datetime(volume.index)
            close_series[t], high_series[t], low_series[t], volume_series[t] = close, high, low, volume
        except Exception:
            continue
    return close_series, high_series, low_series, volume_series


def pull_vix():
    sl = StockList(["^VIX"])
    sl.pull_data(start=START_DATE, end=datetime.date.today() + datetime.timedelta(days=1))
    close = sl["^VIX"].data["close"].dropna()
    close.index = pd.to_datetime(close.index)
    return close


# ---------------------------------------------------------------------------
# feature functions
# ---------------------------------------------------------------------------

def trailing_value(series, date, offset=1):
    ts = pd.Timestamp(date)
    if ts not in series.index:
        return None
    pos = series.index.get_loc(ts)
    if pos - offset < 0:
        return None
    return float(series.iloc[pos - offset])


def trailing_avg(series, date, n):
    ts = pd.Timestamp(date)
    if ts not in series.index:
        return None
    pos = series.index.get_loc(ts)
    seg = series.iloc[max(0, pos - n):pos]
    return float(seg.mean()) if len(seg) and seg.notna().any() else None


def clv(high_val, low_val, close_val):
    """Close location value: 0 = closed at the low, 1 = closed at the high."""
    if high_val is None or low_val is None or close_val is None:
        return None
    if pd.isna(high_val) or pd.isna(low_val) or pd.isna(close_val):
        return None
    rng = high_val - low_val
    if rng <= 0:
        return 0.5
    return float((close_val - low_val) / rng)


def clv_on_date(high_s, low_s, close_s, date):
    ts = pd.Timestamp(date)
    if ts not in close_s.index:
        return None
    return clv(high_s.get(ts), low_s.get(ts), close_s.get(ts))


def clv_prev_day(high_s, low_s, close_s, date):
    ts = pd.Timestamp(date)
    if ts not in close_s.index:
        return None
    pos = close_s.index.get_loc(ts)
    if pos - 1 < 0:
        return None
    return clv(high_s.iloc[pos - 1], low_s.iloc[pos - 1], close_s.iloc[pos - 1])


def range_ratio_entry(high_s, low_s, close_s, date, lookback=10):
    ts = pd.Timestamp(date)
    if ts not in close_s.index:
        return None
    pos = close_s.index.get_loc(ts)
    if pd.isna(high_s.iloc[pos]) or pd.isna(low_s.iloc[pos]) or close_s.iloc[pos] == 0:
        return None
    today_range = (high_s.iloc[pos] - low_s.iloc[pos]) / close_s.iloc[pos]
    if pos - lookback < 0:
        return None
    h = high_s.iloc[pos - lookback:pos]
    l = low_s.iloc[pos - lookback:pos]
    c = close_s.iloc[pos - lookback:pos]
    hist_ranges = (h - l) / c
    avg_range = hist_ranges.mean()
    if pd.isna(avg_range) or avg_range == 0:
        return None
    return float(today_range / avg_range)


def add_features(trades_df, high_series, low_series, close_series, vix):
    trades_df = trades_df.copy()
    trades_df["vix_level"] = [trailing_value(vix, r.entry_date, 1) for r in trades_df.itertuples()]
    trades_df["vix_chg_5d"] = [trailing_return(vix, r.entry_date, 5) for r in trades_df.itertuples()]
    trades_df["vix_chg_10d"] = [trailing_return(vix, r.entry_date, 10) for r in trades_df.itertuples()]

    ratios = []
    for r in trades_df.itertuples():
        v5, v60 = trailing_avg(vix, r.entry_date, 5), trailing_avg(vix, r.entry_date, 60)
        ratios.append(v5 / v60 if v5 is not None and v60 not in (None, 0.0) else None)
    trades_df["vix_ratio_5_60"] = ratios

    trades_df["clv_entry"] = [
        clv_on_date(high_series[r.stock], low_series[r.stock], close_series[r.stock], r.entry_date)
        for r in trades_df.itertuples()
    ]
    trades_df["clv_prev_day"] = [
        clv_prev_day(high_series[r.stock], low_series[r.stock], close_series[r.stock], r.entry_date)
        for r in trades_df.itertuples()
    ]
    trades_df["range_ratio_entry"] = [
        range_ratio_entry(high_series[r.stock], low_series[r.stock], close_series[r.stock], r.entry_date)
        for r in trades_df.itertuples()
    ]
    return trades_df


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyze(label, trades_df, close_series, vix):
    print(f"\n{'='*70}\n{label}  (n={len(trades_df)})\n{'='*70}")
    bonferroni_bar = 0.05 / len(FEATURES)
    rows = []
    for col in FEATURES:
        sub = trades_df.dropna(subset=[col])
        if len(sub) < 30:
            continue
        rho, p = stats.spearmanr(sub[col], sub.excess_ret)
        rows.append({"feature": col, "n": len(sub), "spearman_rho": rho, "p_value": p})
    corr_df = pd.DataFrame(rows).sort_values("p_value")
    print(f"(Bonferroni bar: p<{bonferroni_bar:.4f})")
    print(corr_df.to_string(index=False))

    survivors = corr_df[corr_df.p_value < bonferroni_bar]
    if survivors.empty:
        print("no feature survives Bonferroni here")
        return corr_df, []

    print(f"\nsurvives Bonferroni bar:")
    print(survivors.to_string(index=False))

    real_counts = trades_df.groupby("stock").size()
    placebo_df = placebo_trades(close_series, real_counts, hold_days=HOLD_DAYS)
    ew = equal_weight_curve(close_series)
    placebo_df = attach_benchmark(placebo_df, ew)

    survivor_details = []
    for _, row in survivors.iterrows():
        feat = row["feature"]
        # specificity: does a matched random-entry placebo show the same correlation?
        placebo_feat_vals = None
        if feat.startswith("vix_"):
            if feat == "vix_level":
                placebo_feat_vals = [trailing_value(vix, r.entry_date, 1) for r in placebo_df.itertuples()]
            elif feat == "vix_chg_5d":
                placebo_feat_vals = [trailing_return(vix, r.entry_date, 5) for r in placebo_df.itertuples()]
            elif feat == "vix_chg_10d":
                placebo_feat_vals = [trailing_return(vix, r.entry_date, 10) for r in placebo_df.itertuples()]
            elif feat == "vix_ratio_5_60":
                placebo_feat_vals = []
                for r in placebo_df.itertuples():
                    v5, v60 = trailing_avg(vix, r.entry_date, 5), trailing_avg(vix, r.entry_date, 60)
                    placebo_feat_vals.append(v5 / v60 if v5 is not None and v60 not in (None, 0.0) else None)
        p_sub = placebo_df.copy()
        p_sub[feat] = placebo_feat_vals if placebo_feat_vals is not None else np.nan
        p_sub = p_sub.dropna(subset=[feat])
        if len(p_sub) >= 30:
            p_rho, p_p = stats.spearmanr(p_sub[feat], p_sub.excess_ret)
        else:
            p_rho, p_p = float("nan"), float("nan")
        print(f"\n{feat}: real rho={row['spearman_rho']:+.3f} (p={row['p_value']:.4f})   "
              f"placebo rho={p_rho:+.3f} (p={p_p:.4f}, n={len(p_sub)})")
        chart_path = os.path.join(OUT_DIR, f"vix_candle_{label.replace(' ', '_').replace('/', '_')}_{feat}.png")
        plot_feature_buckets(trades_df, feat, chart_path)
        survivor_details.append({"feature": feat, "real_rho": row["spearman_rho"], "real_p": row["p_value"],
                                   "placebo_rho": p_rho, "placebo_p": p_p})

    return corr_df, survivor_details


def main():
    combo = load_fixed_combo()
    print("pulling VIX...")
    vix = pull_vix()
    print(f"{len(vix)} VIX observations, {vix.index[0].date()} -> {vix.index[-1].date()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []

    for which in ("ticker_list", "oos"):
        print(f"\n\n########## universe: {which} ##########")
        close_series, high_series, low_series, volume_series = load_universe_ohlc(which)
        print(f"{len(close_series)} tickers loaded")

        tickers = list(close_series.keys())
        series_cv = {t: (close_series[t], volume_series[t]) for t in tickers}

        print("building two-touch technical-level population...")
        two_touch_df, _cs, _vs, ew, failed = build_young_level_trades(
            tickers, combo, hold_days=HOLD_DAYS, near_pct=NEAR_PCT, max_age_days=MAX_AGE_DAYS, series=series_cv
        )
        two_touch_df = add_features(two_touch_df, high_series, low_series, close_series, vix)
        two_touch_df.to_csv(os.path.join(OUT_DIR, f"vix_candle_two_touch_{which}_trades.csv"), index=False)
        corr, surv = analyze(f"two-touch / {which}", two_touch_df, close_series, vix)
        all_results.append((f"two-touch / {which}", corr, surv))

        print("\nbuilding bare-local-low population...")
        bare_df = build_full_placebo_trades(close_series, combo)
        if not bare_df.empty:
            bare_df = add_features(bare_df, high_series, low_series, close_series, vix)
            bare_df.to_csv(os.path.join(OUT_DIR, f"vix_candle_bare_local_low_{which}_trades.csv"), index=False)
            corr, surv = analyze(f"bare-local-low / {which}", bare_df, close_series, vix)
            all_results.append((f"bare-local-low / {which}", corr, surv))

    print(f"\n\n{'='*70}\nSUMMARY: features surviving Bonferroni, by population\n{'='*70}")
    for label, corr, surv in all_results:
        names = [s["feature"] for s in surv] if surv else []
        print(f"{label:30s}: {names if names else '(none)'}")


if __name__ == "__main__":
    main()
