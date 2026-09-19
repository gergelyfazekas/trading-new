"""Shared pieces for both the small (run_experiment.py) and full-scale
(run_experiment_full.py) volume-gate experiments -- level construction for
the baseline (unmodified tech_levels.build_levels_causal) and volume-gated
(build_levels_volume_gated.py) arms, plus a placebo/random-entry generator
matched to a target trade count per ticker (same convention
tech_level_naive_strategy.run_backtest uses inline for its own placebo).

Factored out so both scripts build levels identically instead of risking the
two drifting apart, and so the full-scale run doesn't re-implement the
placebo logic from scratch.
"""
import numpy as np
import pandas as pd

from tech_levels import build_levels_causal, mark_broken
from tech_level_naive_strategy import find_touches, active_support_resistance
from build_levels_volume_gated import build_levels_causal_volume_gated


def build_levels_baseline(close, combo):
    """Identical mechanics to tech_level_naive_strategy.build_levels (that
    function isn't importable standalone -- it's private to that module's
    own script flow) -- kept byte-for-byte equivalent so "baseline" here
    means exactly what the live/backtested strategy actually uses.
    """
    combo = dict(combo)
    tech_width = combo.pop("tech_width")
    consider_volume = combo.pop("consider_volume", False)
    volume_height = combo.pop("volume_height", 0.0)
    volume_prominence = combo.pop("volume_prominence", 0.0)
    touches = find_touches(close, consider_volume=consider_volume,
                            volume_height=volume_height, volume_prominence=volume_prominence,
                            **combo)
    levels = build_levels_causal(touches, tech_width)
    mark_broken(levels, close)
    return levels


def build_levels_gated(close, volume, combo, min_ratio=1.0):
    combo = dict(combo)
    tech_width = combo.pop("tech_width")
    consider_volume = combo.pop("consider_volume", False)
    volume_height = combo.pop("volume_height", 0.0)
    volume_prominence = combo.pop("volume_prominence", 0.0)
    touches = find_touches(close, consider_volume=consider_volume,
                            volume_height=volume_height, volume_prominence=volume_prominence,
                            **combo)
    levels, n_gated_out = build_levels_causal_volume_gated(touches, tech_width, volume, min_ratio=min_ratio)
    mark_broken(levels, close)
    return levels, n_gated_out


def simulate_age_gated(ticker, close, levels, hold_days, near_pct=0.01, max_age_days=5, cost_bps=10.0):
    """Copy of tech_level_naive_strategy.simulate with one added entry
    condition: the matched support must be younger than `max_age_days`
    calendar days -- the MAX_AGE_DAYS gate tech_level_continuation_live.py
    actually trades live (age_days < MAX_AGE_DAYS, strict), so this arm asks
    "does the volume gate add anything on top of the already-deployed
    young-level rule" rather than the looser any-age naive rule. Kept as its
    own copy, not a parameter added to the shared simulate(), so the
    untouched naive-strategy backtest can't be affected by this experiment.
    """
    idx = close.index
    n = len(idx)
    levels_sorted = sorted(levels, key=lambda lvl: lvl.birth_date)

    trades = []
    in_position_until = None

    for i, date in enumerate(idx):
        if in_position_until is not None and i <= in_position_until:
            continue
        in_position_until = None

        price = close.iloc[i]
        support, resistance = active_support_resistance(levels_sorted, date, price)

        if support is None:
            continue
        age_days = (pd.Timestamp(date) - pd.Timestamp(support.birth_date)).days
        dist = (price - support.band[1]) / price
        is_entry = age_days < max_age_days and 0 <= dist <= near_pct
        if not is_entry:
            continue

        entry_date, entry_price = date, price
        exit_pos = min(i + hold_days, n - 1)
        exit_reason = "hold_days"

        for j in range(i + 1, min(i + hold_days, n - 1) + 1):
            if resistance is not None and resistance.band[0] <= close.iloc[j] <= resistance.band[1]:
                exit_pos = j
                exit_reason = "resistance"
                break

        if i + hold_days >= n and exit_reason == "hold_days":
            continue

        exit_date, exit_price = idx[exit_pos], close.iloc[exit_pos]
        trades.append({
            "stock": ticker, "hold_days": hold_days, "near_pct": near_pct,
            "entry_date": entry_date, "exit_date": exit_date,
            "entry_price": float(entry_price), "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "ret": float(exit_price / entry_price - 1.0),
            "ret_net": float(exit_price / entry_price - 1.0) - cost_bps / 1e4,
            "support_age_days": age_days,
            "support_birth_date": support.birth_date,
        })
        in_position_until = exit_pos

    return trades


def build_young_level_trades(tickers, combo, hold_days=5, near_pct=0.01, max_age_days=5, series=None):
    """Build baseline (ungated) levels and generate the young-level
    (<max_age_days) trade population -- the one shared starting point for
    every per-trade feature analysis in this experiments folder (volume
    dose-response, momentum/mood, and anything added later), so they're all
    testing the exact same trade set rather than each script quietly
    building its own slightly different population.

    `series`, if given, is a pre-loaded {ticker: (close, volume)} dict --
    e.g. from a frozen local pull for config.oos_ticker_list, which must
    NOT be re-pulled fresh from yfinance (same convention as
    tech_level_oos_strategy.py, so the OOS record doesn't depend on when
    this happens to run). Defaults to a fresh batched pull
    (tech_level_live.pull_all) when omitted, which is correct for
    config.ticker_list.

    Returns (trades_df with excess_ret attached, close_series, volume_series,
    ew equal-weight curve, failed tickers).
    """
    from tech_level_naive_strategy import equal_weight_curve, attach_benchmark

    failed = []
    if series is None:
        from tech_level_live import pull_all
        series, failed = pull_all(tickers)

    close_series, volume_series = {}, {}
    all_trades = []
    for t, (close, volume) in series.items():
        close = close.copy()
        close.index = pd.to_datetime(close.index)
        volume = volume.copy()
        volume.index = pd.to_datetime(volume.index)
        close_series[t] = close
        volume_series[t] = volume

        levels = build_levels_baseline(close, combo)
        all_trades.extend(simulate_age_gated(t, close, levels, hold_days=hold_days,
                                               near_pct=near_pct, max_age_days=max_age_days))

    ew = equal_weight_curve(close_series)
    trades_df = attach_benchmark(pd.DataFrame(all_trades), ew)
    return trades_df, close_series, volume_series, ew, failed


def placebo_trades(series, target_counts, hold_days=5, cost_bps=10.0, seed=0):
    """Random-entry trades, same convention as
    tech_level_naive_strategy.run_backtest's inline placebo block: for each
    ticker, sample candidate start indices and take the first `target_counts[t]`
    of them (in date order) that have room for a full `hold_days` exit --
    matches each ticker's real trade count/frequency so a lift isn't just
    "this stock went up regardless of entry point."
    """
    rng = np.random.default_rng(seed)
    trades = []
    for t, c in series.items():
        target_n = int(target_counts.get(t, 0))
        if target_n == 0:
            continue
        idx = c.index
        n = len(idx)
        if n <= hold_days + 1:
            continue
        candidate_starts = rng.choice(np.arange(0, n - hold_days - 1),
                                        size=min(target_n * 20, n - hold_days - 1), replace=False)
        picked = 0
        for i in sorted(candidate_starts):
            if picked >= target_n:
                break
            exit_pos = min(i + hold_days, n - 1)
            if i + hold_days >= n:
                continue
            trades.append({
                "stock": t, "hold_days": hold_days,
                "entry_date": idx[i], "exit_date": idx[exit_pos],
                "entry_price": float(c.iloc[i]), "exit_price": float(c.iloc[exit_pos]),
                "exit_reason": "hold_days",
                "ret": float(c.iloc[exit_pos] / c.iloc[i] - 1.0),
                "ret_net": float(c.iloc[exit_pos] / c.iloc[i] - 1.0) - cost_bps / 1e4,
            })
            picked += 1
    return pd.DataFrame(trades)
