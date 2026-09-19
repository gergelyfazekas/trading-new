"""Experimental variant of tech_levels.build_levels_causal: a level is only
allowed to be born if its birth date's volume ratio (that day's volume over
the trailing 10-trading-day average, same causal definition used in
tech_level_walkforward_demo.py's volume-ratio check) is at least `min_ratio`.

This is a standalone copy-and-modify, not a patch to tech_levels.py -- the
live strategy and every validated backtest in this package keep using the
original, untouched build_levels_causal. Nothing here is wired into
tech_level_naive_strategy.py or tech_level_continuation_live.py.

Gating rule when a touch matches an open candidate but the birth-date volume
gate fails: treated exactly as if no match had been found at all -- the
matched candidate stays open, and the current touch is filed as its own new
open candidate. Neither is discarded, so a later, higher-volume touch can
still confirm either of them. This keeps the gate a pure additional
condition on birthing rather than a rule that also deletes candidates the
original ungated logic would have kept alive.
"""
import pandas as pd

from tech_levels import Level


def volume_ratio(volume: pd.Series, date, lookback: int = 10):
    """That day's volume over the mean of the `lookback` trading days
    strictly before it (never including the day itself) -- same causal
    definition as tech_level_walkforward_demo.trailing_avg_volume. Returns
    None if the date isn't in the series or there's no usable trailing
    window (e.g. too close to the start of history).
    """
    ts = pd.Timestamp(date)
    if ts not in volume.index:
        return None
    pos = volume.index.get_loc(ts)
    seg = volume.iloc[max(0, pos - lookback):pos]
    if len(seg) == 0 or seg.isna().all():
        return None
    avg = float(seg.mean())
    if avg == 0.0 or pd.isna(avg):
        return None
    return float(volume.iloc[pos]) / avg


def build_levels_causal_volume_gated(touches, tech_width, volume, min_ratio=1.0, lookback=10):
    """Same walk as tech_levels.build_levels_causal, plus the birth-date
    volume gate. Returns (levels, n_gated_out) -- the second is a diagnostic
    count of how many births the gate blocked, to sanity-check the filter is
    actually doing something on a given (ticker, combo).
    """
    levels = []
    open_candidates = []
    n_gated_out = 0

    for date, price in touches:
        if any(lvl.band[0] <= price <= lvl.band[1] for lvl in levels):
            continue

        within = [c for c in open_candidates if abs(price - c[1]) < tech_width * c[1]]
        if not within:
            open_candidates.append((date, price))
            continue

        matched = min(within, key=lambda c: abs(price - c[1]))
        ratio = volume_ratio(volume, date, lookback=lookback)
        if ratio is not None and ratio >= min_ratio:
            band = (min(matched[1], price), max(matched[1], price))
            levels.append(Level(band=band, birth_date=date))
            open_candidates.remove(matched)
        else:
            n_gated_out += 1
            open_candidates.append((date, price))

    return levels, n_gated_out
