"""Causal technical-level construction and out-of-sample hit/miss scoring.

This is the core mechanics module for validating technical-level parameters
(tech_width, find_peaks sensitivity, ...) against what price actually does
afterward, instead of tuning them by eye (see tuning.tune_tech_levels).

Kept dependency-light and I/O-free on purpose: no Stock/StockList import here,
just plain pandas Series in, plain dataclasses/DataFrames out. See
tech_level_search.py for the parallel grid-search driver that calls into this,
and tech_level_scoring.py for turning the resulting log into scores.
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


@dataclass
class Level:
    band: tuple           # (low, high), frozen at birth -- min/max of the two confirming touches
    birth_date: object     # date of the 2nd confirming touch
    touches: list = field(default_factory=list)  # filled in by score_touches: [(date, 'hit'|'miss', volume_zscore), ...]
    broken_at: object = None  # filled in by mark_broken: date price fully crossed the band, or None


def find_touches(close: pd.Series, volume: pd.Series = None, consider_volume=False,
                  volume_height=0.0, volume_prominence=0.0, with_kind=False, **find_peaks_kwargs):
    """Find candidate touch points (price peaks/troughs, optionally + high-volume dates)
    over the whole series at once. Returns [(date, price), ...] sorted chronologically.

    Adapted from Stock.get_tech_levels/get_high_volumes (stock_class.py), but run once
    over the full history instead of a single from_date/to_date window, and without the
    ADF-based stationarity step get_high_volumes uses for volume -- a simple z-score is
    enough here and keeps this out of the hot path of a large grid search.

    Peak detection runs on log(close), so `prominence` (and `height`) mean a
    *fractional* price move: prominence=0.01 is a ~1% swing at any price level.
    This previously ran on close/close.iloc[-1], which made prominence an absolute
    currency threshold pegged to the final price of the series -- for AAPL that was
    $1.67, i.e. 5.7% of the 2015 price but 0.5% of the 2026 price, so the detector
    was ~11x stricter at the start of history than at the end (4-7 touches/year in
    2015-17 vs 26-31/year in 2021-25). It also leaked the future: truncating the
    series moved the threshold, so the same date could gain or lose a touch
    depending on where the history happened to end -- which made anything derived
    from it non-causal, and meant the in-sample and out-of-sample halves of the
    validation were never measured with the same detector. Log space is
    scale-free, uniform across the history, and depends on no series-wide
    constant. Note this changes what a given prominence value means: grids and
    saved combos calibrated against the old scaling do not carry over.
    """
    log_close = np.log(close.to_numpy())
    peak_idx, _ = find_peaks(log_close, **find_peaks_kwargs)
    trough_idx, _ = find_peaks(-log_close, **find_peaks_kwargs)
    touch_dates = close.index[np.concatenate([peak_idx, trough_idx])]
    kind_by_date = {**{close.index[i]: 'peak' for i in peak_idx},
                    **{close.index[i]: 'trough' for i in trough_idx}}

    if consider_volume and volume is not None:
        centered = (volume - volume.mean()) / volume.std()
        vol_peak_idx, _ = find_peaks(centered.to_numpy(), height=volume_height, prominence=volume_prominence)
        touch_dates = touch_dates.append(volume.index[vol_peak_idx])

    touch_dates = pd.Index(touch_dates).unique().sort_values()
    if with_kind:
        return [(d, float(close.loc[d]), kind_by_date.get(d, 'volume')) for d in touch_dates]
    return [(d, float(close.loc[d])) for d in touch_dates]


def build_levels_causal(touches, tech_width, same_type=False):
    """Build levels from a chronologically-sorted touch list without lookahead.

    Walks touches in date order. A touch either:
      - falls inside an already-frozen level's band -> ignored here (post-birth
        activity is scored separately, off the close series, in score_touches)
      - matches an open (unconfirmed) candidate within tech_width -> births a new
        level; the band freezes at [min, max] of just that pair
      - matches nothing -> becomes a new open candidate

    This is what makes it safe to run once over the whole history: a level's
    existence and boundaries only ever depend on touches at or before its birth.

    same_type=True (experimental, default off = the frozen live behavior) only
    pairs a touch with an open candidate of the same kind (peak+peak or
    trough+trough); needs touches from find_touches(with_kind=True).
    """
    levels = []
    open_candidates = []  # [(date, price, kind), ...] not yet confirmed into a level

    for t in touches:
        date, price = t[0], t[1]
        kind = t[2] if same_type else None
        if any(lvl.band[0] <= price <= lvl.band[1] for lvl in levels):
            continue

        within = [c for c in open_candidates
                  if abs(price - c[1]) < tech_width * c[1] and c[2] == kind]
        if within:
            matched = min(within, key=lambda c: abs(price - c[1]))
            band = (min(matched[1], price), max(matched[1], price))
            levels.append(Level(band=band, birth_date=date))
            open_candidates.remove(matched)
        else:
            open_candidates.append((date, price, kind))

    return levels


def score_touches(level: Level, close: pd.Series, volume: pd.Series = None):
    """Classify every post-birth approach to level.band as a hit (bounced, exited
    the same side it entered) or a miss (passed through, exited the other side).

    The birth touch itself sits exactly on the band edge, so it starts "inside"
    the band but isn't itself scored -- only genuine round-trips after that count.
    An excursion still open when the series ends is dropped (neither hit nor miss).

    If volume is given, each resolved touch also records the volume z-score
    (relative to the stock's whole history, same normalization find_touches
    uses) on the *resolving* date -- the breakout day for a miss, the
    bounce-back day for a hit -- as context on whether the market was paying
    unusual attention when the level was tested. This doesn't affect hit/miss
    classification at all, it's just annotated alongside it. Note the touch's
    reported date (touch_date downstream) stays entry-anchored, unchanged from
    before -- the volume z-score can therefore refer to a later calendar date
    than touch_date itself when an excursion takes several days to resolve.

    Mutates level.touches in place: [(date, outcome, volume_zscore), ...].
    """
    band_low, band_high = level.band
    post = close.loc[close.index > level.birth_date]
    volume_z = (volume - volume.mean()) / volume.std() if volume is not None else None

    def inside(p):
        return band_low <= p <= band_high

    touches = []
    in_band = True   # the founding touch is on the boundary
    entry_date = None
    enter_side = None
    prev_price = close.loc[level.birth_date]

    for date, price in post.items():
        if in_band:
            if not inside(price):
                exit_side = 'below' if price < band_low else 'above'
                if entry_date is not None:
                    outcome = 'hit' if exit_side == enter_side else 'miss'
                    vol_z = float(volume_z.loc[date]) if volume_z is not None else None
                    touches.append((entry_date, outcome, vol_z))
                in_band = False
                entry_date = None
                enter_side = None
        else:
            if inside(price):
                enter_side = 'below' if prev_price < band_low else 'above'
                entry_date = date
                in_band = True
        prev_price = price

    level.touches = touches


def mark_broken(levels: list, close: pd.Series) -> None:
    """Flag each level as broken the first time price fully crosses its band
    after birth -- i.e. once seen strictly above band_high AND, at some later
    date, strictly below band_low (or vice versa). A level that has only ever
    been approached and bounced (or is still being tested) is left broken_at
    = None.

    This is deliberately separate from score_touches's hit/miss bookkeeping:
    hit/miss counts every round trip for validating the detector, including
    ones after a level has already been crossed once, because that's exactly
    the data the grid search needs. This function instead answers a different
    question -- "is this band still a live support/resistance candidate for a
    fresh decision made today" -- which callers like the naive-entry strategy
    and live-trading display want, but the touch-scoring pipeline does not.
    Mutates levels in place.
    """
    for level in levels:
        band_low, band_high = level.band
        post = close.loc[close.index > level.birth_date]
        side = None
        for date, price in post.items():
            if price > band_high:
                cur = 'above'
            elif price < band_low:
                cur = 'below'
            else:
                continue
            if side is None:
                side = cur
            elif cur != side:
                level.broken_at = date
                break


def combo_key(combo: dict) -> str:
    """Canonical string id for a param combo, for grouping/joining the log."""
    return "|".join(f"{k}={combo[k]}" for k in sorted(combo))


def levels_to_rows(ticker, combo: dict, levels: list) -> list:
    """Turn already-scored Level objects into log rows for one (stock, combo).

    Levels with zero post-birth touches are still logged (outcome=None) so level
    counts per combo stay honest. combo should be the *full* param dict (already
    including defaults for consider_volume/volume_height/volume_prominence).
    """
    cid = combo_key(combo)
    rows = []
    for i, level in enumerate(levels):
        base_row = {
            'stock': ticker, 'combo_id': cid, **combo,
            'level_id': f'{ticker}_{cid}_{i}', 'band_low': level.band[0], 'band_high': level.band[1],
            'birth_date': level.birth_date,
        }
        if level.touches:
            for touch_date, outcome, volume_zscore in level.touches:
                rows.append({**base_row, 'touch_date': touch_date, 'outcome': outcome,
                             'touch_volume_zscore': volume_zscore})
        else:
            rows.append({**base_row, 'touch_date': pd.NaT, 'outcome': None, 'touch_volume_zscore': None})
    return rows


def run_combo_for_stock(ticker, close: pd.Series, combo: dict, volume: pd.Series = None) -> pd.DataFrame:
    """Run one (stock, param combo) pair end to end and return its log rows.

    combo must include 'tech_width'; may include 'consider_volume', 'volume_height',
    'volume_prominence', and any scipy.signal.find_peaks kwargs (prominence, distance,
    height, rel_height, ...). Convenience wrapper around find_touches/build_levels_causal/
    score_touches/levels_to_rows for a single combo -- see tech_level_search.py for the
    grid-search driver, which reuses those pieces directly to avoid re-running find_touches
    for every tech_width value.
    """
    combo = dict(combo)
    tech_width = combo.pop('tech_width')
    consider_volume = combo.pop('consider_volume', False)
    volume_height = combo.pop('volume_height', 0.0)
    volume_prominence = combo.pop('volume_prominence', 0.0)
    peak_kwargs = combo  # whatever's left goes straight to find_peaks

    full_combo = {
        'tech_width': tech_width,
        'consider_volume': consider_volume,
        'volume_height': volume_height,
        'volume_prominence': volume_prominence,
        **peak_kwargs,
    }

    touches = find_touches(close, volume=volume, consider_volume=consider_volume,
                            volume_height=volume_height, volume_prominence=volume_prominence,
                            **peak_kwargs)
    levels = build_levels_causal(touches, tech_width)
    for level in levels:
        score_touches(level, close, volume=volume)

    return pd.DataFrame(levels_to_rows(ticker, full_combo, levels))
