"""Turn causal technical levels into per-date model features.

Bridges the level work (tech_levels.py + the grid search / scoring that picked a
winning combo per stock) into the forecasting pipeline: for one (stock, combo),
walk the price history and emit, for every date, where price sits relative to
the levels that were already known *on that date*.

Causality rules followed here:
  - a level only exists from its birth_date onward (build_levels_causal already
    freezes bands at birth, so the band itself carries no lookahead)
  - a touch only counts toward a level's touch count / hit rate from the date
    the outcome actually resolved -- not from touch_date, which is
    entry-anchored and can precede resolution by several days (see the note in
    score_touches' docstring)

Verified causal by truncation: recomputing features on the first 70% of AAPL's
history reproduces the full-history values exactly on every shared date except
the last 4 before the truncation point. That residual is a detection-lag
boundary effect, not a leak -- find_peaks cannot confirm a local extremum until
price moves away from it, and `distance` bars of separation are required on top.
Harmless for training (every training date sits far from the series edge), but
it does mean the newest ~`distance` days of features are provisional and can be
revised as data arrives, which matters for live inference.

(This check originally failed on all 16 columns across most of the history,
which is what surfaced the absolute-vs-fractional prominence bug in
find_touches; see its docstring in tech_levels.py.)

Companion to sma_to_input/rsi_calc in stock_class.py -- add_level_features
writes tl_* columns straight into stock.data, so create_long_x_y (forecast.py)
picks them up with no further plumbing.
"""

import numpy as np
import pandas as pd

from tech_levels import find_touches, build_levels_causal


# Distance stand-in for "there is no level on this side at all". 1.0 = a level
# 100% away, i.e. far outside any real band distance, so a tree can split it off
# cleanly. Paired with the tl_has_below / tl_has_above indicators.
NO_LEVEL_DISTANCE = 1.0

FEATURE_COLUMNS = [
    'tl_inside', 'tl_n_levels',
    'tl_has_below', 'tl_dist_below', 'tl_below_touches', 'tl_below_hit_rate',
    'tl_below_age', 'tl_below_width', 'tl_below_days_since_touch',
    'tl_has_above', 'tl_dist_above', 'tl_above_touches', 'tl_above_hit_rate',
    'tl_above_age', 'tl_above_width', 'tl_above_days_since_touch',
]


def resolved_touches(level, close: pd.Series):
    """[(resolve_date, outcome), ...] for one level.

    Mirrors score_touches' band-crossing walk exactly, but keys each touch on the
    date the excursion *resolved* (the day price exits the band) instead of the
    day it entered. score_touches reports entry dates because that's what the
    charts and hit-rate scoring want; features need the date the outcome became
    knowable, which is the exit. Same events either way -- verified by comparing
    hit/miss counts against score_touches on real data.
    """
    band_low, band_high = level.band
    post = close.loc[close.index > level.birth_date]

    events = []
    in_band = True          # the founding touch sits on the boundary
    entered = False         # has a genuine post-birth re-entry happened
    enter_side = None
    prev_price = close.loc[level.birth_date]

    for date, price in post.items():
        if in_band:
            if not (band_low <= price <= band_high):
                exit_side = 'below' if price < band_low else 'above'
                if entered:
                    events.append((date, 'hit' if exit_side == enter_side else 'miss'))
                in_band = False
                entered = False
                enter_side = None
        else:
            if band_low <= price <= band_high:
                enter_side = 'below' if prev_price < band_low else 'above'
                entered = True
                in_band = True
        prev_price = price

    return events


def _level_state(level, close: pd.Series, shrink=1.0):
    """Per-date running state for one level, aligned to close.index.

    Returns (n_touches, hit_rate, days_since_touch) as numpy arrays. hit_rate is
    shrunk toward 0.5 -- (hits + shrink) / (n + 2*shrink) -- because a raw rate
    off 1-2 touches is mostly noise, and levels with few touches are exactly the
    common case. shrink=0 gives the unsmoothed rate.
    """
    idx = close.index

    hits = pd.Series(0.0, index=idx)
    total = pd.Series(0.0, index=idx)
    last_touch = pd.Series(pd.NaT, index=idx, dtype='datetime64[ns]')

    for date, outcome in resolved_touches(level, close):
        total.loc[date] += 1.0
        if outcome == 'hit':
            hits.loc[date] += 1.0
        last_touch.loc[date] = pd.Timestamp(date)

    n_touches = total.cumsum().to_numpy()
    n_hits = hits.cumsum().to_numpy()
    hit_rate = (n_hits + shrink) / (n_touches + 2 * shrink)

    # days since the last resolved touch; before any touch, fall back to age.
    # Dates are coerced here rather than at the caller: stock.data comes off CSV
    # via StockList.load_data with a plain object index, so close.index can't be
    # assumed to be a DatetimeIndex, and the returned frame has to stay aligned
    # to whatever index the caller passed in.
    reference = last_touch.ffill().fillna(pd.Timestamp(level.birth_date))
    days_since = (pd.DatetimeIndex(idx) - pd.DatetimeIndex(reference)).days.to_numpy().astype(float)

    return n_touches, hit_rate, days_since


def level_features(close: pd.Series, combo: dict, volume: pd.Series = None,
                   shrink=1.0):
    """Per-date technical-level features for one stock under one param combo.

    close must be a date-indexed price Series. combo is the same dict shape the
    grid search / live script use (tech_width plus find_peaks kwargs and the
    optional volume flags). Returns a DataFrame indexed like close with the
    FEATURE_COLUMNS above.
    """
    combo = dict(combo)
    tech_width = combo.pop('tech_width')
    consider_volume = combo.pop('consider_volume', False)
    volume_height = combo.pop('volume_height', 0.0)
    volume_prominence = combo.pop('volume_prominence', 0.0)

    touches = find_touches(close, volume=volume, consider_volume=consider_volume,
                           volume_height=volume_height, volume_prominence=volume_prominence,
                           **combo)
    levels = build_levels_causal(touches, tech_width)

    idx = close.index
    close_arr = close.to_numpy(dtype=float)
    out = np.full((len(idx), len(FEATURE_COLUMNS)), np.nan)

    if not levels:
        frame = pd.DataFrame(out, index=idx, columns=FEATURE_COLUMNS)
        frame['tl_inside'] = 0.0
        frame['tl_n_levels'] = 0.0
        frame['tl_has_below'] = 0.0
        frame['tl_has_above'] = 0.0
        frame['tl_dist_below'] = NO_LEVEL_DISTANCE
        frame['tl_dist_above'] = NO_LEVEL_DISTANCE
        return frame.fillna(0.0)

    pos = {d: i for i, d in enumerate(idx)}
    states = []
    for level in levels:
        n_touches, hit_rate, days_since = _level_state(level, close, shrink=shrink)
        states.append({
            'birth_i': pos[level.birth_date],
            'low': level.band[0],
            'high': level.band[1],
            'width': (level.band[1] - level.band[0]) / max(level.band[1], 1e-12),
            'n_touches': n_touches,
            'hit_rate': hit_rate,
            'days_since': days_since,
        })
    states.sort(key=lambda s: s['birth_i'])

    active = []
    next_level = 0
    rows = []
    for i in range(len(idx)):
        while next_level < len(states) and states[next_level]['birth_i'] <= i:
            active.append(states[next_level])
            next_level += 1

        price = close_arr[i]
        inside = 0.0
        best_below = None
        best_above = None
        for st in active:
            if st['low'] <= price <= st['high']:
                inside = 1.0
            elif st['high'] < price:
                if best_below is None or st['high'] > best_below['high']:
                    best_below = st
            else:
                if best_above is None or st['low'] < best_above['low']:
                    best_above = st

        row = {'tl_inside': inside, 'tl_n_levels': float(len(active))}
        for side, st in (('below', best_below), ('above', best_above)):
            if st is None:
                row.update({
                    f'tl_has_{side}': 0.0,
                    f'tl_dist_{side}': NO_LEVEL_DISTANCE,
                    f'tl_{side}_touches': 0.0,
                    f'tl_{side}_hit_rate': 0.5,
                    f'tl_{side}_age': 0.0,
                    f'tl_{side}_width': 0.0,
                    f'tl_{side}_days_since_touch': 0.0,
                })
            else:
                edge = st['high'] if side == 'below' else st['low']
                row.update({
                    f'tl_has_{side}': 1.0,
                    f'tl_dist_{side}': abs(price - edge) / price,
                    f'tl_{side}_touches': float(st['n_touches'][i]),
                    f'tl_{side}_hit_rate': float(st['hit_rate'][i]),
                    f'tl_{side}_age': float(i - st['birth_i']),
                    f'tl_{side}_width': st['width'],
                    f'tl_{side}_days_since_touch': float(st['days_since'][i]),
                })
        rows.append(row)

    return pd.DataFrame(rows, index=idx, columns=FEATURE_COLUMNS)


def select_combos(log, selection_cutoff, min_touches=20):
    """Pick each stock's combo, and a confidence weight, using only pre-cutoff data.

    Returns ({ticker: combo}, {ticker: confidence}) ready for add_level_features.

    Scores on in-sample excess (local_is - global_loo_is) at selection_cutoff, so
    every number behind the choice comes from touches at or before that date.
    This matters more than it looks: the 4-cutoff *out-of-sample* excess used to
    shortlist DHR/MMM is computed over the whole history, so selecting on it and
    then training a forecasting model on earlier dates would leak the future into
    the panel -- the same class of mistake as the prominence-scaling bug, just
    one level up. The cost is that in-sample excess is weaker evidence than the
    OOS work; treat the resulting confidence as a relative weight across stocks,
    not as a validated per-stock edge.

    Callers must start the forecasting panel *after* selection_cutoff for this to
    hold (build_panel in forecast.py enforces it).

    min_touches: minimum in-sample touches a (stock, combo) needs to be eligible.
        Guards the finding-#3 trap where a 3-for-3 sample looks like a perfect combo.
    """
    from tech_level_scoring import score

    scored = score(log, pd.Timestamp(selection_cutoff))
    scored['excess'] = scored['local_is'] - scored['global_loo_is']
    eligible = scored[scored['n_touches_is'] >= min_touches]
    if eligible.empty:
        raise ValueError(f'no (stock, combo) pair has >= {min_touches} in-sample touches '
                         f'at {selection_cutoff}')

    best = (eligible.reset_index()
            .sort_values('excess', ascending=False)
            .groupby('stock').first())

    param_cols = ['tech_width', 'prominence', 'distance', 'consider_volume',
                  'volume_height', 'volume_prominence']
    params = log.drop_duplicates('combo_id').set_index('combo_id')[param_cols]

    combos, confidence = {}, {}
    for stock, row in best.iterrows():
        p = params.loc[row['combo_id']]
        combos[stock] = {
            'tech_width': float(p['tech_width']),
            'prominence': float(p['prominence']),
            'distance': int(p['distance']),
            'consider_volume': bool(p['consider_volume']),
            'volume_height': float(p['volume_height']),
            'volume_prominence': float(p['volume_prominence']),
        }
        # clipped at 0: a negative-excess combo carries no evidence of edge, and a
        # negative weight would invite the model to read the features backwards
        confidence[stock] = max(0.0, float(row['excess']))
    return combos, confidence


def add_level_features(stock_list, combos, confidence=None, shrink=1.0,
                       close_col='close', volume_col='volume'):
    """Write tl_* feature columns into each stock's .data, in place.

    combos: {ticker: combo dict} -- the per-stock winning combo (same shape as
        data/live_combos.json). Tickers absent from combos get neutral features
        (no levels) rather than being dropped, so the pooled panel stays
        rectangular; tl_confidence is what tells the model to discount them.
    confidence: optional {ticker: float}, written as tl_confidence -- intended
        for the validated excess-over-baseline OOS score, so the model can learn
        where the level signal is actually trustworthy instead of averaging its
        usefulness across a universe where most names showed no edge. Missing
        tickers get 0.0.
    """
    for tkr in stock_list.tickers:
        df = stock_list[tkr].data
        close = df[close_col].dropna()
        volume = df[volume_col] if volume_col in df.columns else None
        if volume is not None:
            volume = volume.loc[close.index]

        combo = combos.get(tkr)
        if combo is None:
            feats = pd.DataFrame(0.0, index=close.index, columns=FEATURE_COLUMNS)
            feats['tl_dist_below'] = NO_LEVEL_DISTANCE
            feats['tl_dist_above'] = NO_LEVEL_DISTANCE
            feats['tl_below_hit_rate'] = 0.5
            feats['tl_above_hit_rate'] = 0.5
        else:
            feats = level_features(close, combo, volume=volume, shrink=shrink)

        for col in FEATURE_COLUMNS:
            df[col] = feats[col].reindex(df.index)
        df['tl_confidence'] = float((confidence or {}).get(tkr, 0.0))

    return stock_list
