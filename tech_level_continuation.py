"""Forward-looking check on misses: did price continue in the breakout
direction, or revert back toward the level?

Conditioned on touch_volume_zscore (tech_levels.py's score_touches), this
tests the hypothesis that a high-volume passthrough reflects real conviction
(the level was being watched, it just didn't hold) while a low-volume one is
more likely a fakeout that reverts. Built entirely on top of the existing
touch log + raw close series -- doesn't change level construction or the
hit/miss definition at all.
"""

import pandas as pd
import numpy as np


def add_forward_returns(log: pd.DataFrame, stock_list, n_days: int) -> pd.DataFrame:
    """For every miss in the log, compute the n_days-forward return in the
    breakout direction: positive = continuation (price kept moving away from
    the level), negative = reversion (price moved back). NaN where there
    isn't enough subsequent data to measure it. Returns the miss rows with
    'direction' and 'forward_return' columns added.
    """
    misses = log[log['outcome'] == 'miss'].copy()
    misses['touch_date'] = pd.to_datetime(misses['touch_date']).dt.date

    frames = []
    for ticker, group in misses.groupby('stock'):
        close = stock_list[ticker].data['close']
        forward_price = close.shift(-n_days)

        price_at_touch = close.reindex(group['touch_date']).to_numpy()
        price_forward = forward_price.reindex(group['touch_date']).to_numpy()

        direction = np.where(price_at_touch > group['band_high'].to_numpy(), 1.0, -1.0)
        forward_return = direction * (price_forward - price_at_touch) / price_at_touch

        group = group.assign(direction=direction, forward_return=forward_return)
        frames.append(group)

    return pd.concat(frames, ignore_index=True) if frames else misses.assign(direction=[], forward_return=[])


def compare_by_volume(misses_with_returns: pd.DataFrame, volume_col: str = 'touch_volume_zscore') -> pd.DataFrame:
    """Split misses into high/low volume by a pooled median split on volume_col
    (already a stock-normalized z-score, so a pooled split is reasonable rather
    than per-stock), and compare forward_return between the two groups -- the
    direct test of whether high-volume passthroughs behave differently from
    low-volume ones.
    """
    df = misses_with_returns.dropna(subset=['forward_return', volume_col])
    median_vol = df[volume_col].median()
    df = df.assign(volume_bucket=np.where(df[volume_col] >= median_vol, 'high_volume', 'low_volume'))
    return df.groupby('volume_bucket')['forward_return'].agg(['mean', 'median', 'std', 'count'])
