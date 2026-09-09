"""Event-level P&L for technical-level bounce trades.

The grid search and scoring measured whether price *bounces* at a level
(direction only). Nothing measured what a bounce is worth. A 69% hit rate loses
money if bounces average +1% and passthroughs average -4%, and there is a
structural reason to expect exactly that shape: a bounce is bounded (price turns
near the band and you exit modestly) while a passthrough is a breakout, which is
by construction a large trending move. This module computes the missing number.

Trade rule (exit on confirmation, stop out on passthrough):
  - enter at the close of the day price enters a known level's band
  - exit at the close of the day price leaves the band on the side it came from
    (the bounce is confirmed -- a 'hit' in score_touches terms)
  - stop out at the close of the day price leaves through the far side
    (a 'miss') -- this is the loss case

Only *support* tests are tradeable in a long-only book: price falls into the band
from above and is expected to bounce back up. A resistance test (price rises into
the band from below, expected to be rejected) would need a short, and from an
equal-weight base the most you can express is dropping the name to zero weight,
about -2.5% relative with 40 stocks. Both are recorded here; `side='support'` is
the actionable subset.

The benchmark matters as much as the P&L. Against an equal-weight book, an
overweight only pays if the stock beats *the rest of the portfolio* over the
holding window -- not if it merely goes up. `excess_ret` is the trade return
minus the equal-weight universe return over the identical window, and that is
the column to judge, not `ret`.
"""

import numpy as np
import pandas as pd

from tech_levels import find_touches, build_levels_causal


def trade_events(close: pd.Series, combo: dict, volume: pd.Series = None,
                 hold_days: int = 0) -> pd.DataFrame:
    """Every band entry/exit round trip for one (stock, combo), as trade records.

    Mirrors score_touches' walk, but keeps what a P&L needs and score_touches
    discards: the entry date, the resolution date, and which side price came
    from. Returns one row per resolved excursion; excursions still open when the
    series ends are dropped, exactly as in score_touches.

    hold_days: extra trading days to hold *after* a bounce is confirmed. With
    hold_days=0 the trade exits the moment price leaves the band, which on a
    0.3%-wide band means exiting at the very start of the move -- median holding
    period comes out at one day and the trade can only ever capture the band
    width. Holding on lets the bounce develop. Stop-outs are unaffected: a
    passthrough still exits immediately at the close it breaks through, since
    the thesis is dead at that point. Trades whose extended exit would fall past
    the end of the series are dropped rather than truncated, to avoid biasing
    the sample toward whatever the last few days happened to do.
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
    n = len(idx)
    pos = {d: i for i, d in enumerate(idx)}

    rows = []
    for level in levels:
        band_low, band_high = level.band
        post = close.loc[close.index > level.birth_date]

        in_band = True          # the founding touch sits on the boundary
        entered = False
        enter_side = None
        entry_date = None
        prev_price = close.loc[level.birth_date]

        for date, price in post.items():
            if in_band:
                if not (band_low <= price <= band_high):
                    exit_side = 'below' if price < band_low else 'above'
                    if entered:
                        outcome = 'hit' if exit_side == enter_side else 'miss'
                        # a confirmed bounce is held on; a stop-out is not
                        exit_pos = pos[date] + (hold_days if outcome == 'hit' else 0)
                        if exit_pos < n:
                            entry_price = float(close.loc[entry_date])
                            exit_date = idx[exit_pos]
                            exit_price = float(close.iloc[exit_pos])
                            rows.append({
                                'birth_date': level.birth_date,
                                'band_low': band_low, 'band_high': band_high,
                                'entry_date': entry_date, 'exit_date': exit_date,
                                'confirm_date': date,
                                # price fell into the band from above -> support test
                                'side': 'support' if enter_side == 'above' else 'resistance',
                                'outcome': outcome,
                                'entry_price': entry_price, 'exit_price': exit_price,
                                'ret': exit_price / entry_price - 1.0,
                            })
                    in_band = False
                    entered = False
                    enter_side = None
                    entry_date = None
            else:
                if band_low <= price <= band_high:
                    enter_side = 'below' if prev_price < band_low else 'above'
                    entry_date = date
                    entered = True
                    in_band = True
            prev_price = price

    if not rows:
        return pd.DataFrame(columns=['birth_date', 'band_low', 'band_high', 'entry_date',
                                     'exit_date', 'confirm_date', 'side', 'outcome',
                                     'entry_price', 'exit_price', 'ret'])
    return pd.DataFrame(rows)


def _equal_weight_curve(stock_list, close_col='close'):
    """Equal-weight cumulative index over the universe -- the benchmark a
    long-only deviation has to beat."""
    rets = {}
    for tkr in stock_list.tickers:
        df = stock_list[tkr].data
        c = pd.Series(df[close_col].values, index=pd.to_datetime(df.index)).dropna()
        rets[tkr] = c.pct_change()
    R = pd.DataFrame(rets).sort_index()
    return (1 + R.mean(axis=1).fillna(0)).cumprod()


def evaluate(stock_list, combos, cost_bps=10.0, hold_days=0, close_col='close', volume_col='volume'):
    """Trade records for every (ticker, combo), benchmarked and costed.

    cost_bps: round-trip cost in basis points, subtracted from every trade.
    Returns the per-trade frame; summarise() aggregates it.
    """
    ew = _equal_weight_curve(stock_list, close_col=close_col)

    frames = []
    for tkr, combo in combos.items():
        if tkr not in stock_list.tickers:
            continue
        df = stock_list[tkr].data
        close = pd.Series(df[close_col].values, index=pd.to_datetime(df.index)).dropna()
        volume = None
        if volume_col in df.columns:
            volume = pd.Series(df[volume_col].values, index=pd.to_datetime(df.index)).reindex(close.index)

        t = trade_events(close, combo, volume=volume, hold_days=hold_days)
        if t.empty:
            continue
        t['stock'] = tkr
        t['holding_days'] = (pd.DatetimeIndex(t.exit_date) - pd.DatetimeIndex(t.entry_date)).days

        bench = ew.reindex(pd.DatetimeIndex(t.exit_date)).to_numpy() / \
                ew.reindex(pd.DatetimeIndex(t.entry_date)).to_numpy() - 1.0
        t['bench_ret'] = bench
        t['ret_net'] = t['ret'] - cost_bps / 1e4
        t['excess_ret'] = t['ret_net'] - t['bench_ret']
        frames.append(t)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise(trades: pd.DataFrame, by='stock', side='support'):
    """Per-stock expected value, split by outcome -- the payoff asymmetry check."""
    t = trades[trades.side == side] if side else trades
    out = []
    for key, g in t.groupby(by):
        hits, misses = g[g.outcome == 'hit'], g[g.outcome == 'miss']
        out.append({
            by: key, 'n': len(g), 'hit_rate': len(hits) / len(g) if len(g) else np.nan,
            'avg_gain_hit': hits.excess_ret.mean(), 'avg_loss_miss': misses.excess_ret.mean(),
            'ev_excess': g.excess_ret.mean(),
            't_stat': g.excess_ret.mean() / g.excess_ret.std() * np.sqrt(len(g)) if len(g) > 2 else np.nan,
            'median_days': g.holding_days.median(),
            'total_excess': g.excess_ret.sum(),
        })
    return pd.DataFrame(out).sort_values('ev_excess', ascending=False)
