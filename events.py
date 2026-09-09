"""Event-study harness: judge any sparse signal by the same yardstick.

The premise of the signal-driven design is that edge is *sparse* -- absent on
most days, present on a few. Averaging a forecast over every day is then the
wrong estimator, and conditioning on "the signal fired" is the right one. But
conditioning buys effect size by spending sample size, so the estimator has to
be honest about how little data it rests on. That is what this module is for.

Every signal produces the same object: a `fires` frame with columns
`stock`, `fire_date`, `side`, `strength`. This module turns fires into trades,
benchmarks them, and then tries to break the result three ways:

  1. **Excess over equal weight, not raw return.** From an equal-weight base,
     overweighting a name only pays if it beats *the rest of the book* over the
     holding window. A signal that fires in bull markets will look wonderful on
     raw returns and add nothing.
  2. **Block bootstrap, not a naive t-stat.** Trades overlap in time and cluster
     (a shock day fires several names at once), so the i.i.d. standard error is
     too small. Resampling calendar blocks preserves both.
  3. **Placebo with matched count and holding rule.** Random entries on the same
     stocks, same number of trades, same exit rule. This is what caught an
     apparent technical-level edge that was really one name's drift.

Entry is at the close of the day *after* the fire (`entry_lag=1`). The signal is
computed from a close, the alert goes out after the close, and the trade happens
the next session. Entering at the fire-day close would be free lookahead.
"""

import numpy as np
import pandas as pd

FIRE_COLS = ['stock', 'fire_date', 'side', 'strength']


def equal_weight_curve(stock_list, close_col='close'):
    """Equal-weight cumulative index over the universe -- the benchmark a
    long-only deviation has to beat."""
    rets = {}
    for tkr in stock_list.tickers:
        df = stock_list[tkr].data
        c = pd.Series(df[close_col].values, index=pd.to_datetime(df.index)).dropna()
        rets[tkr] = c.pct_change()
    R = pd.DataFrame(rets).sort_index()
    return (1 + R.mean(axis=1).fillna(0)).cumprod()


def _stock_arrays(stock_list, ew, close_col='close'):
    """Per-ticker (dates, closes, benchmark level) as numpy, built once and
    reused across every placebo repetition."""
    out = {}
    for tkr in stock_list.tickers:
        df = stock_list[tkr].data
        c = pd.Series(df[close_col].values, index=pd.to_datetime(df.index)).dropna()
        b = ew.reindex(c.index).ffill()
        out[tkr] = (c.index, c.to_numpy(dtype=float), b.to_numpy(dtype=float))
    return out


def _simulate_one(closes, bench, fire_pos, sign, horizon, entry_lag,
                  take_profit, stop_loss):
    """Resolve one fire into (entry_off, exit_off, ret, bench_ret, reason).

    Returns None if the full holding window would run past the end of the
    series -- truncating instead would bias the sample toward whatever the last
    few days happened to do.
    """
    n = len(closes)
    entry = fire_pos + entry_lag
    if entry + horizon >= n:
        return None

    path = closes[entry + 1: entry + horizon + 1] / closes[entry] - 1.0
    pnl = sign * path

    exit_off = horizon - 1
    reason = 'time'
    tp_hit = np.flatnonzero(pnl >= take_profit) if take_profit is not None else np.array([], int)
    sl_hit = np.flatnonzero(pnl <= stop_loss) if stop_loss is not None else np.array([], int)
    first_tp = tp_hit[0] if tp_hit.size else np.inf
    first_sl = sl_hit[0] if sl_hit.size else np.inf
    if min(first_tp, first_sl) < exit_off:
        # both triggered on the same close: assume the stop, the conservative read
        exit_off = int(min(first_tp, first_sl))
        reason = 'stop' if first_sl <= first_tp else 'target'

    ex = entry + 1 + exit_off
    ret = sign * (closes[ex] / closes[entry] - 1.0)
    bench_ret = sign * (bench[ex] / bench[entry] - 1.0)
    return entry, ex, ret, bench_ret, reason


def build_trades(stock_list, fires, horizon=10, take_profit=None, stop_loss=None,
                 cost_bps=10.0, entry_lag=1, close_col='close', arrays=None, ew=None):
    """Turn a fires frame into benchmarked, costed trade records.

    take_profit / stop_loss are decimal returns measured from the entry close and
    checked on closes only (no intraday). `excess_ret` is the column to judge.
    """
    if ew is None:
        ew = equal_weight_curve(stock_list, close_col=close_col)
    if arrays is None:
        arrays = _stock_arrays(stock_list, ew, close_col=close_col)

    rows = []
    for tkr, g in fires.groupby('stock'):
        if tkr not in arrays:
            continue
        dates, closes, bench = arrays[tkr]
        pos = {d: i for i, d in enumerate(dates)}
        for rec in g.itertuples():
            fp = pos.get(pd.Timestamp(rec.fire_date))
            if fp is None:
                continue
            sign = -1.0 if rec.side == 'short' else 1.0
            res = _simulate_one(closes, bench, fp, sign, horizon, entry_lag,
                                take_profit, stop_loss)
            if res is None:
                continue
            entry, ex, ret, bench_ret, reason = res
            rows.append({
                'stock': tkr, 'fire_date': dates[fp], 'side': rec.side,
                'strength': getattr(rec, 'strength', np.nan),
                'entry_date': dates[entry], 'exit_date': dates[ex],
                'entry_price': closes[entry], 'exit_price': closes[ex],
                'holding_days': ex - entry, 'exit_reason': reason,
                'ret': ret, 'bench_ret': bench_ret,
                'ret_net': ret - cost_bps / 1e4,
                'excess_ret': ret - cost_bps / 1e4 - bench_ret,
            })

    if not rows:
        return pd.DataFrame(columns=['stock', 'fire_date', 'side', 'strength', 'entry_date',
                                     'exit_date', 'entry_price', 'exit_price', 'holding_days',
                                     'exit_reason', 'ret', 'bench_ret', 'ret_net', 'excess_ret'])
    return pd.DataFrame(rows).sort_values('entry_date').reset_index(drop=True)


def pooled_stats(trades, col='excess_ret'):
    """Headline numbers. `t_naive` is deliberately named -- see block_bootstrap."""
    x = trades[col].to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return {'n': 0}
    sd = x.std(ddof=1) if n > 1 else np.nan
    return {
        'n': n,
        'n_stocks': trades.stock.nunique(),
        'ev_excess': x.mean(),
        'sd': sd,
        'se_naive': sd / np.sqrt(n) if n > 1 else np.nan,
        't_naive': x.mean() / sd * np.sqrt(n) if n > 1 and sd > 0 else np.nan,
        'win_rate': float((x > 0).mean()),
        'median_days': trades.holding_days.median(),
        'ann_contrib_5pp': 0.05 * x.mean() * n / _years(trades),
    }


def _years(trades):
    d = pd.DatetimeIndex(trades.entry_date)
    return max((d.max() - d.min()).days / 365.25, 1e-9)


def block_bootstrap(trades, col='excess_ret', block='Q', n_boot=2000, seed=0):
    """Resample whole calendar blocks of trades, not individual trades.

    Overlapping holding windows and same-day clustering across stocks both make
    trades dependent, so an i.i.d. standard error understates the true one. A
    block resample keeps trades that share a period together and lets the
    dependence show up in the spread.
    """
    x = trades[[col]].copy()
    x['blk'] = pd.DatetimeIndex(trades.entry_date).to_period(block)
    groups = [g[col].to_numpy(dtype=float) for _, g in x.groupby('blk')]
    if len(groups) < 2:
        return {'n_blocks': len(groups)}

    rng = np.random.default_rng(seed)
    k = len(groups)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        means[b] = np.concatenate([groups[i] for i in pick]).mean()

    obs = x[col].to_numpy(dtype=float).mean()
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        'n_blocks': k, 'ev_excess': obs, 'se_block': means.std(ddof=1),
        'ci_lo': float(np.quantile(means, 0.025)), 'ci_hi': float(np.quantile(means, 0.975)),
        'p_block': float(min(p, 1.0)),
        't_block': obs / means.std(ddof=1) if means.std(ddof=1) > 0 else np.nan,
    }


def placebo(stock_list, fires, n_reps=500, seed=0, horizon=10, take_profit=None,
            stop_loss=None, cost_bps=10.0, entry_lag=1, close_col='close',
            arrays=None, ew=None):
    """Random entries matched on stock, trade count, and exit rule.

    The stock is *preserved* and only the date is randomized, which is the
    version that matters: it asks whether the trades beat simply holding an
    overweight in those same names at arbitrary times. A signal that is really
    just one stock's drift dies here.

    Dates are drawn from each stock's own observed fire window, so a signal that
    only ever fires in one regime is compared against that regime.
    """
    if ew is None:
        ew = equal_weight_curve(stock_list, close_col=close_col)
    if arrays is None:
        arrays = _stock_arrays(stock_list, ew, close_col=close_col)

    plan = []
    for tkr, g in fires.groupby('stock'):
        if tkr not in arrays:
            continue
        dates, closes, bench = arrays[tkr]
        lo = dates.searchsorted(pd.Timestamp(g.fire_date.min()))
        hi = min(dates.searchsorted(pd.Timestamp(g.fire_date.max())),
                 len(dates) - horizon - entry_lag - 1)
        if hi <= lo:
            continue
        signs = np.where(g.side.to_numpy() == 'short', -1.0, 1.0)
        plan.append((tkr, closes, bench, lo, hi, signs))

    rng = np.random.default_rng(seed)
    means = np.empty(n_reps)
    for b in range(n_reps):
        vals = []
        for tkr, closes, bench, lo, hi, signs in plan:
            draws = rng.integers(lo, hi, len(signs))
            for fp, sign in zip(draws, signs):
                res = _simulate_one(closes, bench, int(fp), sign, horizon,
                                    entry_lag, take_profit, stop_loss)
                if res is not None:
                    vals.append(res[2] - cost_bps / 1e4 - res[3])
        means[b] = np.mean(vals) if vals else np.nan
    return means


def placebo_report(trades, placebo_means, col='excess_ret'):
    obs = trades[col].mean()
    m = placebo_means[~np.isnan(placebo_means)]
    return {
        'observed': obs, 'placebo_mean': m.mean(), 'placebo_sd': m.std(ddof=1),
        'edge_over_placebo': obs - m.mean(),
        'p_placebo': float((m >= obs).mean()),
    }


def by_period(trades, col='excess_ret', freq='Y'):
    """Per-year EV and count -- the stability check a pooled mean hides."""
    g = trades.assign(period=pd.DatetimeIndex(trades.entry_date).to_period(freq))
    return g.groupby('period').agg(n=(col, 'size'), ev=(col, 'mean'),
                                   win=(col, lambda s: (s > 0).mean()))


def full_report(stock_list, fires, horizon=10, take_profit=None, stop_loss=None,
                cost_bps=10.0, entry_lag=1, n_placebo=500, close_col='close',
                arrays=None, ew=None, seed=0):
    """Everything at once: trades, pooled stats, block bootstrap, placebo, per-year."""
    if ew is None:
        ew = equal_weight_curve(stock_list, close_col=close_col)
    if arrays is None:
        arrays = _stock_arrays(stock_list, ew, close_col=close_col)

    kw = dict(horizon=horizon, take_profit=take_profit, stop_loss=stop_loss,
              cost_bps=cost_bps, entry_lag=entry_lag, close_col=close_col,
              arrays=arrays, ew=ew)
    trades = build_trades(stock_list, fires, **kw)
    if trades.empty:
        return {'trades': trades, 'stats': {'n': 0}}

    out = {'trades': trades, 'stats': pooled_stats(trades),
           'boot': block_bootstrap(trades, seed=seed),
           'by_year': by_period(trades)}
    if n_placebo:
        pm = placebo(stock_list, fires, n_reps=n_placebo, seed=seed, **kw)
        out['placebo'] = placebo_report(trades, pm)
    return out
