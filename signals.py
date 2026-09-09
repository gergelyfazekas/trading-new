"""Signal registry. Each signal is a scan that returns a `fires` frame.

Contract: `scan(stock_list, **params) -> DataFrame[stock, fire_date, side, strength]`.
Nothing here is fitted and nothing here trades; `events.full_report` judges the
output, and every signal is judged the same way so none gets a bespoke metric
that flatters it.

Causality rule for anything added here: a fire on date t may use data up to and
including t's close, and nothing after. Normalizers get an extra `.shift(1)` so
the event day is not inside its own denominator.
"""

import numpy as np
import pandas as pd

from alpha import market_return


def dedupe_fires(fires, min_gap=10):
    """Drop repeat fires on the same stock within min_gap trading rows.

    A multi-day selloff otherwise emits a fire every day and the "independent
    events" count becomes fiction -- inflating n and shrinking the standard
    error while adding no information.
    """
    keep = []
    for tkr, g in fires.sort_values('fire_date').groupby('stock'):
        last = None
        for rec in g.itertuples():
            d = pd.Timestamp(rec.fire_date)
            if last is None or (d - last).days >= min_gap * 7 / 5:
                keep.append(rec.Index)
                last = d
    return fires.loc[sorted(keep)].reset_index(drop=True)


def idio_shock(stock_list, z_threshold=2.5, resid_vol_window=66, beta_window=252,
               market_z_max=1.0, min_gap=10, return_col='log_return', side='avoid'):
    """Fire when a stock makes a large *idiosyncratic* one-day move.

    Sign of `z_threshold` picks the arm: positive fires on up-shocks, negative on
    down-shocks. The two words doing the work are:

      *idiosyncratic* -- the move is measured as the residual to the equal-weight
      market, r_i - beta_i * r_m, and the fire is vetoed when the market itself
      moved more than `market_z_max` sigma. A stock that moved because everything
      moved is not a dislocation, and from an equal-weight base there is nothing
      to exploit in it anyway -- the rest of the book moved too.

      *large* -- the residual is scaled by its own trailing volatility, so the
      threshold means the same thing for a utility and a high-beta tech name, and
      the same thing in 2017 and in March 2020. `.shift(1)` on that volatility
      keeps the shock day out of its own denominator, where it would inflate
      sigma and partly hide itself.

    **REFUTED -- do not trade this.** Kept as a worked example for the harness and
    as a record of how a signal can pass every in-sample check and still be noise.

    Built to test short-horizon reversal after a large idiosyncratic *drop*. That
    arm is flat: -0.09% over 10 days, t = -0.55, at every threshold and horizon.

    The up arm looked spectacular in sample -- after a +2.5 sigma idiosyncratic
    gain the stock underperformed equal weight by 0.60% over 10 days, t = -4.1
    naive and -4.7 block-bootstrapped, beating a 400-rep placebo unanimously,
    negative in all 11 calendar years, negative for 31 of 40 names, and stable to
    within 10bp under leave-one-stock-out. On 60 tickers outside the original 40,
    with every parameter frozen, it came back **+0.135%, t = +0.85** -- opposite
    sign, no significance, and 15 of 20 grid cells flipped with it. Splitting the
    new universe by return and by volatility puts every subgroup at ~0, so it is
    not a composition effect. See `signals_notes.md`.

    The lesson that survived: in-sample robustness checks test robustness within
    the sample, not generalisation to new assets. Only a fresh cross-section
    caught this -- a held-out time window would have confirmed the false result,
    since it was stable across all 11 years.

    `side='avoid'` reflects the long-only constraint: the actionable trade for a
    negative view is dropping the name to zero weight, not shorting it.
    `events.build_trades` treats 'avoid' like a long, so `excess_ret` is the
    return being declined -- a negative number would mean the overlay is working.
    """
    mkt = market_return(stock_list, return_col=return_col)
    mkt_z = mkt / mkt.rolling(beta_window).std().shift(1)

    rows = []
    for tkr in stock_list.tickers:
        df = stock_list[tkr].data
        idx = pd.to_datetime(df.index)
        r = pd.Series(df[return_col].values, index=idx).astype(float)
        m = mkt.reindex(idx)

        beta = r.rolling(beta_window).cov(m) / m.rolling(beta_window).var().replace(0, np.nan)
        resid = r - beta * m
        sigma = resid.rolling(resid_vol_window).std().shift(1)
        z = resid / sigma.replace(0, np.nan)

        hit = (z >= z_threshold) if z_threshold >= 0 else (z <= z_threshold)
        fire = (hit & (mkt_z.reindex(idx).abs() <= market_z_max)).fillna(False)
        for d in idx[fire.to_numpy()]:
            rows.append({'stock': tkr, 'fire_date': d, 'side': side,
                         'strength': float(abs(z.loc[d]))})

    if not rows:
        return pd.DataFrame(columns=['stock', 'fire_date', 'side', 'strength'])
    return dedupe_fires(pd.DataFrame(rows), min_gap=min_gap)


def efficiency_ratio(r, window=20):
    """|net displacement| / total path length over `window` days, through t.

    Near 1 = clean trend, near 0 = chop. Causal, one parameter, and a direct
    measurement of "is this price going somewhere" rather than a proxy for it.
    """
    return r.rolling(window).sum().abs() / r.abs().rolling(window).sum().replace(0, np.nan)


def reversal_in_trend(stock_list, top_n=3, rev_window=5, er_window=20,
                      er_min_rank=0.5, min_gap=10, return_col='log_return',
                      side='long'):
    """Buy the biggest short-horizon losers, but only among *trending* names.

    Each date: rank the universe by efficiency ratio, keep the top half, and fire
    on the `top_n` largest `rev_window`-day decliners within it. Repeat fires on a
    name inside `min_gap` days are deduped, so a name that stays cheap does not
    count as several independent events.

    **Why the filter runs this way round.** The original hypothesis was the
    opposite -- that support/resistance and mean reversion should work in *quiet,
    non-trending* names. That was tested first on a high-powered instrument and
    refuted: reversal IC was +0.0016 in the choppy half against +0.0136 in the
    trending half, with the sign negative in 8 of 8 cells. Taken to the 60-ticker
    validation universe as a single pre-specified test, the conditioning effect
    replicated in sign and magnitude (-0.0102 vs -0.0121 in sample) and all 12
    cells came back negative. Reversal restricted to trending names scored
    IC +0.0212 (t_NW +2.35) out of sample.

    Note the effect is *conditional only*: the efficiency ratio has no standalone
    cross-sectional IC (+0.0000 out of sample, against +0.0139 in sample). It says
    nothing about which names go up -- only about where reversion is worth
    trusting.

    **The IC does not convert into a trade rule -- do not trade this either.**
    Run through `events.py` at the pre-specified top_n=3 / hold 10 / 10bps, the
    excess EV is -0.0001 (t_block -0.08) on the 40 and +0.0017 (+1.88) on the 60.
    Zero on the universe the effect was found on. On the 40 the ER filter is
    actually *worse* than no filter (-0.0001 vs +0.0005), and the best top_n
    disagrees between universes (1 vs 2). See `signals_notes.md`.

    Two reasons it failed, both general:
      - cross-sectional IC is a rank correlation over the *whole* universe, while
        a top-N rule only touches the tail. An IC of +0.02 is consistent with a
        well-ordered middle and a noisy tail.
      - at 280-310 fires/yr and 10-day holds this carries ~12 concurrent
        positions. That is a continuously rebalanced tilt, not a sparse alert --
        the wrong shape for the design regardless of its EV.

    Caveat carried from `signals_notes.md`: the conditioning effect is not
    individually significant in either universe (t ~ -1.1, -1.3), only its sign
    replicated, and the two universes' IC series correlate 0.61 -- one
    confirmation, not two.
    """
    rev, er = {}, {}
    for tkr in stock_list.tickers:
        df = stock_list[tkr].data
        idx = pd.to_datetime(df.index)
        r = pd.Series(df[return_col].values, index=idx).astype(float)
        rev[tkr] = -r.rolling(rev_window).sum()
        er[tkr] = efficiency_ratio(r, er_window)

    REV = pd.DataFrame(rev).sort_index()
    ER = pd.DataFrame(er).sort_index().reindex(REV.index)
    eligible = ER.rank(axis=1, pct=True) > er_min_rank
    scored = REV.where(eligible)

    rows = []
    for date, row in scored.iterrows():
        row = row.dropna()
        if len(row) < top_n:
            continue
        for tkr, v in row.nlargest(top_n).items():
            rows.append({'stock': tkr, 'fire_date': date, 'side': side,
                         'strength': float(v)})

    if not rows:
        return pd.DataFrame(columns=['stock', 'fire_date', 'side', 'strength'])
    return dedupe_fires(pd.DataFrame(rows), min_gap=min_gap)


def shock_reversal(stock_list, z_threshold=-3.0, **kw):
    """The original down-shock reversal hypothesis. Kept because it was tested
    and refuted, not because it is usable -- see `idio_shock`."""
    return idio_shock(stock_list, z_threshold=z_threshold, side='long', **kw)
