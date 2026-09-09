"""Stage A: cross-sectional return forecast.

The panel that came out of the technical-level work carried only own-stock
features (rsi_*, sma_cross_*, variance_global) and scored a mean cross-sectional
IC of about zero. That is the expected result: nothing in it told the model what
the rest of the universe was doing on a given day, so it had no way to express
"this stock is cheap *relative to its peers*", which is the only question a
long-only selection decision actually asks.

This module adds the two families that were missing:

  - time-series predictors with a track record in the cross-sectional equity
    literature: momentum (12-1), short-horizon reversal, realized vol
  - genuinely cross-sectional views of those: each stock's within-date
    percentile rank across the universe, plus market aggregates and dispersion

The rank transform is the important one. A tree splitting on raw momentum learns
thresholds ("mom_252 > 0.3") that mean different things in different regimes; a
tree splitting on within-date rank learns "top decile of the universe today",
which is regime-invariant and is the quantity a long-only book can act on.

Target is the vol-scaled forward return from forecast.build_panel. Both the raw
and the cross-sectionally demeaned variants are supported: demeaning removes the
market component, which is close to unforecastable and which a long-only book
inherits whether or not it is predicted, but it also discards the only signal
that could ever say "hold cash instead". Which is better is an empirical
question -- evaluate() runs both.
"""

import numpy as np
import pandas as pd

# lookbacks in trading days
MOM_WINDOWS = [21, 63, 126, 252]
REV_WINDOWS = [5, 10]
RV_WINDOWS = [5, 22, 66]
EPS = 1e-8

# columns converted to within-date cross-sectional percentile ranks
RANK_COLS = ['a_mom_21', 'a_mom_63', 'a_mom_126', 'a_mom_252', 'a_mom_12_1',
             'a_rev_5', 'a_rev_10', 'a_rv_5', 'a_rv_22', 'a_rv_ratio',
             'a_beta', 'rsi_14', 'rsi_50']


def market_return(stock_list, return_col='log_return'):
    """Equal-weight universe log return -- the common component every stock shares."""
    cols = {}
    for tkr in stock_list.tickers:
        df = stock_list[tkr].data
        cols[tkr] = pd.Series(df[return_col].values, index=pd.to_datetime(df.index))
    return pd.DataFrame(cols).mean(axis=1)


def add_stock_features(stock_list, mkt=None, return_col='log_return'):
    """Write a_* time-series features into each stock's .data, in place.

    Follows the sma_to_input / rsi_calc convention in stock_class.py so the
    columns are picked up by create_long_x_y with no extra plumbing.
    """
    if mkt is None:
        mkt = market_return(stock_list, return_col=return_col)

    for tkr in stock_list.tickers:
        df = stock_list[tkr].data
        idx = pd.to_datetime(df.index)
        r = pd.Series(df[return_col].values, index=idx)

        f = pd.DataFrame(index=idx)
        for w in MOM_WINDOWS:
            f[f'a_mom_{w}'] = r.rolling(w).sum()
        # 12-1 momentum: the classic construction, skipping the most recent month
        # because short-horizon reversal runs the opposite way and would cancel it
        f['a_mom_12_1'] = f['a_mom_252'] - f['a_mom_21']
        for w in REV_WINDOWS:
            f[f'a_rev_{w}'] = r.rolling(w).sum()
        for w in RV_WINDOWS:
            f[f'a_rv_{w}'] = np.log(np.sqrt((r ** 2).rolling(w).mean()) + EPS)
        f['a_rv_ratio'] = f['a_rv_5'] - f['a_rv_22']

        # rolling beta to the equal-weight market: a stock characteristic the
        # model can generalize across, unlike the one-hot stock identity which
        # only ever describes one ticker
        m = mkt.reindex(idx)
        cov = r.rolling(252).cov(m)
        var = m.rolling(252).var()
        f['a_beta'] = cov / var.replace(0, np.nan)
        f['a_resid_mom_63'] = f['a_mom_63'] - f['a_beta'] * m.rolling(63).sum()

        for c in f.columns:
            df[c] = f[c].reindex(idx).values

    return stock_list


def add_cross_sectional(X, rank_cols=None):
    """Add within-date percentile ranks, market aggregates and dispersion.

    Computed on the assembled panel rather than per stock, because every one of
    these needs the whole universe on a given date.
    """
    rank_cols = [c for c in (rank_cols or RANK_COLS) if c in X.columns]
    X = X.copy()

    ranks = X.groupby(level=0)[rank_cols].rank(pct=True)
    ranks.columns = [f'{c}_xrank' for c in rank_cols]
    X = pd.concat([X, ranks], axis=1)

    # market state: what the universe as a whole is doing today
    for c in ['a_mom_21', 'a_mom_252', 'a_rv_22']:
        if c in X.columns:
            X[f'{c}_mkt'] = X.groupby(level=0)[c].transform('mean')
            # own value relative to the market -- the demeaned version of the
            # same quantity, which is what a relative-value decision needs
            X[f'{c}_rel'] = X[c] - X[f'{c}_mkt']
    # cross-sectional dispersion: how much opportunity there is to differentiate
    if 'a_mom_21' in X.columns:
        X['xs_dispersion'] = X.groupby(level=0)['a_mom_21'].transform('std')
    return X


def composite_score(X):
    """Stage-A prediction: a zero-parameter rank composite. Nothing is fitted.

    Long-term momentum rank minus intermediate-horizon momentum rank. Both terms
    carry the sign the equity literature assigns them -- 12-month momentum is
    positive, 3-month momentum reverses -- and neither is estimated from this
    data, which is the point.

    This is the stage-A model because every fitted alternative lost to it:

        RF, min_samples_leaf=10            IC -0.0155
        RF, min_samples_leaf=1000          IC +0.0076
        Ridge, alpha=1                     IC -0.0046
        Ridge, alpha=10000                 IC +0.0099
        this composite                     IC +0.0392

    Both model families improved monotonically as regularization increased,
    which says the panel supports far less capacity than an RF has. Raw
    single-feature ICs bear that out: the strongest feature in the whole set
    reaches only 0.031, so there was never enough signal for a flexible learner
    to find without fitting noise. Note the composite is measured with no
    train/test split because it estimates nothing -- there is no parameter that
    could have leaked.

    The two legs are complementary rather than merely averaged: long-term
    momentum alone is negative in fold 1 (-0.003) and intermediate reversal
    alone is negative in fold 4 (-0.013), while the combination is positive in
    all four (+0.048/+0.027/+0.032/+0.051).

    **The +0.039 above is stale -- do not quote it.** Two corrections, both from
    the 2026-07-21 out-of-sample work; `signals_notes.md` has the detail.

    1. *Overlap.* A 10-day target means consecutive daily ICs share 9 of 10 days
       of outcome, so mean/std*sqrt(n_dates) over ~2540 dates is badly inflated.
       Newey-West at lag 10, or keeping every 10th date, cuts the effective
       sample to ~254 and the t-stat with it. Recomputed on the same 40 names:
       full period IC +0.0113, t_NW **+0.89**; post-2021 IC +0.0306, t_NW
       **+1.85**. Positive in only 5 of 11 calendar years. The partial 2026 year
       carries much of it -- dropping it takes the post-2021 figure to +0.0205.

    2. *Fresh cross-section.* On the 60 tickers of `config.oos_ticker_list`,
       which had no part in choosing the legs: full period IC **+0.0135**
       (t_NW +1.30), post-2021 **+0.0250** (t_NW +1.75), positive in 7 of 11
       years. Sign and magnitude both hold -- this is the check that killed
       `signals.idio_shock`, and the composite passes it.

    So: not refuted, and free of the ticker-selection pathology, but not
    demonstrated either -- nothing here reaches conventional significance once
    the overlap is handled. Note also that the two universes' daily IC series
    correlate 0.61, so they are not two independent confirmations; the OOS test
    controls for ticker selection, not for period.

    This stays the stage-A model on the strength of its prior and its zero
    parameters, not its t-stat. **Plan on IC ~ +0.015, not +0.039.**

    Standing caveat on the legs: they were picked after looking at full-sample
    single-feature ICs on the 40, which included the evaluation windows. The
    defence is that momentum and intermediate-horizon reversal are among the most
    replicated effects in the cross-sectional equity literature, and the signs
    came from there rather than being fitted. Which leg carries is also not
    stable: mom252 dominates on the 40 (+0.0139 vs +0.0061), the reversal leg
    dominates on the 60 (+0.0132 vs +0.0068). Both are positive in both
    universes, which is the argument for keeping the pair rather than either one.
    """
    return (X['a_mom_252_xrank'] - X['a_mom_63_xrank']).rename('score')


def build_alpha_panel(stock_list, horizon=10, selection_cutoff='2021-06-01',
                      include_tl=False, demean_target=False):
    """Full stage-A panel: alpha features + cross-sectional views + vol-scaled target."""
    from forecast import build_panel

    mkt = market_return(stock_list)
    add_stock_features(stock_list, mkt=mkt)

    X, Y = build_panel(stock_list, selection_cutoff=selection_cutoff,
                       horizon=horizon, vol_scale=True)
    if not include_tl:
        X = X.drop(columns=[c for c in X.columns if c.startswith('tl_')])

    X = add_cross_sectional(X)
    ok = X.notna().all(axis=1)
    X, Y = X[ok], Y[ok]

    if demean_target:
        # predict return relative to the universe that day rather than outright
        Y = Y - Y.groupby(level=0).transform('mean')

    return X, Y
