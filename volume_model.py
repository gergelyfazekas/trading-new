"""Forward trading-volume model: how predictable are high-volume periods?

A pure predictability study, not a trading rule. The question is how accurately
the *next* h-day stretch of trading volume can be called ahead of time, and --
more importantly -- how much of that accuracy is available from two things that
cost nothing: the calendar, and the fact that volume lately was high.

Everything below is walk-forward, 4 folds, purged by `horizon * 1.5` days,
developed on the 40 names of `config.ticker_list` and confirmed once, with every
choice frozen, on the disjoint 60 of `config.oos_ticker_list`. Full tables in
`report_volume.md`.

---------------------------------------------------------------------------
HEADLINE
---------------------------------------------------------------------------
Volume is genuinely and substantially predictable -- far more so than returns,
and on a par with volatility. At a 10-day horizon the full model reaches
**R^2 = 0.28** on forward normalised log volume (0.30 on the OOS 60), against
0.21 for a four-parameter linear HAR cascade and 0.15 for a calibrated
one-parameter persistence model.

But the honest headline is the decomposition, and it has two parts that both
cut against the obvious stories:

  1. **It is not the calendar.** Calendar-only R^2 is 0.06 at h=10 and the
     calendar adds ~0.05 on top of the volume model (RF 0.284 with, 0.232
     without). Real, worth having, not the story. In the *cross-sectional*
     problem the calendar is worth almost exactly nothing (AUC 0.51) -- expiry
     and month-end lift the whole tape at once, so they carry no information
     about which name will be busy.
  2. **It is not volatility either.** A pure trailing-realized-vol model gets
     R^2 0.08 at h=10, and regressing the full forecast on the vol-only forecast
     leaves R^2 = 0.24 -- roughly three quarters of the volume forecast is
     *not* a restatement of the vol forecast. Volume's own history is the
     dominant input: an RF given realized vol and the calendar but no volume
     history scores 0.11 against the full model's 0.28.

The thing that does most of the work is the trailing volume cascade, and the
serious caveat is that raw persistence has to be *shrunk*: carrying last
period's volume forward at full weight scores **negative** R^2 at h >= 5.

---------------------------------------------------------------------------
Design decisions, and why
---------------------------------------------------------------------------

**Never raw share volume.** Share volume has a strong secular trend (AAPL's mean
log volume falls from 19.0 in 2015 to 17.7 in 2026) and differs across names by
orders of magnitude. A raw level in a pooled panel is a stock-identity label
plus a clock; `NON_STATIONARY_COLS` in `forecast.py` exists because that already
destroyed one model in this repo. Everything here is

    nvol_t = log(volume_t) - log(median(volume_{t-65..t}))

-- log for the right skew, trailing 66-day median for the level, median rather
than mean because a single earnings print is 3-8x normal volume and would drag a
rolling mean for a whole quarter, making the normaliser a lagged copy of the
spikes being predicted. The normaliser at t uses only data at or before t.

**The target's normaliser is frozen at t, not at t+h.** The target is

    y_t = mean(log volume_{t+1..t+h}) - log(median(volume_{t-65..t}))

a genuinely forward quantity measured against a base an observer knows at t.
Normalising the forward window by its own contemporaneous median would have
hidden most of the thing being predicted inside the denominator.

**Calendar features describe the forward window, not the current day.** Exchange
holiday schedules and expiry dates are published years ahead, so knowing that
t+1..t+h contains a triple-witching Friday is legitimately free information.
Each flag enters as the *fraction of the forward window* carrying it. Holiday
adjacency is derived from the trading calendar itself (a missing business day),
not a hard-coded holiday list, so it transfers to any date range.

The calendar effects that survive, largest first (h=5, log units, per unit
fraction of the window): triple witching +0.86, turn-of-year -0.32, month-end
+0.32, holiday-adjacent -0.19, Monday -0.17, plain monthly opex +0.13. Note two
of the six are *negative* -- the holiday and year-end lulls are as much of the
calendar signal as the expiry spikes.

**Earnings are a known unmodelled driver and the proxy failed.** No earnings
dates were available. The proxy is `days_since_spike` projected forward into
`earnings_due`, the fraction of the forward window landing 55-75 trading days
after the last spike. It is causal (the counter at t uses only spikes at or
before t; the projection is arithmetic). Its OLS coefficient is **+0.001 at
h=5 and -0.009 at h=1 -- indistinguishable from zero.** Spike-dating is too
imprecise to recover the quarterly cycle. Real earnings dates would be the
single most valuable addition to this model and are not in it.

**Causality checked by truncation, not inspection.** Rebuilding the panel on
history truncated at 2022-06-30 reproduces all 32 features, the target, and the
expanding time-series label bit-for-bit on every shared row (max abs difference
0.0, 65839 rows, 0 label disagreements).

---------------------------------------------------------------------------
RESULTS -- regression, mean R^2 over 4 purged folds (the 40 / the OOS 60)
---------------------------------------------------------------------------
                      h=1        h=5         h=10        h=20
  rw (no shrink)     0.131    -0.028/-0.03 -0.118/-0.13 -0.186/-0.19
  rw + fitted slope  0.313     0.210/0.210  0.147/0.149  0.088/0.092
  calendar only      0.110     0.084/0.079  0.057/0.055  0.031/0.029
  HAR cascade        0.333     0.268/0.271  0.210/0.214  0.158/0.162
  HAR + calendar     0.414     0.343/0.336  0.271/0.269  0.194/0.194
  realized vol only  0.101     0.094/0.087  0.081/0.077  0.069/0.067
  RF, everything     0.436     0.363/0.362  0.284/0.297  0.189/0.221

Three things to take from this table:

  * **The naive random walk is worse than the unconditional mean at every
    horizon past a day.** Normalised volume mean-reverts hard; the single most
    important modelling act is shrinking persistence, not adding features.
  * **The RF beats HAR+calendar by 0.02, exactly as in `volatility.py`.** Most
    of the predictability is linear. At h=20 on the 40 the RF is *behind*
    HAR+calendar (0.189 vs 0.194) -- it only wins there on the OOS 60.
  * **The mean hides a large regime effect.** RF at h=10 by fold: 0.217
    (2018-20), 0.423 (2020-22), 0.299 (2022-24), **0.197 (2024-26)**. The
    covid fold flatters every number here and the most recent fold is the
    weakest. Plan on the low end.

---------------------------------------------------------------------------
RESULTS -- classification, "is the next h days a high-volume period?"
---------------------------------------------------------------------------
Two label definitions, both evaluated:
  * `cross_sectional` -- top quintile of the universe on that date. No lookahead
    is possible: a within-date comparison of the same forward window for all
    names.
  * `time_series` -- top quintile of that stock's *own* history, with an
    expanding 80th-percentile threshold built only from targets whose forward
    window closed at or before t.

At h=10, base rate 0.20, mean over folds (the 40 / the OOS 60):

                 cross-sectional              own-history
                 AUC    prec@10%  lift        AUC    prec@10%  lift
  calendar only  0.509/0.510  0.22  1.11      0.621/0.626  0.31  1.55
  persistence    0.618/0.627  0.40  1.99      0.675/0.668  0.46  2.24
  RF, everything 0.689/0.696  0.49  2.45      0.717/0.739  0.52  2.56

So: **a top-decile call is right about half the time against a 20% base rate --
a lift of ~2.5x** -- and a free persistence rule already gets 2.0-2.2x of that.
The model's contribution over knowing nothing but the current volume level is
roughly +0.07 AUC and +0.4x lift.

The calendar's split personality is the most interesting result here: worthless
cross-sectionally (AUC 0.509) and clearly useful in the time-series problem
(AUC 0.62). Market-wide scheduled events tell you *when* the tape is busy, never
*which stock*.

Calibration is good for the cross-sectional label and for own-history at
h<=10: the RF's probability deciles track realised frequencies to within a few
points, and Brier beats the base-rate constant (0.144 vs 0.160 at h=10). It is
**bad for own-history at h=20**, where the top bins are badly over-confident
(predicted 0.84, realised 0.39). That is the expanding-threshold drift showing
up exactly where it was expected -- see the negatives.

**Transition vs persistence -- the expectation was refuted, in a specific
way.** Splitting the test set on whether the stock is currently in a high-volume
state (trailing 22-day nvol above the training folds' 80th percentile):

  h=10, cross-sectional     base rate   AUC     prec@10%   lift
    currently high            0.296    0.721      0.70      2.37
    currently normal          0.174    0.660      0.37      2.12

Relative skill barely degrades: AUC falls 0.06 and lift falls 0.25x. *Absolute*
accuracy nearly halves (precision 0.70 -> 0.37), but almost all of that is the
base rate falling from 0.30 to 0.17. The prior expectation -- that the headline
is nothing but persistence and transitions are near-unpredictable -- is **not
supported**. The model retains ~2.1x lift on stocks that are *not* currently
busy. Transitions into high volume are genuinely, if modestly, forecastable.

---------------------------------------------------------------------------
Negatives, stated plainly
---------------------------------------------------------------------------
  * **Naive persistence loses to the unconditional mean** (negative R^2) at
    every horizon >= 5 days. Anyone quoting "volume is persistent" without
    shrinkage is quoting a model that would lose money on a mean-squared bet.
  * The RF's edge over a four-parameter linear HAR + calendar is ~0.02 R^2 at
    every horizon, and at h=20 on the 40 it is negative. The nonlinearity is
    not where the value is.
  * The most recent walk-forward fold (2024-2026) is the weakest at every
    horizon and roughly half the covid fold. The reported means are optimistic
    relative to what the current regime supports.
  * Predictability decays fast in the horizon: RF R^2 0.44 -> 0.36 -> 0.28 ->
    0.19 for h = 1, 5, 10, 20. Twenty-day volume is not well forecast.
  * The own-history label degrades out of sample at h=20 and its probabilities
    are miscalibrated there, because an expanding quantile on a series with a
    secular trend is not a stationary threshold. Use the cross-sectional label
    if you need calibrated probabilities at long horizons.
  * The earnings proxy contributes nothing measurable. Earnings are a
    first-order driver of exactly the tail this study is about and the model
    does not see them.
  * The calendar-only story that would have been the tidy headline is not
    supported: 0.06 R^2 at h=10, and zero cross-sectional content.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPS = 1e-8

# Trailing windows for the volume cascade -- daily, weekly, monthly, quarterly.
# Same shape as the HAR cascade in volatility.py, for the same reason: volume
# has long-memory persistence that four trailing averages capture almost fully.
VOL_WINDOWS = [1, 5, 22, 66]

# Window for the level normaliser. 66 days = one quarter, long enough to be a
# stable level estimate and short enough to track the secular trend out.
BASE_WINDOW = 66

# nvol above this counts as a "spike" for the earnings-cycle proxy. 0.6 in logs
# = volume ~82% above its own trailing median.
SPIKE_THRESHOLD = 0.6

HAR_COLS = ['nvol_1', 'nvol_5', 'nvol_22', 'nvol_66']

CALENDAR_COLS = [
    'cal_opex', 'cal_triple_witch', 'cal_month_end', 'cal_quarter_end',
    'cal_holiday_adj', 'cal_turn_of_year', 'cal_mon', 'cal_fri',
    'cal_month_sin', 'cal_month_cos', 'cal_summer', 'earnings_due',
]

# Realized-vol features, kept separate so the volume-vs-volatility overlap
# question can be answered by fitting on one group at a time.
RV_COLS = ['rv_5', 'rv_22', 'rv_66', 'rv_ratio_5_22', 'ret_22', 'neg_ret_share']

VOLUME_COLS = HAR_COLS + [
    'nvol_ratio_5_22', 'nvol_ratio_22_66', 'nvol_of_vol', 'nvol_max_22',
    'dollar_share', 'nvol_vs_market', 'market_nvol_5', 'market_nvol_22',
    'days_since_spike',
]


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def normalized_log_volume(volume: pd.Series, base_window: int = BASE_WINDOW):
    """Return (nvol, log_base): volume in logs, de-trended by its own past.

    log_base_t = log(rolling median of volume over the `base_window` days
    ending at and including t) -- causal, uses nothing after t.
    nvol_t = log(volume_t) - log_base_t.

    Median not mean: a single earnings-day print is 3-8x normal volume and would
    drag a rolling mean for a whole quarter, making the normaliser itself a
    (lagged) copy of the spikes we are trying to predict.
    """
    lv = np.log(volume.clip(lower=1.0))
    log_base = np.log(volume.clip(lower=1.0).rolling(base_window).median())
    return lv - log_base, log_base


def forward_norm_volume(volume: pd.Series, log_base: pd.Series, horizon: int) -> pd.Series:
    """Mean log volume over t+1..t+horizon, minus the log base known at t.

    Two separate causality points, both load-bearing:
      - the averaged window is strictly t+1..t+h, sharing no day with any
        trailing feature at t;
      - the normaliser is `log_base` *at t*, not over the forward window. Using
        the forward window's own level would put most of the target inside the
        denominator and make the problem look far easier than it is.
    """
    lv = np.log(volume.clip(lower=1.0))
    fwd = lv.rolling(horizon).mean().shift(-horizon)
    return fwd - log_base


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------

def daily_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-session calendar flags, all knowable years in advance.

    `cal_holiday_adj` is inferred from the trading calendar itself: a session is
    holiday-adjacent when a business day is missing immediately before or after
    it. No hard-coded holiday list, so it transfers to any date range, and the
    exchange calendar is public well ahead of time.
    """
    idx = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    c = pd.DataFrame(index=idx)

    dow = idx.dayofweek
    c['cal_mon'] = (dow == 0).astype(float)
    c['cal_fri'] = (dow == 4).astype(float)

    # third Friday = monthly options expiry; in Mar/Jun/Sep/Dec it is also
    # index-future/option expiry and the S&P quarterly rebalance ("triple
    # witching"), historically the largest scheduled volume day of the quarter.
    is_fri = dow == 4
    third_fri = is_fri & (idx.day >= 15) & (idx.day <= 21)
    c['cal_opex'] = third_fri.astype(float)
    c['cal_triple_witch'] = (third_fri & idx.month.isin([3, 6, 9, 12])).astype(float)

    ym = pd.Series(idx.to_period('M'), index=idx)
    last_of_month = ym != ym.shift(-1)
    c['cal_month_end'] = last_of_month.astype(float).values
    c['cal_quarter_end'] = (last_of_month.values & idx.month.isin([3, 6, 9, 12])).astype(float)

    full = pd.bdate_range(idx[0], idx[-1])
    missing = full.difference(idx)
    prev_missing = idx.map(lambda d: (d - pd.tseries.offsets.BDay(1)) in set(missing))
    next_missing = idx.map(lambda d: (d + pd.tseries.offsets.BDay(1)) in set(missing))
    c['cal_holiday_adj'] = (np.asarray(prev_missing) | np.asarray(next_missing)).astype(float)

    c['cal_turn_of_year'] = ((idx.month == 12) & (idx.day >= 20)).astype(float) \
        + ((idx.month == 1) & (idx.day <= 5)).astype(float)
    c['cal_summer'] = idx.month.isin([7, 8]).astype(float)
    c['cal_month_sin'] = np.sin(2 * np.pi * idx.month / 12)
    c['cal_month_cos'] = np.cos(2 * np.pi * idx.month / 12)
    return c


def forward_calendar(cal: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Aggregate each daily flag over the forward window t+1..t+horizon.

    Fraction of the window carrying the flag. This is the form that makes the
    calendar-only baseline a fair comparison: it is exactly the calendar
    information an observer at t has about the period being forecast, no more.
    """
    return cal.rolling(horizon).mean().shift(-horizon)


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

def _days_since_spike(nvol: pd.Series, threshold: float = SPIKE_THRESHOLD) -> pd.Series:
    """Trading days since the last volume spike, counted causally.

    Earnings proxy. A day at or above `threshold` in normalised log volume
    resets the counter; the value at t reflects only spikes at or before t.
    """
    spike = (nvol >= threshold).fillna(False).to_numpy()
    out = np.full(len(spike), np.nan)
    last = -1
    for i, s in enumerate(spike):
        if last >= 0:
            out[i] = i - last
        if s:
            last = i
    return pd.Series(out, index=nvol.index)


def build_volume_panel(stock_list, horizon=10, return_col='log_return',
                       base_window=BASE_WINDOW):
    """Long panel (one row per date-stock) of volume, volatility and calendar
    features with the forward normalised-log-volume target.

    Returns (X, y). X carries a 'stock' column for the one-hot, exactly as
    `volatility.build_vol_panel` does.
    """
    cal = None
    frames = []
    for tkr in stock_list.tickers:
        df = stock_list[tkr].data
        idx = pd.to_datetime(df.index)
        v = pd.Series(np.asarray(df['volume'], dtype=float), index=idx)
        r = pd.Series(np.asarray(df[return_col], dtype=float), index=idx)
        px = pd.Series(np.asarray(df['close'], dtype=float), index=idx)

        nvol, log_base = normalized_log_volume(v, base_window)

        f = pd.DataFrame(index=idx)
        for w in VOL_WINDOWS:
            f[f'nvol_{w}'] = nvol.rolling(w).mean()
        # term structure of volume: is activity picking up or dying off,
        # independent of the stock's overall level
        f['nvol_ratio_5_22'] = f['nvol_5'] - f['nvol_22']
        f['nvol_ratio_22_66'] = f['nvol_22'] - f['nvol_66']
        f['nvol_of_vol'] = nvol.rolling(22).std()
        # recent extreme: a single spike in the last month is information the
        # trailing means smooth away
        f['nvol_max_22'] = nvol.rolling(22).max()
        f['days_since_spike'] = _days_since_spike(nvol).clip(upper=250)

        # dollar volume relative to the universe -- a size/liquidity descriptor
        # that is a share (stationary) rather than a level
        f['dollar_vol'] = np.log((v * px).clip(lower=1.0))

        # realized volatility, kept as its own group for the overlap question
        for w in [5, 22, 66]:
            f[f'rv_{w}'] = np.log(np.sqrt((r ** 2).rolling(w).mean()) + EPS)
        f['rv_ratio_5_22'] = f['rv_5'] - f['rv_22']
        f['ret_22'] = r.rolling(22).sum()
        f['neg_ret_share'] = (r < 0).rolling(22).mean()

        if cal is None:
            cal = daily_calendar(idx)
        fcal = forward_calendar(cal, horizon).reindex(idx)
        for col in fcal.columns:
            f[col] = fcal[col]

        # earnings-cycle projection: fraction of the forward window that lands
        # 55-75 trading days after the last observed spike. Pure arithmetic on
        # the counter at t -- no forward observation involved.
        dss = f['days_since_spike']
        due = np.zeros(len(f))
        for k in range(1, horizon + 1):
            proj = dss + k
            due += ((proj >= 55) & (proj <= 75)).to_numpy(dtype=float)
        f['earnings_due'] = due / horizon

        f['target'] = forward_norm_volume(v, log_base, horizon)
        f['stock'] = tkr
        # persistence baseline: "the next h days look like the last h days",
        # measured against the same base as the target
        f['rw_pred'] = nvol.rolling(horizon).mean()
        frames.append(f)

    panel = pd.concat(frames).sort_index()

    # cross-sectional market activity factor. A pooled model with only own-stock
    # features cannot see that the entire tape is busy, which is a large part of
    # what moves any single name's volume.
    panel['market_nvol_5'] = panel.groupby(level=0)['nvol_5'].transform('mean')
    panel['market_nvol_22'] = panel.groupby(level=0)['nvol_22'].transform('mean')
    panel['nvol_vs_market'] = panel['nvol_22'] - panel['market_nvol_22']
    # dollar volume as a within-date share: stationary, unlike the level
    panel['dollar_share'] = panel['dollar_vol'] - panel.groupby(level=0)['dollar_vol'].transform('mean')
    panel = panel.drop(columns=['dollar_vol'])

    panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
    y = panel['target']
    X = panel.drop(columns=['target'])
    return X, y


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def cross_sectional_label(y: pd.Series, X: pd.DataFrame, q: float = 0.8) -> pd.Series:
    """1 if the stock is in the top (1-q) of the universe on that date.

    A within-date comparison of the same forward window across names. No
    causality question arises: nothing outside the labelled period is used, and
    at prediction time the model produces a score for every name on the date, so
    the comparison it is scored against is the one it can actually make.
    """
    df = pd.DataFrame({'y': y.values, 'stock': X['stock'].values}, index=y.index)
    pct = df.groupby([df.index])['y'].rank(pct=True)
    return (pct > q).astype(int)


def time_series_label(y: pd.Series, X: pd.DataFrame, horizon: int, q: float = 0.8,
                      min_history: int = 250) -> pd.Series:
    """1 if the forward window is in the top (1-q) of *that stock's own history*.

    The threshold is an expanding quantile that only sees targets whose forward
    window has already closed: at date t it uses targets dated t-horizon and
    earlier. Without that shift the threshold would be computed partly from the
    very observation being labelled -- the classic way this kind of label leaks.

    Returns NaN for the first `min_history` observations of each stock, where
    the expanding quantile is too noisy to mean anything.
    """
    out = []
    for tkr, grp in pd.DataFrame({'y': y.values, 'stock': X['stock'].values},
                                 index=y.index).groupby('stock'):
        s = grp['y'].sort_index()
        closed = s.shift(horizon)                       # targets fully observed by t
        thr = closed.expanding(min_periods=min_history).quantile(q)
        lab = (s > thr).astype(float)
        lab[thr.isna()] = np.nan
        out.append(pd.Series(lab.values, index=pd.MultiIndex.from_arrays(
            [s.index, [tkr] * len(s)], names=['date', 'stock'])))
    return pd.concat(out)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def build_volume_pipeline(kind='reg', n_estimators=300, min_samples_leaf=20,
                          max_features=0.5, use_stock=True):
    est = (RandomForestRegressor if kind == 'reg' else RandomForestClassifier)(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
        max_features=max_features, random_state=0, n_jobs=-1)
    if use_stock:
        pre = ColumnTransformer(
            [('stock_ohe', OneHotEncoder(handle_unknown='ignore'), ['stock'])],
            remainder='passthrough')
        return Pipeline([('preprocess', pre), ('rf', est)])
    return Pipeline([('rf', est)])


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def _folds(index, n_folds):
    dates = np.array(sorted(pd.DatetimeIndex(index).unique()))
    return np.array_split(dates, n_folds + 1)[1:]


# ---------------------------------------------------------------------------
# evaluation -- regression
# ---------------------------------------------------------------------------

def evaluate_walkforward(X, y, horizon, n_folds=4, feature_sets=None,
                         with_rf=True):
    """Walk-forward R^2 with the baselines reported alongside, not after.

    Models, in the order they matter:
      rw        -- random walk: next h days = last h days (zero parameters)
      rw_ols    -- the same single predictor, but with a fitted slope. Worth
                   reporting separately because raw `rw` scores *negative* R^2
                   at h >= 5: normalised volume mean-reverts, so carrying the
                   last period forward at full weight overshoots badly enough to
                   lose to the unconditional mean. The shrinkage is most of what
                   a persistence model needs.
      cal       -- calendar only: OLS on the forward-window calendar features
      har       -- OLS on the trailing nvol cascade (1/5/22/66)
      har_cal   -- HAR + calendar, the honest "everything cheap" benchmark
      rv_only   -- OLS on trailing realized volatility only, no volume input.
                   This is the volatility-overlap test: if it lands close to
                   har, volume prediction is largely a restatement of vol
                   prediction.
      rf        -- RF over all features including stock one-hot

    Each test fold is purged by `horizon * 1.5` days: the last training target
    before a fold spans days inside it, and without the gap the scores are
    optimistic.
    """
    sets = feature_sets or {
        'rw_ols': ['rw_pred'],
        'cal': CALENDAR_COLS,
        'rw_cal': ['rw_pred'] + CALENDAR_COLS,
        'har': HAR_COLS,
        'har_cal': HAR_COLS + CALENDAR_COLS,
        'rv_only': RV_COLS,
        'rv_cal': RV_COLS + CALENDAR_COLS,
        'har_rv': HAR_COLS + RV_COLS,
    }
    rows = []
    for f in _folds(X.index, n_folds):
        purge = f[0] - pd.Timedelta(days=int(horizon * 1.5))
        tr = X.index < purge
        te = (X.index >= f[0]) & (X.index <= f[-1])
        if tr.sum() < 5000:
            continue
        rec = {'test_start': f[0].date(), 'test_end': f[-1].date(),
               'n_test': int(te.sum()), 'r2_rw': _r2(y[te], X.loc[te, 'rw_pred'])}
        for name, cols in sets.items():
            cols = [c for c in cols if c in X.columns]
            m = LinearRegression().fit(X.loc[tr, cols], y[tr])
            rec[f'r2_{name}'] = _r2(y[te], m.predict(X.loc[te, cols]))
        if with_rf:
            pipe = build_volume_pipeline('reg').fit(X[tr], y[tr])
            rec['r2_rf'] = _r2(y[te], pipe.predict(X[te]))
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.loc['mean'] = out.mean(numeric_only=True)
    return out


# ---------------------------------------------------------------------------
# evaluation -- classification
# ---------------------------------------------------------------------------

def _clf_metrics(label, score, prob=None, top_frac=0.1):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    label = np.asarray(label, dtype=float)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(label) & np.isfinite(score)
    label, score = label[ok], score[ok]
    base = label.mean()
    k = max(1, int(round(top_frac * len(score))))
    top = np.argsort(-score)[:k]
    prec = label[top].mean()
    rec = label[top].sum() / max(label.sum(), 1)
    out = {'base_rate': base, 'auc': roc_auc_score(label, score) if 0 < base < 1 else np.nan,
           'prec_top10': prec, 'recall_top10': rec, 'lift_top10': prec / base if base > 0 else np.nan,
           'n': len(label)}
    if prob is not None:
        p = np.asarray(prob, dtype=float)[ok]
        out['brier'] = brier_score_loss(label, np.clip(p, 0, 1))
        out['brier_base'] = brier_score_loss(label, np.full_like(p, base))
    return out


def evaluate_classification(X, y, horizon, label_kind='cross_sectional',
                            n_folds=4, q=0.8, top_frac=0.1, state_split=True):
    """Walk-forward classification of "next h days is a high-volume period".

    Same purge as the regression. Scores compared:
      persist   -- the trailing h-day nvol, i.e. the current volume decile,
                   used directly as the score. Zero parameters.
      cal       -- logistic regression on forward-window calendar only
      rf        -- RF classifier on everything (also supplies probabilities,
                   so it is the only one with a calibration number)

    `state_split` additionally reports the RF's metrics separately for rows
    where the stock is currently in a high-volume state and rows where it is
    not (transition vs persistence). The split uses `nvol_22` against the
    training folds' 80th percentile, so it is causal.
    """
    if label_kind == 'cross_sectional':
        lab = cross_sectional_label(y, X, q=q)
        lab.index = pd.MultiIndex.from_arrays([y.index, X['stock'].values],
                                              names=['date', 'stock'])
    else:
        lab = time_series_label(y, X, horizon, q=q)
    key = pd.MultiIndex.from_arrays([X.index, X['stock'].values], names=['date', 'stock'])
    labels = lab.reindex(key).to_numpy(dtype=float)

    rows, calib = [], []
    for f in _folds(X.index, n_folds):
        purge = f[0] - pd.Timedelta(days=int(horizon * 1.5))
        tr = (X.index < purge) & np.isfinite(labels)
        te = (X.index >= f[0]) & (X.index <= f[-1]) & np.isfinite(labels)
        if tr.sum() < 5000 or te.sum() < 500:
            continue
        ytr = labels[tr]
        if ytr.min() == ytr.max():
            continue

        cal_cols = [c for c in CALENDAR_COLS if c in X.columns]
        lr = Pipeline([('sc', StandardScaler()),
                       ('lr', LogisticRegression(max_iter=2000))]).fit(X.loc[tr, cal_cols], ytr)
        s_cal = lr.predict_proba(X.loc[te, cal_cols])[:, 1]

        # Deliberately a coarser forest than the regressor: with h-day
        # overlapping targets, adjacent rows are ~1-1/h independent, so a leaf
        # of 20 rows is nowhere near 20 observations and the extra depth buys
        # variance, not signal. min_samples_leaf=100 also keeps the predicted
        # probabilities smooth enough for the calibration table to mean
        # something. Not tuned -- fixed before the OOS run and never revisited.
        clf = build_volume_pipeline('clf', n_estimators=150,
                                    min_samples_leaf=100).fit(X[tr], ytr)
        p_rf = clf.predict_proba(X[te])[:, 1]

        s_persist = X.loc[te, 'rw_pred'].to_numpy()
        yte = labels[te]

        for name, sc, pr in [('persist', s_persist, None), ('cal', s_cal, s_cal),
                             ('rf', p_rf, p_rf)]:
            m = _clf_metrics(yte, sc, pr, top_frac)
            m.update({'model': name, 'test_start': f[0].date(), 'subset': 'all'})
            rows.append(m)

        if state_split:
            thr = np.nanpercentile(X.loc[tr, 'nvol_22'], 100 * q)
            hi = X.loc[te, 'nvol_22'].to_numpy() >= thr
            for sub, mask in [('currently_high', hi), ('currently_normal', ~hi)]:
                if mask.sum() < 200 or np.nanstd(yte[mask]) == 0:
                    continue
                m = _clf_metrics(yte[mask], p_rf[mask], p_rf[mask], top_frac)
                m.update({'model': 'rf', 'test_start': f[0].date(), 'subset': sub})
                rows.append(m)
                m = _clf_metrics(yte[mask], s_persist[mask], None, top_frac)
                m.update({'model': 'persist', 'test_start': f[0].date(), 'subset': sub})
                rows.append(m)

        bins = np.clip((p_rf * 10).astype(int), 0, 9)
        c = pd.DataFrame({'bin': bins, 'p': p_rf, 'y': yte}).groupby('bin').agg(
            pred=('p', 'mean'), actual=('y', 'mean'), n=('y', 'size'))
        c['fold'] = str(f[0].date())
        calib.append(c.reset_index())

    res = pd.DataFrame(rows)
    summary = res.groupby(['subset', 'model']).mean(numeric_only=True)
    calib = pd.concat(calib).groupby('bin').apply(
        lambda g: pd.Series({'pred': np.average(g['pred'], weights=g['n']),
                             'actual': np.average(g['actual'], weights=g['n']),
                             'n': g['n'].sum()}), include_groups=False)
    return summary, calib


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def predict_high_volume(stock_list, horizon=10, cutoff=None, q=0.8,
                        label_kind='cross_sectional'):
    """Fit on history before `cutoff` and score every row at or after it.

    Returns a frame indexed by date with columns:
      stock, nvol_hat  -- predicted forward normalised log volume
      vol_hat          -- the same in share terms, relative to the trailing
                          66-day median, smearing-corrected (see below)
      p_high           -- probability the next `horizon` days are a high-volume
                          period under `label_kind`
      score_persist    -- the zero-parameter baseline, returned deliberately so
                          nobody consumes p_high without seeing what it beat

    Retransformation: the model is fit in logs, so exp(nvol_hat) is a conditional
    *median* volume ratio, not a mean -- low by roughly exp(s^2/2). The smearing
    factor is applied to `vol_hat` only; `nvol_hat` is left in logs.

    Training stops `horizon * 1.5` days before the cutoff so no training target
    overlaps the prediction period.
    """
    X, y = build_volume_panel(stock_list, horizon=horizon)
    cutoff = pd.Timestamp(cutoff) if cutoff is not None else X.index[int(0.7 * len(X))]
    tr = X.index < cutoff - pd.Timedelta(days=int(horizon * 1.5))
    te = X.index >= cutoff
    if tr.sum() < 5000:
        raise ValueError('not enough training history before cutoff')

    reg = build_volume_pipeline('reg').fit(X[tr], y[tr])
    nvol_hat = reg.predict(X[te])
    resid = y[tr] - reg.predict(X[tr])
    smearing = float(np.exp(np.var(resid) / 2))

    if label_kind == 'cross_sectional':
        lab = cross_sectional_label(y, X, q=q)
        lab.index = pd.MultiIndex.from_arrays([y.index, X['stock'].values],
                                              names=['date', 'stock'])
    else:
        lab = time_series_label(y, X, horizon, q=q)
    key = pd.MultiIndex.from_arrays([X.index, X['stock'].values], names=['date', 'stock'])
    labels = lab.reindex(key).to_numpy(dtype=float)
    ok = tr & np.isfinite(labels)
    clf = build_volume_pipeline('clf').fit(X[ok], labels[ok])
    p_high = clf.predict_proba(X[te])[:, 1]

    return pd.DataFrame({'stock': X.loc[te, 'stock'].values,
                         'nvol_hat': nvol_hat,
                         'vol_hat': np.exp(nvol_hat) * smearing,
                         'p_high': p_high,
                         'score_persist': X.loc[te, 'rw_pred'].values},
                        index=X.index[te])
