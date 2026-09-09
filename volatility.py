"""Forward realized-volatility model (stage C of the forecasting pipeline).

Separate from the return model on purpose. Volatility is far more predictable
than direction (R^2 in the tens of percent vs low single digits), so training
both as one multi-output model would let the easy target dominate every split
and degrade the return forecast. Two models, one job each.

Produces sigma_hat, which is used twice downstream:
  - as the denominator that turns raw returns into the vol-scaled target the
    pooled return model trains on (without it, squared-error loss over a pooled
    panel is dominated by whichever stocks happen to be noisiest)
  - as a per-stock risk input to portfolio construction

Target is log realized vol over the *next* `horizon` trading days. Log because
realized vol is strongly right-skewed while log RV is close to Gaussian, which
is what squared-error loss assumes. Features are the HAR cascade (Corsi 2009):
trailing RV over 1/5/22/66 days, which captures the well-documented long-memory
persistence of volatility with very few parameters, plus vol-of-vol, term
structure ratios, and a cross-sectional market vol factor.

Causality: every feature at date t uses returns at or before t; the target uses
t+1..t+horizon only. Because targets span `horizon` days, adjacent train/test
folds overlap -- evaluate_walkforward purges `horizon` days before each test
fold, without which the reported scores would be optimistic.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# HAR cascade: daily, weekly, monthly, quarterly
RV_WINDOWS = [1, 5, 22, 66]
EPS = 1e-8


def realized_vol(log_ret: pd.Series, window: int) -> pd.Series:
    """Trailing realized vol over `window` days, ending at (and including) each date.

    sqrt of mean squared return rather than std: the standard RV definition, and
    it avoids estimating a mean that is indistinguishable from zero at daily
    frequency anyway.
    """
    return np.sqrt((log_ret ** 2).rolling(window).mean())


def forward_vol(log_ret: pd.Series, horizon: int) -> pd.Series:
    """Realized vol over the next `horizon` days -- the prediction target.

    Strictly forward-looking: the value at date t covers t+1..t+horizon, so it
    shares no observation with any trailing feature computed at t.
    """
    return np.sqrt((log_ret ** 2).rolling(horizon).mean()).shift(-horizon)


def build_vol_panel(stock_list, horizon=10, return_col='log_return'):
    """Long panel (one row per date-stock) of vol features and the log forward-vol target."""
    frames = []
    for tkr in stock_list.tickers:
        df = stock_list[tkr].data
        r = pd.Series(df[return_col].values, index=pd.to_datetime(df.index), name='r').dropna()

        f = pd.DataFrame(index=r.index)
        for w in RV_WINDOWS:
            f[f'rv_{w}'] = np.log(realized_vol(r, w) + EPS)
        # term structure: short vol relative to long vol -- is the stock calming
        # down or heating up, independent of its overall level
        f['rv_ratio_5_22'] = f['rv_5'] - f['rv_22']
        f['rv_ratio_22_66'] = f['rv_22'] - f['rv_66']
        # vol of vol: how unstable the vol estimate itself has been
        f['vol_of_vol'] = f['rv_5'].rolling(22).std()
        # sign asymmetry (leverage effect): downside moves raise future vol more
        f['neg_ret_share'] = (r < 0).rolling(22).mean()
        f['ret_22'] = r.rolling(22).sum()

        if 'volume' in df.columns:
            v = pd.Series(df['volume'].values, index=pd.to_datetime(df.index)).reindex(r.index)
            f['volume_ratio'] = np.log((v.rolling(5).mean() / v.rolling(66).mean()).clip(lower=EPS))

        f['target'] = np.log(forward_vol(r, horizon) + EPS)
        f['stock'] = tkr
        frames.append(f)

    panel = pd.concat(frames).sort_index()

    # cross-sectional market vol: the common component every stock shares. A
    # pooled model with only own-stock features cannot see that the whole market
    # is in a high-vol regime, which is most of what moves vol.
    panel['market_rv_5'] = panel.groupby(level=0)['rv_5'].transform('mean')
    panel['market_rv_22'] = panel.groupby(level=0)['rv_22'].transform('mean')
    panel['rv_22_vs_market'] = panel['rv_22'] - panel['market_rv_22']

    panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
    y = panel['target']
    X = panel.drop(columns=['target'])
    return X, y


def build_vol_pipeline(max_features=0.5, n_estimators=300, min_samples_leaf=20):
    regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=0,
        n_jobs=-1,
    )
    preprocess = ColumnTransformer(
        [('stock_ohe', OneHotEncoder(handle_unknown='ignore'), ['stock'])],
        remainder='passthrough',
    )
    return Pipeline([('preprocess', preprocess), ('rf', regressor)])


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def evaluate_walkforward(X, y, horizon, n_folds=4, har_cols=('rv_1', 'rv_5', 'rv_22')):
    """Walk-forward evaluation against the two baselines that matter.

    Reported alongside the RF, because volatility is persistent enough that
    beating a naive carry-forward is the whole question:
      - random walk: predict the trailing RV over the same length as the horizon
        (log rv_22 for horizon<=22), i.e. "vol tomorrow = vol lately"
      - HAR: OLS on log rv_1/rv_5/rv_22, the standard benchmark in the realized
        volatility literature and a genuinely hard one to beat

    Each test fold is preceded by a purge of `horizon` days: the last training
    target before a fold spans days that fall inside it, so without the gap the
    model would be fit on partly-overlapping information.
    """
    dates = np.array(sorted(X.index.unique()))
    folds = np.array_split(dates, n_folds + 1)[1:]
    rw_col = 'rv_22' if horizon <= 22 else 'rv_66'

    rows = []
    for f in folds:
        purge = f[0] - pd.Timedelta(days=int(horizon * 1.5))
        tr = X.index < purge
        te = (X.index >= f[0]) & (X.index <= f[-1])
        if tr.sum() < 5000:
            continue

        pipe = build_vol_pipeline().fit(X[tr], y[tr])
        pred_rf = pipe.predict(X[te])

        har = LinearRegression().fit(X.loc[tr, list(har_cols)], y[tr])
        pred_har = har.predict(X.loc[te, list(har_cols)])

        # hybrid: HAR carries the level, RF fits only what HAR leaves behind.
        # A tree cannot predict outside the range of targets it saw in training,
        # so a pure RF systematically under-forecasts vol spikes larger than
        # anything in its training window -- precisely the regime where the
        # forecast matters most. Keeping a linear model underneath restores
        # extrapolation while still capturing nonlinearity in the residual.
        resid = y[tr] - har.predict(X.loc[tr, list(har_cols)])
        pipe_resid = build_vol_pipeline().fit(X[tr], resid)
        pred_hybrid = pred_har + pipe_resid.predict(X[te])

        pred_rw = X.loc[te, rw_col].to_numpy()

        rows.append({
            'test_start': f[0].date(), 'test_end': f[-1].date(), 'n_test': int(te.sum()),
            'r2_rw': _r2(y[te], pred_rw),
            'r2_har': _r2(y[te], pred_har),
            'r2_rf': _r2(y[te], pred_rf),
            'r2_hybrid': _r2(y[te], pred_hybrid),
        })

    out = pd.DataFrame(rows)
    out.loc['mean'] = out.mean(numeric_only=True)
    return out


def fit_predict_sigma(X, y, cutoff, horizon, har_cols=('rv_1', 'rv_5', 'rv_22')):
    """Fit on data before `cutoff`, return sigma_hat (in return units, not logs)
    for every row at/after it -- the denominator for the return model's target.

    Uses the HAR+RF hybrid, which scored best in evaluate_walkforward and, more
    importantly, degrades most gracefully in high-vol regimes where a pure RF
    cannot extrapolate. Training stops `horizon * 1.5` days before the cutoff so
    no training target overlaps the prediction period.
    """
    har_cols = list(har_cols)
    tr = X.index < pd.Timestamp(cutoff) - pd.Timedelta(days=int(horizon * 1.5))
    te = X.index >= pd.Timestamp(cutoff)

    har = LinearRegression().fit(X.loc[tr, har_cols], y[tr])
    resid = y[tr] - har.predict(X.loc[tr, har_cols])
    pipe_resid = build_vol_pipeline().fit(X[tr], resid)

    # Retransformation (Jensen) correction. The model is fit in logs, so
    # exp(prediction) recovers the conditional *median* vol, not the mean --
    # it under-forecasts by roughly exp(s^2/2). Harmless when sigma_hat is only
    # used to rescale a target (a constant factor cancels), but it biases risk
    # estimates low, so correct it here rather than leaving a known-low number
    # for portfolio construction to consume.
    in_sample_resid = resid - pipe_resid.predict(X[tr])
    smearing = np.exp(np.var(in_sample_resid) / 2)

    log_sigma = har.predict(X.loc[te, har_cols]) + pipe_resid.predict(X[te])
    return pd.DataFrame({'stock': X.loc[te, 'stock'].values,
                         'sigma_hat': np.exp(log_sigma) * smearing}, index=X.index[te])
