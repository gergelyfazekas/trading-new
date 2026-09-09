from stock_class import StockList
import pandas as pd
import numpy as np
from pathlib import Path
from functools import reduce
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline   import Pipeline
from sklearn.metrics    import mean_absolute_error
import joblib
import pathlib, datetime, sklearn

# config
from config import ticker_list, bronze, silver, gold, forecast_folder, model_folder, model_meta_file, portfolio_folder

test_size  = 250
model_name = f'rf_{datetime.date.today()}'


def create_x_y(stock_list, y_col, x_cols, last_day_only=False):
    # create feature matrix: X, Y
    features, targets = [], []

    x_cols.append(y_col)
    x_and_y = x_cols.copy()

    for tkr in stock_list.tickers:
        df = stock_list[tkr].data.loc[:, x_and_y]
        
        # at this point df only contains x and y
        df = df.add_prefix(tkr + '_')

        if last_day_only:
            y = df.loc[:, [f'{tkr}_{y_col}']].shift(-1) # we are not using this in forecasting today's y_hat
            x = df.drop(columns=[f'{tkr}_{y_col}'], errors='ignore')
            x = pd.DataFrame(dict(x.iloc[-1, :]), index=[x.iloc[-1, :].name]) # Only last row but keeping it a dataframe
        else:
            y = df.loc[:, [f'{tkr}_{y_col}']].shift(-1) # We shift the the real log_return up (to yesterday's date) 
            x = df.drop(columns=[f'{tkr}_{y_col}'], errors='ignore')
        
        features.append(x)
        targets.append(y)

    # Wide feature matrix and target matrix, aligned on the intersection of dates
    X = pd.concat(features, axis=1, join="inner")
    Y = pd.concat(targets , axis=1, join="inner").loc[X.index]


    print(X)
    print('................................')
    print(Y)

    # If we'll use both X and Y (aka not last_day_only) --> dropna from Y and only keep corresponding X
    # Else we want to keep the last row where Y is nan because we don't know the log_return from today to tomorrow --> but X is not nan here
    if not last_day_only:
        # Drop NaNs from X created by rolling indicators
        #X = X.dropna()
        X = X.iloc[200:, :]
        # Keep only the Ys that have X after the dropna
        Y = Y.loc[X.index]
        Y = Y.dropna()
        X = X.loc[Y.index]
    return X, Y

def create_long_x_y(stock_list, y_col, x_cols=None, cols_to_drop=None, add_time_features=True,
                    horizon=1):
    """
    Returns X (features) and Y (targets) in long format:
    - one row per (date, stock)
    - shared feature set

    horizon: how many days forward the target spans. 1 reproduces the original
    next-day behaviour; h > 1 gives the cumulative log return over t+1..t+h.
    Note that h > 1 makes consecutive rows' targets overlap, so any cross
    validation over the result must purge at least h days between train and test.
    """
    all_rows = []

    for tkr in stock_list.tickers:
        df = stock_list[tkr].data.copy()

        # Drop columns if specified
        if x_cols:
            df = df[x_cols + [y_col]]
        elif cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors='ignore')

        # Add stock identifier
        df['stock'] = tkr

        # Forward target over t+1..t+horizon (log returns add, so a plain sum)
        df['target'] = df[y_col].rolling(horizon).sum().shift(-horizon)
        df = df.drop(columns=[y_col])

        all_rows.append(df)



    # Combine into one long dataframe
    full_df = pd.concat(all_rows)
    full_df = full_df.dropna()

    full_df.index = pd.to_datetime(full_df.index)
    full_df['dayofweek'] = full_df.index.dayofweek

    # Features and target
    Y = full_df['target']
    X = full_df.drop(columns=['target'])

    return X, Y

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# Columns in the gold tier that are identifiers or leftovers rather than features:
# the date survives as several duplicated columns (dt, dt.1 ... dt.3) from the
# pull/merge steps, and 'ticker' duplicates the 'stock' column create_long_x_y adds.
# Leaving any of them in makes the panel non-numeric and RandomForest.fit raises.
NON_FEATURE_COLS = ['dt', 'dt.1', 'dt.2', 'dt.3', 'ticker']

# Raw price and volume *levels*. A tree splits on absolute values, so in a panel
# pooled across stocks these act as stock identity and regime labels -- "close >
# 180" identifies a handful of tickers in a handful of years, which fits the
# training window and generalizes to nothing. Dropping them took the baseline
# from consistently anti-predictive (mean IC -0.018, negative in all four
# walk-forward folds) to roughly zero, and improved the with-tl_ arm as well.
# The information is not lost: the tl_* distances, sma_cross_*, rsi_* and the
# vol model's own features all express the same content in stationary form.
NON_STATIONARY_COLS = ['close', 'high', 'low', 'open', 'volume',
                       'sma_9', 'sma_14', 'sma_20', 'sma_50', 'sma_100']


def build_panel(stock_list, y_col='log_return', selection_cutoff='2021-06-01',
                tech_level_log=None, cols_to_drop=None, horizon=10, vol_scale=True):
    """Load technical-level features onto stock_list and return the long panel.

    Wraps create_long_x_y with the tl_* columns attached (tech_level_input), and
    enforces the one ordering rule that keeps the combo choice honest: combos and
    their confidence weights are selected using only data at or before
    selection_cutoff, so the panel is truncated to start strictly after it.
    Without that truncation the model would train on dates whose feature
    *configuration* was chosen with knowledge of those same dates.

    horizon / vol_scale: the target is the cumulative log return over the next
    `horizon` days, divided by the volatility model's sigma_hat * sqrt(horizon)
    when vol_scale is on. Scaling matters specifically because this panel is
    pooled across stocks: squared-error loss on raw returns is dominated by
    whichever names are noisiest, so the model spends its capacity on them and
    treats a large move in a calm stock as near-zero signal. Dividing by
    forecast vol makes a "surprise" comparable across stocks and across regimes.
    sqrt(horizon) is the usual scaling of cumulative return dispersion, so the
    target comes out roughly unit-variance.

    sigma_hat is produced by volatility.fit_predict_sigma trained strictly before
    selection_cutoff, so it carries no information from the panel's own period.

    tech_level_log: the grid-search log (tech_level_scoring.load_log). If None,
        it's loaded from config.tech_level_folder.
    """
    from tech_level_input import select_combos, add_level_features
    from tech_level_scoring import load_log
    from config import tech_level_folder

    log = tech_level_log if tech_level_log is not None else load_log(tech_level_folder)
    combos, confidence = select_combos(log, selection_cutoff)
    add_level_features(stock_list, combos, confidence=confidence)

    drop = (NON_FEATURE_COLS + NON_STATIONARY_COLS) if cols_to_drop is None else cols_to_drop
    X, Y = create_long_x_y(stock_list, y_col=y_col, cols_to_drop=drop, horizon=horizon)
    X.index = pd.to_datetime(X.index)
    Y.index = X.index

    start = pd.Timestamp(selection_cutoff)
    keep = X.index > start
    X, Y = X.loc[keep], Y.loc[keep]

    if vol_scale:
        from volatility import build_vol_panel, fit_predict_sigma
        vX, vy = build_vol_panel(stock_list, horizon=horizon, return_col=y_col)
        sigma = fit_predict_sigma(vX, vy, cutoff=selection_cutoff, horizon=horizon)

        key = pd.MultiIndex.from_arrays([X.index, X['stock']])
        sig_key = pd.MultiIndex.from_arrays([sigma.index, sigma['stock']])
        aligned = pd.Series(sigma['sigma_hat'].values, index=sig_key).reindex(key)

        ok = aligned.notna().values & (aligned.values > 0)
        X, Y = X[ok], Y[ok]
        Y = Y / (aligned.values[ok] * np.sqrt(horizon))

    print(f'panel: {X.shape[0]:,} rows x {X.shape[1]} features, '
          f'{X.index.min().date()} -> {X.index.max().date()}, '
          f'horizon={horizon}d, vol_scaled={vol_scale}, target sd={Y.std():.3f}')
    return X, Y

def build_pipeline(max_features=0.5):
    regressor = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=10,
        # sklearn's regressor default is 1.0 -- every split considers every
        # feature, so a handful of dominant features can crowd weaker ones out of
        # the forest entirely. Capping it forces splits to be chosen without them
        # part of the time, which is what gives minority-useful features (the
        # tl_* level columns, useful on a few stocks) a chance to be used at all.
        max_features=max_features,
        random_state=0,
        n_jobs=-1
    )

    # Preprocessing: OneHotEncode the 'stock' column
    preprocessor = ColumnTransformer([
        ('stock_ohe', OneHotEncoder(handle_unknown='ignore'), ['stock'])
    ], remainder='passthrough')  # keep other features

    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('rf', regressor)
    ])
    return pipe


# Predict the next day for all stocks
def predict_next(pipe, stock_list, x_cols):
    rows = []

    for tkr in stock_list.tickers:
        df = stock_list[tkr].data.copy()
        last_row = df.iloc[[-1]][x_cols].copy()

        row = last_row.copy()
        row['stock'] = tkr
        row['date'] = df.index[-1] + pd.Timedelta(days=1)
        row['dayofweek'] = row['date'].dt.dayofweek
        row['month'] = row['date'].dt.month
        rows.append(row)

    X_future = pd.concat(rows)
    preds = pipe.predict(X_future)

    return pd.DataFrame({
        'stock': X_future['stock'],
        'date': X_future['date'],
        'prediction': preds
    })


def model_init(model_name=None, model_folder=None):
    regressor = RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=10,
                random_state=0,
                n_jobs=-1,
        )
        
    # random forest regressor init
    pipe = Pipeline([     
        ("rf", regressor)
    ])

    return pipe

def fit(pipe, X, Y):
    # Fit the model to whole dataset
    pipe.fit(X, Y)
    return pipe

def save_model(pipe, model_folder, model_name):
    # Save the model into the models folder
    joblib.dump(pipe, model_folder + f"/{model_name}.joblib", compress=("gzip", 3))      # gzip-compress the file (level 3)
    print(f"Saved to {model_folder}/{model_name}.joblib")

def load_model(
        model_name,
        model_folder=model_folder
        ):
    pipe = model_init()
    pipe = joblib.load(model_folder + f"/{model_name}.joblib")
    return pipe

def predict(pipe, x, stock_list):
    preds = pipe.predict(x)
    df = pd.DataFrame(dict(zip(stock_list.tickers, preds[0].tolist())), index=[x.index+datetime.timedelta(days=1)])
    return df

def cross_validate(pipe, X, Y, test_size, stock_list):
    # rolling window cross validation (folds are determined by test_size)
    scores = []

    # train-test split
    n_splits   = len(X) // test_size - 1
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)


    for fold, (train, test) in enumerate(tscv.split(X)):
        pipe.fit(X.iloc[train], Y.iloc[train])
        pred  = pipe.predict(X.iloc[test])
        score = mean_absolute_error(Y.iloc[test], pred, multioutput="raw_values")
        scores.append(score)

    print("Average MAE:\n") 
    print(pd.DataFrame(scores, columns=stock_list.tickers).mean())
    return pipe

def model_meta():
    # save meta data about the model
    meta = pd.DataFrame(
        {
            'name': regressor.__str__().split('(')[0],
            'start': X.index.min(),
            'end': X.index.max(),
            'model_folder': model_folder,
            'model_name': model_name,
            'features': ';'.join(X.columns.to_list())
            
        }, 
        index = [0]
    )

    model_params = pipe.get_params()
    model_params.pop('steps')
    model_params.pop('rf')
    model_params = pd.DataFrame(model_params, index=[0])


    meta = pd.concat([meta, model_params], axis=1)

    try:
        all_meta = pd.read_csv(model_meta_file, index_col=0)
        pd.concat([all_meta, meta]).reset_index().to_csv(model_meta_file)
    except:
        meta.to_csv(model_meta_file)


if __name__ == '__main__':
    stock_list = StockList(ticker_list)
    stock_list.load_data(path=gold)

    X, Y = create_x_y(stock_list=stock_list)
    pipe = model_init(model_name=model_name, model_folder=model_folder)
    pipe = fit(pipe=pipe, X=X, Y=Y)
    save_model(pipe=pipe, model_name=model_name, model_folder=model_folder)


    pipe = load_model(model_name=model_name, model_folder=model_folder)
    predict(pipe, x=X.iloc[-1, :], stock_list=stock_list)