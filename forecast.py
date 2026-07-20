from stock_class import StockList
import pandas as pd
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

def create_long_x_y(stock_list, y_col, x_cols=None, cols_to_drop=None, add_time_features=True):
    """
    Returns X (features) and Y (targets) in long format:
    - one row per (date, stock)
    - shared feature set
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

        # Shift target up by 1 day (predict t+1 log_return)
        df['target'] = df[y_col].shift(-1)
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

def build_pipeline():
    regressor = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=10,
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