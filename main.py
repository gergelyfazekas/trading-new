from stock_class import Stock, StockList
from portfolio_class import Portfolio, PortfolioList
from broker import IBroker
import pandas as pd 
import numpy as np
import time
import cProfile, pstats, io
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import forecast


# constants
from config import ticker_list, bronze, silver, gold, forecast_folder, model_folder, model_meta_file, portfolio_folder


model_name = f'rf_{datetime.date.today()}'
process_bronze = False
process_silver = False
process_gold = False
create_new_model = False
create_weights = True
pull_today = False
y_name = 'log_return'
x_names = ['tech_strong', 'tech_medium']



def add_data_to_bronze(stock_list, bronze):
    stock_list.clear_data()
    stock_list.load_data(bronze)
    broker = IBroker()
    broker.connect()
    for ticker in stock_list.tickers:
        # Get market data
        close, volume = broker.get_market_data(ticker)
        # Create a new dataframe with that data
        today = datetime.date.today()
        new_data = pd.DataFrame({'close': close, 'volume': volume, 'dt': today}, index=[0]).set_index('dt')
        # Append it the old data
        df = pd.concat([stock_list[ticker].data, new_data])
        # Deduplicate so that we only have one row per date
        stock_list[ticker].data = df.groupby(df.index).last()
    # Write
    stock_list.save_data(path=bronze)


stock_list = StockList(ticker_list)
stock_list.clear_data()

# ----- process bronze -----
if process_bronze:
    stock_list.pull_data()
    stock_list.save_data(path=bronze)

# ----- process silver -----
if process_silver:
    stock_list.clear_data()
    stock_list.load_data(bronze)
    for ticker, stock in stock_list.stocks.items():
        stock.calc_return()
        stock.calc_variance()
        stock.sma_calc(9)
        stock.sma_calc(14)
        stock.sma_calc(20)
        stock.sma_calc(50)
        stock.sma_calc(100)
        stock.rsi_calc(9)
        stock.rsi_calc(14)
        stock.rsi_calc(20)
        stock.rsi_calc(50)
        stock.rsi_calc(100)
    stock_list.save_data(path=silver)



# ----- process gold -----
if process_gold:
    stock_list.clear_data()
    stock_list.load_data(silver)
    for ticker, stock in stock_list.stocks.items():
        stock.sma_cross(9, 14)
        stock.sma_cross(14, 20)
        stock.sma_cross(20, 50)
        stock.sma_cross(50, 100)
        stock.tech_level_input_calc(100)
    stock_list.save_data(path=gold)


if create_new_model:
    # ----- load gold data for forecasting -----
    stock_list.clear_data()
    stock_list.load_data(path=gold)

    # ----- create a new model and save, or load the already trained model -----
    X, Y = forecast.create_x_y(stock_list=stock_list, y_col=y_name, x_cols=x_names)
    pipe = forecast.model_init(model_name=model_name, model_folder=model_folder)
    pipe = forecast.fit(pipe=pipe, X=X, Y=Y)
    forecast.save_model(pipe=pipe, model_name=model_name, model_folder=model_folder)
else:
    pipe = forecast.load_model(model_name=model_name, model_folder=model_folder)

# ----- add today's data to the bronze layer (first it loads it) -----
if pull_today:
    add_data_to_bronze(stock_list=stock_list, bronze=bronze)

    # ----- recalculate the silver and gold layers again with today's data -----
    #  --- silver ---
    stock_list.clear_data()
    stock_list.load_data(bronze)
    for ticker, stock in stock_list.stocks.items():
        stock.calc_return()
        stock.calc_variance()
        stock.sma_calc(9)
        stock.sma_calc(14)
        stock.sma_calc(20)
        stock.sma_calc(50)
        stock.sma_calc(100)
        stock.rsi_calc(9)
        stock.rsi_calc(14)
        stock.rsi_calc(20)
        stock.rsi_calc(50)
        stock.rsi_calc(100)
    stock_list.save_data(path=silver)

    #  --- gold ---
    stock_list.clear_data()
    stock_list.load_data(silver)
    for ticker, stock in stock_list.stocks.items():
        stock.sma_cross(9, 14)
        stock.sma_cross(14, 20)
        stock.sma_cross(20, 50)
        stock.sma_cross(50, 100)
        stock.tech_level_input_calc(100)
    stock_list.save_data(path=gold)




stock_list.clear_data()
stock_list.load_data(gold)
x, _ = forecast.create_x_y(stock_list=stock_list, y_col=y_name, x_cols=x_names, last_day_only=True)

y_hat = forecast.predict(pipe=pipe, x=x, stock_list=stock_list)

portfolio = Portfolio(stock_list=stock_list, features=stock_list.tickers)
if create_weights:
    portfolio.create_weights()
else:
    portfolio.load_weights(file_path='/Users/gergelyfazekas/Documents/python_projects/trading_new_structure/data/portfolios/gen3_id0_2025-07-14 14:38:25.321841.json')

decision = portfolio.decide(dt=y_hat.index.tolist()[0][0], data=y_hat.to_numpy())
action = portfolio.decision_to_action(decision=decision)

broker = IBroker()
broker.connect()

for ticker, value in action.items():
    if value > 0:
        amount = portfolio.buy(stock=stock_list[ticker], value=value)
        amount = round(amount)
        if amount > 0:
            print(ticker, amount)
            broker.buy_stock(ticker=ticker, quantity=amount)
    else:
        amount = portfolio.sell(stock=stock_list[ticker], value=value)
        amount = round(abs(amount))
        if amount > 0:
            print(ticker, amount)
            broker.sell_stock(ticker=ticker, quantity=amount)

print(portfolio.log.head(20))

