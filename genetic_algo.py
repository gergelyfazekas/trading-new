from stock_class import Stock, StockList
from portfolio_class import Portfolio, PortfolioList
import pandas as pd 
import numpy as np
import cProfile, pstats, io
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import json
import datetime

# constants
from config import ticker_list, bronze, silver, gold, forecast_folder, model_folder, model_meta_file, portfolio_folder

# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────
def create_portfolios(stock_list, features, cash=1000, n=8, scale=1):
    portfolio_list = PortfolioList([])
    for _ in range(n):
        p = Portfolio(cash=cash, stock_list=stock_list, features=features)
        p.create_weights(scale=scale)
        portfolio_list.add_portfolio(p)
    return portfolio_list


def run_single_portfolio(portfolio, data, stock_list):
    """
    One process handles ONE portfolio through the ENTIRE time-series.
    Returns the updated Portfolio (fitness already computed).
    """
    for slice_ in data.itertuples(index=True, name=None):
        dt, slice_ = slice_[0].date(), slice_[1:]
        decision    = portfolio.decide(dt=dt, data=slice_)
        transaction = portfolio.decision_to_action(decision)

        for ticker, cash in transaction.items():
            price    = stock_list[ticker].get_price(as_of=dt)
            qty      = cash / price
            if cash > 0:
                portfolio.buy (stock=stock_list[ticker], amount=qty, as_of=dt)
            else:
                portfolio.sell(stock=stock_list[ticker], amount=qty, as_of=dt)

    portfolio.calculate_fitness(as_of=data.index[-1].date())
    return portfolio


def simulate(pool, portfolio_list, data, stock_list):
    """
    Runs one generation completely in parallel (process-level).
    Returns the mutated PortfolioList and a dict of fitness scores.
    """
    worker = partial(run_single_portfolio, data=data, stock_list=stock_list)

    portfolio_list.portfolios = list(pool.map(worker, portfolio_list.portfolios))

    fitness = [p.fitness for p in portfolio_list.portfolios]
    print(f"generation {portfolio_list.generation} → {fitness}")

    fitness_per_generation = {f'gen_{portfolio_list.generation}': fitness}

    # evolve population
    portfolio_list = portfolio_list.mutate(
            n=max(2, len(portfolio_list.portfolios) // 2)
        )
    return portfolio_list, fitness_per_generation


# ────────────────────────────────────────────────────────────────────
# main entry point
# ────────────────────────────────────────────────────────────────────
def main():
    s = StockList(ticker_list)
    s.load_data(path=gold)

    df = (
        pd.read_csv(forecast_file, parse_dates=["Date"]).rename(columns={"Date": "dt"}).set_index("dt")
    )
    data = df.loc[:, s.tickers]

    portfolio_list = create_portfolios(
        stock_list=s, features=s.tickers, n=64
    )

    # macOS / Windows need 'spawn'; set once at program start
    if mp.get_start_method(allow_none=True) != "fork":
        mp.set_start_method("spawn", force=True)
    
    max_workers=8
    checkpoint_interval = 2
    
    with ProcessPoolExecutor(max_workers=min(max_workers, len(portfolio_list.portfolios))) as pool:
        while len(portfolio_list.portfolios) >= 1:
            portfolio_list, fitness_hist = simulate(
                            pool=pool,
                            portfolio_list=portfolio_list,
                            data=data,
                            stock_list=s
                        )
            
            checkpoint(portfolio_list=portfolio_list, interval=None, portfolio_folder=portfolio_folder)   

    return portfolio_list, fitness_hist

def checkpoint(portfolio_list, interval, portfolio_folder):
    if interval:
        condition = ((portfolio_list.generation > 0) and (portfolio_list.generation % interval)) or (len(portfolio_list.portfolios)==1)
    else:
        condition = len(portfolio_list.portfolios)==1
    if condition:
        print('Saving checkpoint.')
        for p in portfolio_list.portfolios:
            p.save_weights(
                file_path = portfolio_folder + f'/gen{portfolio_list.generation}_id{p.id}_{datetime.datetime.now()}.json'
                )

def profile(func):
    pr = cProfile.Profile()
    pr.enable()
    res = func()
    pr.disable()
    txt = io.StringIO()
    pstats.Stats(pr, stream=txt).sort_stats("cumtime").print_stats(30)
    print(txt.getvalue())
    return res


if __name__ == '__main__':
    #main()
    portfolio_list, fitness_per_generation = profile(main)