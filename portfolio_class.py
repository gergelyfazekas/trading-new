import datetime
# turn of FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import stock_class
from stock_class import Stock
from typing import List, Dict
import random
import math
import json

# turn off chained assignment warning
pd.options.mode.chained_assignment = None


class PortfolioList:
    def __init__(self, portfolios, generation=0):
        self.portfolios = portfolios
        self.max_id = max([p.id for p in self.portfolios]) if self.portfolios else 0
        self.generation = generation

    def add_portfolio(self, p):
        if isinstance(p, Portfolio):
            p.id = self.max_id
            self.max_id += 1
            self.portfolios.append(p)
            
        else:
            raise ValueError(f'Not a portfolio: {type(p)} --- {p}')
        

    def mutate(self, n=None):
        if n is None:
            n = len(self.portfolios)
        items = self.n_fittest(n)
        random.shuffle(items)                    # shuffle in place
        pairs = list(zip(items[::2], items[1::2]))

        new_portfolio_list = self.__class__([], generation=self.generation+1)
        
        for item in pairs:
            avg = {k: (item[0].weights[k] + item[1].weights[k]) / 2 for k in item[0].weights.keys()}
            new_portfolio_list.add_portfolio(
                Portfolio(
                    cash=item[0].cash_init, 
                    stock_list=item[0].stock_list,
                    features=item[0].features,
                    weights=avg
                    )
                )
        return new_portfolio_list

    def n_fittest(self, n):
        fitnesses = [(p, p.fitness) for p in self.portfolios]
        fitnesses = sorted(fitnesses, key=lambda x: x[1])
        return [item[0] for item in fitnesses[-n:]]


class Portfolio:
    def __init__(self, cash=1000, stock_list=None, features=None, weights=None):
        self.fitness = 0
        self.stock_list = stock_list
        self.features = features
        self.weights = weights
        self.cash_init = cash
        self.cash_spent = 0
        self.breach = 0 # amount of times the constraints would not be respected be the decision and we do nothing instead
        self.number_of_stocks = int
        self.currencies = list
        self.log = pd.DataFrame({
                                'dt': [np.nan] * stock_class.PLACEHOLDER,
                                'stock_name': [np.nan] * stock_class.PLACEHOLDER,
                                'direction': [np.nan] * stock_class.PLACEHOLDER,
                                'amount': [np.nan] * stock_class.PLACEHOLDER,
                                'price': [np.nan] * stock_class.PLACEHOLDER,
                                'value': [np.nan] * stock_class.PLACEHOLDER,
                                'breach': [np.nan] * stock_class.PLACEHOLDER
                                })
        self.balance = pd.DataFrame({
            'name': ['cash'], 
            'amount': [cash], 
            'sector': [np.nan],
            'price': [1]
            })

        self.balance['value'] = self.balance['price'] * self.balance['amount']

        if self.weights:
            for name, value in self.weights.items():
                setattr(self, name, value)

        self.total_portfolio_value_hist = []
        self.portfolio_return_hist = []

    def load_weights(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            weights = json.load(f)
            self.weights = {k: np.array(v) for k, v in weights.items()}

    def save_weights(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            weights = {k: v.tolist() for k, v in self.weights.items()}
            json.dump(weights, f, indent=2, ensure_ascii=False)

    def calculate_fitness(self, as_of):
        self.fitness = float(self.value(as_of=as_of, sum=True))

    def create_weights(self, scale=1):
        # KxS
        self.weights = np.random.normal(loc=0, scale=random.randint(1, scale), size=(len(self.features), len(self.stock_list)+1))
    
    def decide(self, dt, data):
        """
        Get for a single day and decide the different proportions
        If we only used market data as input (e.g. each stock's forecast) and not portfolio attributes then this could work vectorized
        and not only per-day.
        The decision represents the desired split of the portfolio tomorrow.
        """
        #data = np.array(data).reshape((self.weights.shape[1], self.weights.shape[0]))
        split = self.softmax(data.dot(self.weights)).tolist()[0]
        decision = {
            'dt': dt,
            'split': dict(zip(self.stock_list.tickers, split))
        } # example --> {'cash': 0.55, 'AAPL': 0.2, 'MSFT': 0.25}
        return decision
    
    def decision_to_action(self, decision):
        dt = decision['dt']
        decision = decision['split']
        desired_quantity = {}
        action = {}
        portfolio_value = self.value(as_of=dt, strict=False, sum=True)
        for ticker, split in decision.items():
            if ticker != 'cash':
                price = self.stock_list[ticker].get_price(as_of=dt, strict=False)
                current_amount = self.balance.loc[self.balance['name']==ticker, 'amount'] if not self.balance.loc[self.balance['name']==ticker, 'amount'].empty else 0
                desired_quantity[ticker] = (split * portfolio_value) / price # example --> {'AAPL': (0.2*1000)/250, ...}
                action[ticker] = desired_quantity[ticker] - current_amount
                action[ticker] = math.ceil(action[ticker]) if action[ticker] <= 0 else math.floor(action[ticker])  # always round closer to zero -1.8 --> -1; 1.8 --> 1

        # We have a quantity to sell (if negative) or buy (if positive)
        return {k: v for k, v in action.items() if v != 0}

    @staticmethod
    def sigmoid(arr):
        try:
            return self.stable_sigmoid(arr)
        except:
            return 1/(1+np.exp(-arr))

    @staticmethod
    def stable_sigmoid(arr):
        """Vectorised, overflow-safe logistic function."""
        x = np.asarray(x, dtype=np.float64)        # ensure float64 for headroom
        positive = x >= 0
        negative = ~positive
        out = np.empty_like(x, dtype=np.float64)
        # For x >= 0: use the standard form
        out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        # For x < 0: rewrite sigmoid(x) = exp(x) / (1 + exp(x))
        # Here exp(x) is <= 1, so no overflow.
        exp_x = np.exp(x[negative])
        out[negative] = exp_x / (1.0 + exp_x)
        return out

    @staticmethod
    def softmax(arr):
        shifted = arr - arr.max(axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        weights = exp_vals / exp_vals.sum(axis=1, keepdims=True)
        return weights


    def value(self, as_of, strict=False, sum=False):
        df = self.balance.copy()
        df['price'] = [self.stock_list[ticker].get_price(as_of) if ticker != 'cash' else 1 for ticker in df['name']]
        df['value'] = df['price'] * df['amount']

        if sum:
            return df[f'value'].sum()
        else:
            return df

    def entropy_stock(self, as_of):
        df = self.value(as_of=as_of)
        cutoff = 0.000001
        weights = np.array(df['value'] / df['value'].sum())
        weights[weights < cutoff] = cutoff
        return -1 * np.sum(weights * np.log(weights))

    def entropy_sector(self, as_of):
        df = self.value(as_of=as_of)
        cutoff = 0.000001
        sector_values = list(df.groupby(by="sector")['value'].sum())
        total_value = df['value'].sum()
        weights = np.array(sector_values / total_value)
        weights[weights < cutoff] = cutoff
        return -1 * np.sum(weights * np.log(weights))

    @property
    def proportion_invested(self):
        return self.cash_spent / self.cash_init

    @property
    def sectors(self):
        return list(self.balance['sector'].unique())

    def got_enough_cash(self, value):
        if type(value) in (int, np.integer, float, np.float16, np.float32, np.float64):
            if value >= 0:
                if value <= self.balance.loc[self.balance['name']=='cash', 'amount'].sum():
                    return True
                else:
                    return False
            else:
                # print(f'Negative value: {value}')
                return False
        else:
            raise TypeError('value not in (int, float)')

    def got_enough_amount(self, stock, amount):
        if type(amount) in (int, np.integer, float, np.float16, np.float32, np.float64):
            if amount <= 0:
                if any(abs(amount) <= self.balance.loc[self.balance['name'] == stock.name, 'amount']):
                    return True
                else:
                    return False
            else:
                return False
        elif type(amount) is pd.Series:
            if any(amount <= 0):
                if any(abs(amount)) <= any(self.balance.loc[self.balance['name'] == stock.name, 'amount']):
                    return True
                else:
                    return False
            else:
                return False
        else:
            raise TypeError('value not in (int, float)')

    def get_stock_amount(self, stock):
        if self.balance.loc[self.balance['name'] == stock.name, 'amount'].empty:
            return 0
        else:
            return self.balance.loc[self.balance['name'] == stock.name, 'amount']

    def deduct_cash(self, amount):
        if type(amount) in (int, np.integer, float, np.float16, np.float32, np.float64):
            if amount >= 0:
                self.balance.loc[self.balance['name']=='cash', 'amount'] -= amount
                return 0
        else:
            amount = float(amount)
            self.deduct_cash(amount)

    def add_cash(self, amount):
        if type(amount) in (int, np.integer, float, np.float16, np.float32, np.float64):
            if amount >= 0:
                self.balance.loc[self.balance['name']=='cash', 'amount'] += amount
                return 0
        else:
            amount = float(amount)
            self.add_cash(amount)

    def update_cash_spent(self, value):
        """accepts negative value as well meaning that we get back cash by selling a stock"""
        try:
            self.cash_spent += float(value)
        except TypeError:
            print("cash_spent not updated with value, ", value)

    def update_balance_transaction(self, stock, amount, sector):
        non_zero = True
        try:
            if np.round(abs(amount), 2) == 0:
                non_zero = False
        except ValueError:
            if any(np.round(abs(amount), 2) == 0):
                non_zero = False
        if non_zero:
            tmp = pd.DataFrame({
                    'name': stock.name,
                    'amount': amount,
                    'sector': sector
                    },
                    index=[0])
            self.balance = pd.concat([self.balance, tmp])
            self.balance = self.balance.groupby('name', as_index=False).agg({'amount': 'sum', 'sector': 'max'})
        self.balance = self.balance.loc[(self.balance['amount'] > 0) | (self.balance['name'] == 'cash')].copy().reset_index(drop=True)

    def update_log(self, as_of, stock, direction, amount, price, value, breach):
        non_zero = True
        try:
            if np.round(abs(amount), 2) == 0:
                non_zero = False
        except ValueError:
            if any(np.round(abs(amount), 2) == 0):
                non_zero = False
        if non_zero:
            idx = self.log['stock_name'].last_valid_index()
            if type(idx) in (int, np.integer, float, np.float16, np.float32, np.float64):
                try:
                    self.log.iloc[idx + 1, self.log.columns.get_loc('dt')] = as_of
                    self.log.iloc[idx + 1, self.log.columns.get_loc('stock_name')] = stock.name
                    self.log.iloc[idx + 1, self.log.columns.get_loc('direction')] = direction
                    self.log.iloc[idx + 1, self.log.columns.get_loc('amount')] = amount
                    self.log.iloc[idx + 1, self.log.columns.get_loc('price')] = price
                    self.log.iloc[idx + 1, self.log.columns.get_loc('value')] = value
                    self.log.iloc[idx + 1, self.log.columns.get_loc('breach')] = breach
                except IndexError:
                    print('max index reached, iloc cannot expand the dataframe', self.log)

            elif idx is None:
                self.log.iloc[0, self.log.columns.get_loc('dt')] = as_of
                self.log.iloc[0, self.log.columns.get_loc('stock_name')] = stock.name
                self.log.iloc[0, self.log.columns.get_loc('direction')] = direction
                self.log.iloc[0, self.log.columns.get_loc('amount')] = amount
                self.log.iloc[0, self.log.columns.get_loc('price')] = price
                self.log.iloc[0, self.log.columns.get_loc('value')] = value
                self.log.iloc[0, self.log.columns.get_loc('breach')] = breach
            else:
                raise NotImplementedError

    def update_balance_eod(self, as_of):
        if self.balance.last_valid_index():
            for stock_name in self.balance['name']:
                if type(stock_name) is str:
                    stock = Stock.get(stock_name)
                    price = stock.get_price(as_of)
                    # update price
                    self.balance.loc[self.balance['name'] == stock_name, "price"] = price
                    # update value = price * amount
                    self.balance.loc[self.balance['name'] == stock_name, "value"] = \
                        price * self.balance.loc[self.balance['name'] == stock_name, "amount"]
        self.update_total_portfolio_value_hist(as_of)
        self.update_portfolio_return_hist(as_of)

    def update_number_of_stocks(self):
        self.number_of_stocks = len(self.balance['name'].unique())

    def buy(self, stock, amount=None, value=None, as_of=datetime.date.today()):
        # if not type(stock) is Stock:
        #     raise TypeError(f'Not Stock instance! type given: {type(stock)}')
        if type(as_of) is not datetime.date:
            raise TypeError(f'as_of should be datetime.date and not {type(as_of)}')
        
        if not amount:
            if not value:
                raise ValueError('Either amount or value should be not None')
        
        if amount:
            if value:
                raise ValueError('Either amound or value should be None')

        if amount:
            if amount >= 0:
                price = stock.get_price(as_of, strict=True)
                value = price * amount
            else:
                raise ValueError('buy requires a positive amount')
        
        if value:
            if value >= 0:
                price = stock.get_price(as_of, strict=True)
                amount = value / price
            else:
                raise ValueError('buy requires a positive value')


        if self.got_enough_cash(value):
            self.deduct_cash(value)
            self.update_cash_spent(value)
            self.update_balance_transaction(stock, amount, stock.sector)
            self.update_log(as_of=as_of, stock=stock, direction='buy', amount=amount, price=price, value=value, breach=False)
            print(f'{stock.name} : BUY : {amount}')
            return amount
        else:
            self.update_log(as_of=as_of, stock=stock, direction='buy', amount=amount, price=price, value=value, breach=True)
    

    def sell(self, stock, amount=None, value=None, as_of=datetime.date.today()):
        if type(stock) is not Stock:
            raise TypeError(f'Not Stock instance! type given: {type(stock)}')
        
        if not amount:
            if not value:
                raise ValueError('Either amount or value should be not None')
        
        if amount:
            if value:
                raise ValueError('Either amound or value should be None')

        if amount:
            if amount <= 0:
                price = stock.get_price(as_of, strict=True)
                value = price * amount
            else:
                raise ValueError('sell requires a negative amount')
        
        if value:
            if value <= 0:
                price = stock.get_price(as_of, strict=True)
                amount = value / price
            else:
                raise ValueError('sell requires a negative value')

        # amount is negative, value is negative, price is positive
        if self.got_enough_amount(stock, amount):
            self.add_cash(abs(sell_value))
            self.update_cash_spent(sell_value)
            self.update_balance_transaction(stock, amount, stock.sector)
            self.update_log(as_of=as_of, stock=stock, direction='sell', amount=amount, price=sell_price, value=sell_value, breach=False)
            print(f'{stock.name} : SELL : {amount}')
            return amount
        else:
            self.update_log(as_of=as_of, stock=stock, direction='sell', amount=amount, price=sell_price, value=sell_value, breach=True)


    def calc_variance(self, lookback=None):
        """calculates total portfolio variance based on daily portfolio returns

        args:
        lookback: if None global variance is calc'd for each date, e.g. var(data[:current_date])
                  if int then the global_var and a rolling.var() is calc'd for each date
        """
        # check if returns exist
        if 'stock_return' not in self.data.columns:
            user_input = str(input('stock_return does not exist, want to calculate: y/n'))
            if user_input.upper() in ['YES', 'Y']:
                self.calc_return()
            else:
                raise InterruptedError('use calc_return before calc_variance')

        # check if variance_global exists
        if 'variance_global' in self.data.columns:
            user_input = str(input('variance_global already exists, want to recalculate: y/n'))
        else:
            user_input = "Y"

        # if not in it or user wants to recalculate
        if 'variance_global' not in self.data.columns or user_input.upper() in ['YES', 'Y']:
            variance_lst = []
            for current_date in self.data.index:
                # ddof=1 to be consistent with the default degrees-of-freedom of pd.rolling.var
                vari = np.var(self.data.loc[:current_date, 'stock_return'].dropna(), ddof=1)
                variance_lst.append(vari)
            self.data['variance_global'] = variance_lst

        # here we have 'variance_global' for sure, so we can use it to replace the first nan entries created by rolling
        if lookback:
            if f'variance_{lookback}' in self.data.columns:
                user_input = str(input(f'variance_{lookback} already exists, want to recalculate: y/n'))

            if f'variance_{lookback}' not in self.data.columns or user_input.upper() in ['YES', 'Y']:
                self.data[f'variance_{lookback}'] = self.data['stock_return'].rolling(lookback).var()
                self.data[f'variance_{lookback}'].mask(self.data[f'variance_{lookback}'].isna(),
                                                       self.data['variance_global'], inplace=True)

    def update_total_portfolio_value_hist(self, current_date):
        """updates the list of historical portfolio values, needs to be updated every EoD
        update_balance_eod triggers / calls this function
        """
        self.total_portfolio_value_hist.append((current_date, self.total_portfolio_value))

    def update_portfolio_return_hist(self, current_date):
        """updates the list of historical portfolio returns, needs to be updated every EoD
        update_balance_eod triggers / calls this function

        the return is calc'd from the history of total_portfolio_values
        total_portfolio_value is calc'd from cash_current + balance
        """
        if self.total_portfolio_value_hist:

            portfolio_value_today, today_idx = \
                [(item[1], self.total_portfolio_value_hist.index(item)) for item in self.total_portfolio_value_hist \
                 if item[0] == current_date][0]

            if len(self.total_portfolio_value_hist) >= 2:
                previous_idx = today_idx - 1
                portfolio_value_yesterday = self.total_portfolio_value_hist[previous_idx][1]
                self.portfolio_return_hist.append((current_date, portfolio_value_today / portfolio_value_yesterday))
            else:
                self.portfolio_return_hist.append((current_date, 1))
        else:
            self.portfolio_return_hist.append((current_date, 1))

