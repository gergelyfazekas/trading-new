import asyncio
from ib_insync import IB, Stock, MarketOrder, Forex, Ticker, Trade
import datetime
import math
import time

class IBroker:
    def __init__(self):
        self.ib = IB()

    def connect(self):
        # Connect to IB Gateway or TWS
        self.ib.connect('127.0.0.1', 4002, clientId=1)

    def disconnect(self):
        self.ib.disconnect()

    def get_forex_price(self, pair='EURUSD'):
        fx = Forex(pair)
        self.ib.qualifyContracts(fx)
        market_data = self.ib.reqMktData(fx, '', True, False)
        self.track_events(market_data, label='get_forex_price')
        price = market_data.marketPrice()
        self.ib.cancelMktData(fx)
        return price

    def buy_currency(self, sell: str, buy: str, sell_quantity: float):
        """
        Buy currency by selling another. The quantity is in the selling currency. 
        sell='EUR', buy='USD', sell_quantity=10
        """
        pair = f"{sell}{buy}"
        fx = Forex(pair)
        self.ib.qualifyContracts(fx)
        
        # To buy USD by selling EUR, you SELL EUR.USD
        order = MarketOrder('SELL', round(sell_quantity))
        trade = self.ib.placeOrder(fx, order)
        self.ib.waitOnUpdate(timeout=5)
        return trade

    def get_market_data(self, ticker, snapshot=False):
        stock = Stock(ticker, 'SMART', 'USD')
        self.ib.qualifyContracts(stock)
        market_data_type = self.ib.reqMarketDataType(3)
        market_data = self.ib.reqMktData(stock, '', snapshot, False)

        # Wait for market data to be available
        self.ib.waitUntil(datetime.datetime.now() + datetime.timedelta(seconds=1))
        
        # wait another 3 seconds if we haven't received the price yet
        if math.isnan(market_data.marketPrice()) or math.isnan(market_data.volume) :
            self.ib.waitUntil(datetime.datetime.now() + datetime.timedelta(seconds=2))
        price, volume = market_data.marketPrice(), market_data.volume
        self.ib.cancelMktData(stock)
        return price, volume

    def buy_stock(self, ticker, quantity):
        stock = Stock(ticker, 'SMART', 'USD') 
        self.ib.qualifyContracts(stock)
        order = MarketOrder('BUY', quantity)
        trade = self.ib.placeOrder(stock, order)
        self.track_events(trade, label='BUY')
        return trade
    
    def track_events(self, event, label, cnt=1, max_cnt=6, timeout=10, verbose=True):
        self.ib.waitOnUpdate(timeout=timeout)

        if isinstance(event, Ticker):
            if verbose & cnt==1:
                print(label, ' - ', 'Ticker event: ', event.contract.pair())
        elif isinstance(event, Trade):
            if verbose & cnt==1:
                print(label, ' - ', 'Trade event: ', event.contract.symbol)

            states = ['PendingSubmit', 'Submitted', 'PreSubmitted', 'Filled', 'PartiallyFilled', 'Cancelled', 'Inactive', 'ApiCancelled', 'ApiPending']
            state = event.orderStatus.status
            print('State: ', state)
            if (state != 'Filled') & (cnt < max_cnt):
                self.track_events(event, label=label, cnt=cnt+1)
            elif (state != 'Filled') & (cnt >= max_cnt):
                print('Count reached. Quitting.')
            else:
                print('Filled.')
        else:
            print('Else')

    def sell_stock(self, ticker, quantity):
        stock = Stock(ticker, 'SMART', 'USD')
        self.ib.qualifyContracts(stock)
        order = MarketOrder('SELL', quantity)
        trade = self.ib.placeOrder(stock, order)
        self.track_events(trade, label='SELL')
        return trade
    
    def get_portfolio(self):
        """
        Fetch the current portfolio positions.
        Returns a list of Position objects.
        """
        portfolio = self.ib.portfolio()
        return portfolio

    def get_cash(self, currency):
            """
            Retrieve the available cash in the account.
            Uses accountValues to find the 'TotalCashBalance' in a given currency.
            """
            account_values = self.ib.accountValues()
            cash = next((av.value for av in account_values if av.tag == 'TotalCashBalance' and av.currency == currency), None)
            return cash

if __name__ == '__main__':
    broker = IBroker()
    broker.connect()
    ticker = 'AAPL'

    #broker.get_forex_price(pair='EURUSD')
    #print('FX rate: ', fx_rate*stock_price*2)

    #status = broker.buy_currency(sell='EUR', buy='USD', sell_quantity=400)
    #result = broker.sell_stock(ticker, quantity=1)
    #price = broker.get_market_data(ticker, snapshot=False)
    #print('AAPL: ', price)


    portfolio = broker.get_portfolio()
    print('Cash: ', broker.get_cash(currency='USD'), 'USD ,', broker.get_cash(currency='EUR'), 'EUR')
    print('Portfolio: ')
    print(portfolio)