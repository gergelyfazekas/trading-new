from stock_class import Stock, StockList
# constants
from config import ticker_list, bronze, silver, gold, forecast_folder, model_folder, model_meta_file, portfolio_folder
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


s = StockList(ticker_list)

# calculate indicators
""" s.clear_data()
s.load_data(path=gold)
for ticker, stock in s.stocks.items():
    stock.tech_level_input_calc(150, tech_width=0.0005, volume_prominence=2, rel_height=2)
    s.save_data(path=gold)
    print(f'Saved {ticker}') """




from concurrent.futures import ProcessPoolExecutor, as_completed

def calc_indicator(ticker, stock_data):
    stock_data.tech_level_input_calc(
        150, tech_width=0.0005, volume_prominence=2, rel_height=2
    )
    return ticker, stock_data

def main():
    # Setup
    s.clear_data()
    s.load_data(path=gold)

    # Submit parallel jobs
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(calc_indicator, ticker, stock)
            for ticker, stock in s.stocks.items()
        ]

        results = {}
        for future in as_completed(futures):
            ticker, updated_stock = future.result()
            results[ticker] = updated_stock
            print(f"Calculated {ticker}")

    # Replace the original stocks with updated ones
    s.stocks = results
    s.save_data(path=gold)

if __name__ == '__main__':
    main()
