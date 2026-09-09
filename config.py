bronze = '/Users/gergelyfazekas/Documents/python_projects/trading_new_structure/data/bronze'
silver = '/Users/gergelyfazekas/Documents/python_projects/trading_new_structure/data/silver'
gold = '/Users/gergelyfazekas/Documents/python_projects/trading_new_structure/data/gold'
forecast_folder = "/Users/gergelyfazekas/Documents/python_projects/trading_new_structure/data/forecast"
model_folder = "/Users/gergelyfazekas/Documents/python_projects/trading_new_structure/data/models"
model_meta_file = model_folder + '/meta.csv'
portfolio_folder = "/Users/gergelyfazekas/Documents/python_projects/trading_new_structure/data/portfolios"
tech_level_folder = "/Users/gergelyfazekas/Documents/python_projects/trading_new_structure/data/tech_levels"

# Validation universe: 60 large caps disjoint from ticker_list, same date range.
# Reserved for out-of-sample confirmation only -- nothing here may be used to
# construct a signal or choose a threshold, or it stops being a fresh
# cross-section. Raw pull: callers must run calc_return themselves.
oos = "/Users/gergelyfazekas/Documents/python_projects/trading_new_structure/data/oos"
oos_ticker_list = [
    'NVDA', 'TSLA', 'META', 'NFLX', 'PEP', 'WMT', 'TGT', 'LOW', 'UNH', 'PFE',
    'LLY', 'BMY', 'GILD', 'CVS', 'TMO', 'MDT', 'SYK', 'BDX', 'ISRG', 'ORCL',
    'CRM', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO', 'MU', 'NOW', 'INTU', 'MA',
    'V', 'WFC', 'MS', 'BLK', 'SCHW', 'USB', 'PNC', 'SO', 'DUK', 'NEE',
    'UPS', 'FDX', 'UNP', 'CSX', 'RTX', 'GD', 'NOC', 'EMR', 'ITW', 'PG',
    'CL', 'KMB', 'GIS', 'SBUX', 'YUM', 'TJX', 'ORLY', 'ADP', 'ECL', 'APD',
]

ticker_list = [
    'MMM','ABT','ABBV','ACN','ADBE','GOOG','MO','AMZN','AXP','AEP','AMGN','AAPL','AMAT','ADM','T','BAC','BA','CAT','CVX','CSCO','C','KO','COP',
    'COST',
    'DHR',
    'DE',
    'DOV',
    'XOM',
    'GS',
    'HD',
    'HON',
    'IBM',
    'JNJ',
    'JPM',
    'LMT',
    'MCD',
    'MRK',
    'MSFT',
    'NKE',
    'DIS',
    'GOOGL',
]

