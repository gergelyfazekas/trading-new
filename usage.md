1. pull_data.ipynb:
    Pulls historical data from yahoo and saves it
    Calculates metrics and indicators

2. forecast.ipynb
    reads the historical data from the gold folder and builds a multitarget random forest on it
    evaluates the models with cross validation
    fits the best model to all the data and saves the model to its folder
    adds the model parameters to a model meta file so models can be compared easier

    ORIGINAL DATA
    close   log_return      x1          x2
    1     log(1/nan)=nan    1           4
    10    log(10/1)=log(10) 2           5
    30    log(30/10)=log(3) 3           6

    FITTING y=XB --> we use today's x to map to tomorrow's y (there is a shift(-1) in y)
    log(10) = (1 4)B
    log(3) = (2 5)B

    PREDICT
    


3. portfolio.ipynb
    creates a portfolio
    we can simulate buy/sell transactions based on the gold historical data
    shows how to use the engine associated to the portfolio

4. main.py - Daily loop
    load the stocks with their data and the best portfolio
    get today's price from ibkr
    append it to the older data
    generate the 


