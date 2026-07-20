# ---------------------------------------------- Data ----------------------------------------------
    + 1000 tickers * 10 years * 365 days = 3_650_000 prices
    + 100 metrics (moving average with different windows, rsi, different returns, variances, P/E)
    + a few quasi-constants (sector, latest earnings, profit)
    --------------------
    = 3.6 million rows * 120 columns

We query the data in two ways. One stock over time, and multiple stock for the same point in time. The first one is dominant.
Most of the calculations for the metrics (sma20, sma50) and the indicators (when the two smas cross) are stock based. Do we have to calculate the metrics though (is it not  available on Interactive Brokers)?
If we calculate something accross stocks (like sector based indicators) then we might use the second querying type.
Plus for the final buy/sell every day we consider the whole stock list and our whole portfolio so that falls under the second query type.

The question is: does it make sense to keep the same data in two different structures? One partitioned by ticker and the other partitioned by time. If so, we have to make sure that the two are in sync all the time for example by only treating one as the gold standard and deriving the other from that using simple queries. 
CSV is probably enough stored in S3. Then we would have a ~3000-4000 row csv for each stock which is easy enough to read into a pandas dataframe. 

The data gathering module could look like this:

 - Historical data gathering python scripts in one folder. This should only be run once, to get the initial data for the past 10 years. After that these scripts only exist to recreate the whole thing if needed.
 
 - Daily data gathering scripts in another folder. These scripts should run once every day (maybe even hourly or minutely in the future) get new stock prices and append the data to the historical collection. This is the part where we manage the Interactive Brokers data API and where we manage the files on S3. Therefore this part should provide the data for other algorithms/engines and persist the data as well. 
 Cronjob triggers a daily update (either running on an EC2 or as Lambda functions) -> Get the prices from the API -> Load the files from S3 into memory ?? -> Append the new data -> signal that the data is ready for use (while still in memory) -> write back to S3 -> Signal that new data was persisted (this could trigger the backup)

- Backup scripts (EC2 or Lambda): either cron or triggered by the daily update.

- Transform: take the gold standard data created by the daily update and transform it to the other partitioning.
Do the same steps: load -> signal that the transform is ready to use in memory -> write -> signal that the write is finished

- Imputation?? There will be missing data historically and we might want to fill the gaps with some method. I don't know if we want to treat the filled data as the gold standard or just fill the missing when needed for an algorithm. This can be a part of the transform folder/logic.

To keep the daily data gathering as simple and fault tolerant as possible we should probably get the price for all tickers for the day and write this as a csv. This would make the time based partitioning the gold standard. But if one stock needs multiple API calls to return a price, then this would block the downstream calculations. Plus the ticker based partitioning is more important so gathering the data that way and treating that as the gold standard would make sense.

The data gathering is not crucial beacuse we can always get the yesterdays data retrospectively. One thing we have to make sure is that if we own a stock and we could not get a price for it today then that should sound an alarm because we cannot measure what the portfolio value is (although if we only trade stocks of big companies then we should not be afraid of huge price drops unlike for unknown penny stocks). It might make sense to have a backup price API that we only use if the primary failed and only to measure if there is something terribly wrong but not to persist this price data in the database.

Depending on what latency we can tolerate between the price API request and our buy/sell we might want to use an EC2 to gather the data and then do the calculations and buy/sell. Otherwise we can separate the different building blocks and run most of it with Lambda functions. If we go the EC2 route, we have to make sure that the system is up and running beacuse if the machine fails then we cannot trade and could get stuck with a bad position. This means logging and alerting (Cloudwatch to see the EC2 metrics and SNS to send notifications). depending on how Lambda works we might have to hook up alerting to that too although we only have to monitor the scripts failure and not the infrastructure (with an EC2 we have to monitor both).

Even if we go serverless, we might want to inspect the data sometimes so we either have to set up a small EC2 for manual investigations and brainstorming or we have to plug our data into Athena (or some other AWS SQL engine).

Another storage type could be a serverless SQL engine. That way we could store the data in one large table and query it both ways just by filtering for time ranges or stock symbols. This can also work since the total data size is quite small. Have to look at the AWS pricing though. We could also implement a MySQL server on a dedicated EC2 and query that from other machines or Lambda functions.



# ---------------------------------------------- Forecast ----------------------------------------------
Ask different AIs to list 50 metics and see if we can query it from the Interactive Brokers API. Indicators will have to be calculated by us. Indicators have to be updated every day so that the forecasting model can be fed even if the model parameters are not updated every day. Recalculating the model parameters from scratch takes a lot of resources so either have a model that can be updated incrementally (kind of Bayesian where the initial coefficients are updated as new data comes in), or we can only update the model once in a while.

The forecast model will predict some form of return (1-5-10-20 day, log-return vs normal return). 
We can either predict one timeframe at once (e.g. 20 day) and then use that as an input for predicting a lower timescale (e.g. 10 day). Or we can predict all timeframes (1,5,10,20) using a single model with multiple outputs.
What we must do though is use panefl information, so predicting the either one timescale of return for all stocks or multiple timescales for all stocks. Therefore the model should have multiple output nodes even if we only predict one timescale (e.g. 5 day return for AAPL, MSFT, ...). This is called 
1.12.4. Multioutput regression in https://scikit-learn.org/stable/modules/multiclass.html.


Is it better to predict the category, let's say the stock will be in the top 10 performers today vs bottom 10? In that case stock with outstanding returns (like 10% in a day) would not produce a huge loss even if we predicted 6% because it ends up in the top 10 either way.

We should probably predict volatility in the next 5, 10, 20 days. And maybe volume too. 




# 2025-07-13
rewrite the engine.ipynb so that the engine class has two kinds of weights. One will be used to determine the cash proportion that we will invest on a single day (this param times market data will give the proportion). The other set of params will affect how to split the investment among the stocks. 

Still not good: we can set how much we want to spend tomorrow and split that across stocks but how to determine how much to sell? If we have cash and we spend 30% of it, and we want to split it as 80% AAPL and 20% AMZN, then we calculate how much is the 30% (x dollars) and what amount does that equal to if it's split between AAPL and AMZN (let's say the x dollars will equal 2 AAPL shares and 2 AMZN and this way it is 80%-20% given the prices of these stocks). 
What is the equivalent in selling? We want to sell 40% of the portfolio value and we want to split it as 70% MSFT and 30% AMZN. We calculate what the 40% is and how much does it mean in MSFT and AMZN quantity if we respect the split. Before selling we net this quantity with the buy side quantities to minimize transactions. 


dollar value invested * proportion to divest * stock's weight for divestment
dollar value in cash * proportion to invest * stock's weight for investment


# 2025-08-20
The portfolio includes the decision engine as well. We want to detrermine the desired composition of tomorrow's portfolio.
{'cash': 0.2, 'AAPL':0.5, 'AMZN': 0.3} 
We relate this to the current state of the portfolio and trade the difference. If the difference that should be traded is small then we don't do that one. 

y = softmax(XB)

cash | [forecast_cash, risk_cash, current_share_cash] x [forecast_beta, risk_beta, current_share_beta]
AAPL | [forecast_AAPL, risk_AAPL, current_share_AAPL] x [forecast_beta, risk_beta, current_share_beta]
AMZN | [forecast_AMZN, risk_AMZN, current_share_AMZN] x [forecast_beta, risk_beta, current_share_beta]

