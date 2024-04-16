import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

from datetime import datetime
from dateutil.relativedelta import relativedelta

# Suppose you start with this date:
start_date = datetime.strptime("2023-01-01", "%Y-%m-%d")

# Move the date forward by 6 months:
new_date = start_date + relativedelta(months=6)

# Convert the new date back to a string:
date_string = new_date.strftime("%Y-%m-%d")
print(date_string)  # Output will be "2023-07-01"


def run_strategy(start="2015-01-01", end="2021-01-01", error="hdr", flag="backtest", train_period=1, test_period=0.5):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    total_portfolio_value = 1  # Start with a normalized value of 1
    
    while start_date < end_date:
        backtest_start = start_date + relativedelta(years=train_period)
        backtest_end = backtest_start + relativedelta(months=6)  # Running the strategy every 6 months

        if backtest_end > end_date:
            backtest_end = end_date
        
        ticker_returns, index_returns = preparePipeline(backtest_start.strftime("%Y-%m-%d"), test_period)
        
        if error == "hdr":
            portfolio_weights = track_hdr(ticker_returns, index_returns, reg=0.2, u=0.5, hub=0.05)
        elif error == "ete":
            portfolio_weights = track(ticker_returns, index_returns, reg=0.2)
        else:
            raise Exception("Invalid tracking error!")
        
        returns_df = pd.DataFrame(ticker_returns.numpy())
        index_returns_df = pd.DataFrame(index_returns.numpy())
        
        non_zero_weights = portfolio_weights[portfolio_weights > 0]
        filtered_returns_df = returns_df.loc[:, non_zero_weights.index]
        weighted_returns = filtered_returns_df.mul(non_zero_weights.values.flatten(), axis=1)
        daily_portfolio_returns = weighted_returns.sum(axis=1)
        
        period_return = daily_portfolio_returns.cumsum().iloc[-1] / 100  # Assuming returns are percentages
        total_portfolio_value *= (1 + period_return)  # Compound the returns

        start_date = backtest_end
    
    # Visualization of overall returns (optional)
    plt.figure(figsize=(10, 6))
    plt.plot([total_portfolio_value], label='Aggregate Portfolio Returns', color='blue')
    plt.title("Total Aggregate Returns")
    plt.xlabel('Periods')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return total_portfolio_value - 1  # Subtract 1 to get the net total return in percentage

# Note: The above code assumes that the daily returns data is formatted as percentage changes.
