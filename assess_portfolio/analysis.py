"""Analyze a portfolio.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2017, Georgia Tech Research Corporation  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332-0415  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
# This is the function that will be tested by the autograder  		  	   		 	   		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality
# some changes
def assess_portfolio(  		  	   		 	   		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	   		  		  		    	 		 		   		 		  
    allocs=[0.1, 0.2, 0.3, 0.4],  		  	   		 	   		  		  		    	 		 		   		 		  
    sv=1000000,  		  	   		 	   		  		  		    	 		 		   		 		  
    rfr=0.0,  		  	   		 	   		  		  		    	 		 		   		 		  
    sf=252.0,  		  	   		 	   		  		  		    	 		 		   		 		  
    gen_plot=False,  		  	   		 	   		  		  		    	 		 		   		 		  
):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Estimate a set of test points given the model we built.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
    :param syms: A list of 2 or more symbols that make up the portfolio (note that your code should support any symbol in the data directory)  		  	   		 	   		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		 	   		  		  		    	 		 		   		 		  
    :param allocs:  A list of 2 or more allocations to the stocks, must sum to 1.0  		  	   		 	   		  		  		    	 		 		   		 		  
    :type allocs: list  		  	   		 	   		  		  		    	 		 		   		 		  
    :param sv: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
    :type sv: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :param rfr: The risk free return per sample period that does not change for the entire date range (a single number, not an array)  		  	   		 	   		  		  		    	 		 		   		 		  
    :type rfr: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param sf: Sampling frequency per year  		  	   		 	   		  		  		    	 		 		   		 		  
    :type sf: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	   		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: A tuple containing the cumulative return, average daily returns,  		  	   		 	   		  		  		    	 		 		   		 		  
        standard deviation of daily returns, Sharpe ratio and end value  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    # Generate allocs of 1/n, where n is number of symbols
    allocs = np.full(len(syms), 1 / len(syms))

    # add code here to find the allocations
    cr, adr, sddr, sr = [
        0.25,
        0.001,
        0.0005,
        2.1,
    ]

    # Get initial daily return
    norm_price = normalize(prices)
    alloc_price = (norm_price * allocs) * sv
    port_val = alloc_price.sum(axis=1)
    daily_ret = compute_daily_returns(port_val)

    # Computes initial values with initial data
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    k = np.sqrt(252).astype(np.float64)
    sr = (adr / sddr) * k
    # print("initial: ", cr, adr, sddr, sr, allocs)

    # Optimizer - minimize
    constraints = {'type': 'eq', 'fun': lambda allocs: 1- allocs.sum()}
    bounds = [(0, 1) for _ in range(allocs.shape[0])]
    optimizer = minimize(negative_sharpe_ratio, allocs, args=(adr, sddr, norm_price, k, sv),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    # print("Optimizer: ", optimizer)

    # Get best daily portfolio value
    allocs = optimizer.x
    # print(norm_price)
    best_alloc_val = (norm_price * allocs) * sv
    best_port_val = best_alloc_val.sum(axis=1)
    best_daily_ret = compute_daily_returns(best_port_val)
    # print(best_daily_ret)

    # Updating return values
    cr = (best_port_val[-1] / best_port_val[0]) - 1
    adr = best_daily_ret.mean()
    sddr = best_daily_ret.std()
    sr = k * (adr / sddr)
    # print("final: ", cr, adr, sddr, sr, allocs)

    norm_SPY = prices_SPY / prices_SPY.iloc[0]
    norm_SPY[0] = 1.0

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	   		  		  		    	 		 		   		 		  
    if not gen_plot:
        # add code to plot here  		  	   		 	   		  		  		    	 		 		   		 		  
        df_temp = pd.concat(
            [best_daily_ret, norm_SPY], keys=["Portfolio", "SPY"], axis=1
        )

        plt.plot(df_temp)
        plt.title("Daily Portfolio Value")
        plt.xlabel("Dates")
        plt.ylabel("Price")
        plt.legend(["Portfolio", "SPY"])
        plt.show()
        # plt.savefig('./images/plot.png')
        plt.clf()
        pass  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Add code here to properly compute end value  		  	   		 	   		  		  		    	 		 		   		 		  
    ev = sv  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    return cr, adr, sddr, sr, ev

def negative_sharpe_ratio(allocs, adr, sddr, norm_price, k, sv):
    new_alloc_price = (norm_price * allocs)* sv
    new_port_val = new_alloc_price.sum(axis=1)
    new_daily_ret = compute_daily_returns(new_port_val)

    # add code here to compute stats
    new_adr = new_daily_ret.mean()
    new_sddr = new_daily_ret.std()
    sharpe_ratio = (new_adr/ new_sddr) * k
    return -sharpe_ratio

def normalize(df):
    """Compute and return the normalized price values."""
    normalize_prices = df.copy()
    for i in range(1, normalize_prices.shape[0]):
        for j in range(normalize_prices.shape[1]):
            normalize_prices.iloc[i, j] = df.iloc[i, j] / df.iloc[0, j]
    normalize_prices.iloc[0, :] = 1.0
    return normalize_prices

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    # daily_returns = (df / df.shift(1)) - 1 # much easier with Pandas!
    daily_returns.iloc[0] = 0 # Pandas leaves the 0th row full of Nans
    return daily_returns
  		  	   		 	   		  		  		    	 		 		   		 		  
def test_code():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Performs a test of your code and prints the results  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    # This code WILL NOT be tested by the auto grader  		  	   		 	   		  		  		    	 		 		   		 		  
    # It is only here to help you set up and test your code  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		 	   		  		  		    	 		 		   		 		  
    # Note that ALL of these values will be set to different values by  		  	   		 	   		  		  		    	 		 		   		 		  
    # the autograder!  		  	   		 	   		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2009, 1, 1)  		  	   		 	   		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2010, 1, 1)  		  	   		 	   		  		  		    	 		 		   		 		  
    symbols = ["GOOG", "AAPL", "GLD", "XOM"]  		  	   		 	   		  		  		    	 		 		   		 		  
    allocations = [0.2, 0.3, 0.4, 0.1]  		  	   		 	   		  		  		    	 		 		   		 		  
    start_val = 1000000  		  	   		 	   		  		  		    	 		 		   		 		  
    risk_free_rate = 0.0  		  	   		 	   		  		  		    	 		 		   		 		  
    sample_freq = 252  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
    cr, adr, sddr, sr, ev = assess_portfolio(  		  	   		 	   		  		  		    	 		 		   		 		  
        sd=start_date,  		  	   		 	   		  		  		    	 		 		   		 		  
        ed=end_date,  		  	   		 	   		  		  		    	 		 		   		 		  
        syms=symbols,  		  	   		 	   		  		  		    	 		 		   		 		  
        allocs=allocations,  		  	   		 	   		  		  		    	 		 		   		 		  
        sv=start_val,  		  	   		 	   		  		  		    	 		 		   		 		  
        gen_plot=False,  		  	   		 	   		  		  		    	 		 		   		 		  
    )  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Allocations: {allocations}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	   		  		  		    	 		 		   		 		  
