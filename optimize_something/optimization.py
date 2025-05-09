""""""
import time

"""MC1-P2: Optimize a portfolio.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Student Name: Jason Zheng (replace with your name)  		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: jzheng429 (replace with your User ID)  		  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 903510650 (replace with your GT ID)  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import datetime as dt

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt  		  	   		 	   		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 	   		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
# This is the function that will be tested by the autograder  		  	   		 	   		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		 	   		  		  		    	 		 		   		 		  
def optimize_portfolio(  		  	   		 	   		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	   		  		  		    	 		 		   		 		  
    gen_plot=False,
):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	   		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	   		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	   		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	   		  		  		    	 		 		   		 		  
    statistics.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	   		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		 	   		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		 	   		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	   		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	   		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	   		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		 	   		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	   		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols
    fill_missing_values(prices)
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
    fill_missing_values(prices_SPY)

    # find the allocations for the optimal portfolio  		  	   		 	   		  		  		    	 		 		   		 		  

    # Generate allocs of 1/n, where n is number of symbols
    allocs = np.full(len(syms), 1/len(syms))

    # add code here to find the allocations
    cr, adr, sddr, sr = [
        0.25,  		  	   		 	   		  		  		    	 		 		   		 		  
        0.001,  		  	   		 	   		  		  		    	 		 		   		 		  
        0.0005,  		  	   		 	   		  		  		    	 		 		   		 		  
        2.1,  		  	   		 	   		  		  		    	 		 		   		 		  
    ]

    # Get initial daily return
    norm_price = normalize(prices)
    alloc_price = norm_price*allocs
    port_val = alloc_price.sum(axis = 1)
    daily_ret = compute_daily_returns(port_val)

    # Computes initial values with initial data
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    k = np.sqrt(252).astype(np.float64)
    sr = (adr/sddr) * k
    # print("initial: ", cr, adr, sddr, sr, allocs)

    # Optimizer - minimize
    constraints = {'type': 'eq', 'fun': lambda allocs: 1- allocs.sum()}
    bounds = [(0, 1) for _ in range(allocs.shape[0])]
    optimizer = minimize(negative_sharpe_ratio, allocs, args=(adr, sddr, norm_price, k),
                  method='SLSQP', bounds=bounds, constraints=constraints)
    # print("Optimizer: ", optimizer)

    # Get best daily portfolio value
    allocs = optimizer.x
    # print(norm_price)
    best_alloc_val = norm_price * allocs
    best_port_val = best_alloc_val.sum(axis=1)
    best_daily_ret = compute_daily_returns(best_port_val)
    # print(best_daily_ret)

    # Updating return values
    cr = (best_port_val[-1] / best_port_val[0]) - 1
    adr = best_daily_ret.mean()
    sddr = best_daily_ret.std()
    sr = k * (adr / sddr)
    # print("final: ", cr, adr, sddr, sr, allocs)

    norm_SPY = prices_SPY/prices_SPY.iloc[0]
    norm_SPY[0] = 1.0

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	   		  		  		    	 		 		   		 		  
    if gen_plot:
        # add code to plot here  		  	   		 	   		  		  		    	 		 		   		 		  
        df_temp = pd.concat(  		  	   		 	   		  		  		    	 		 		   		 		  
            [best_port_val, norm_SPY], keys=["Portfolio", "SPY"], axis=1
        )

        plt.plot(df_temp)
        plt.title("Daily Portfolio Value And SPY")
        plt.xlabel("Dates")
        plt.ylabel("Normalized Price")
        plt.legend(["Portfolio", "SPY"])
        # plt.show()
        plt.savefig('./images/Figure_1.png')
        plt.clf()
        pass  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    return allocs, cr, adr, sddr, sr

def negative_sharpe_ratio(allocs, adr, sddr, norm_price, k):
    new_alloc_price = norm_price * allocs
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
    normalize_prices = df.div(df.iloc[0, :])
    normalize_prices.iloc[0, :] = 1.0
    return normalize_prices

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    # daily_returns = (df / df.shift(1)) - 1 # much easier with Pandas!
    daily_returns.iloc[0] = 0 # Pandas leaves the 0th row full of Nans
    return daily_returns

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method="ffill", inplace=True)
    df_data.fillna(method="bfill", inplace=False)

def test_code():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    # symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]
    symbols = ['IBM', 'X', 'GLD', 'JPM']
    # Assess the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	   		  		  		    	 		 		   		 		  
        sd=start_date, ed=end_date, syms=symbols, gen_plot=False  		  	   		 	   		  		  		    	 		 		   		 		  
    )  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		 	   		  		  		    	 		 		   		 		  
    # print(f"Start Date: {start_date}")
    # print(f"End Date: {end_date}")
    # print(f"Symbols: {symbols}")
    # print(f"Allocations:{allocations}")
    # print(f"Sharpe Ratio: {sr}")
    # print(f"Volatility (stdev of daily returns): {sddr}")
    # print(f"Average Daily Return: {adr}")
    # print(f"Cumulative Return: {cr}")


def author():
    """
    To be implemented beginnging Summer 2024

    Returns
        The GT username of the student

    Return type
        str
    """
    return "jzheng429"


def study_group():
    """
    Returns
        A comma separated string of GT_Name of each member of your study group
        # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone

    Return type
        str
    """
    return  "jzheng429"
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		  	   		 	   		  		  		    	 		 		   		 		  
    # Do not assume that it will be called
    # Record the start time
    # start_time = time.time()
    #
    # # Call the function
    # test_code()
    #
    # # Record the end time
    # end_time = time.time()
    #
    # # Calculate the elapsed time
    # elapsed_time = end_time - start_time
    #
    # print('time: ', elapsed_time)

    test_code()