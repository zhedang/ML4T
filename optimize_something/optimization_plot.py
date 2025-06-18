""""""
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
  		  	   		 	 	 			  		 			     			  	 
Student Name: Zhe Dang (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: zdang31 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 904080678 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""
def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "zdang31"  # replace tb34 with your Georgia Tech username.

def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 904080678  # replace with your GT ID number

def study_group():
    return "zdang31"

import datetime as dt

import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from util import get_data, plot_data


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality

def f(weights,nps):
    port_np=np.dot(nps,weights)
    port_drts=port_np[1:]/port_np[:-1]-1
    avg_drts=np.mean(port_drts)
    std_drts=np.std(port_drts,ddof=1)
    return -np.sqrt(252)*avg_drts/std_drts

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
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    nums_row=len(syms)
    nums_col=len(prices)
    # construct normalized array
    normed_prices= prices.div(prices.iloc[0])
    np_prices= normed_prices.to_numpy()

    #use optimizer
    bnds = tuple((0, 1) for i in range(nums_row))
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1},)
    allocs_test=np.full(nums_row,1/nums_row)
    max_sharpe = spo.minimize(f, allocs_test, args=(np_prices,),bounds=bnds,constraints=cons,method='SLSQP', options={'disp': True})
 # add code here to compute stats
    allocs=max_sharpe.x
    portfolio_weighted_prices=np.dot(np_prices,allocs)
    portfolio_daily_returns = portfolio_weighted_prices[1:] / portfolio_weighted_prices[:-1] - 1
    cr=np.dot((np_prices[-1]-np_prices[0]),allocs)
    adr=np.mean(portfolio_daily_returns)
    sddr=np.std(portfolio_daily_returns,ddof=1)
    sr=np.sqrt(252)*adr/sddr
    """
    Plot portfolio vs SPY
    """
    port_SPY = (prices_SPY / prices_SPY[0]).to_frame()
    port_val = np.dot(normed_prices,allocs)
    port_SPY['port']=port_val


    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        #df_temp = pd.concat([port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1)
        plt.figure(figsize=(12, 8))
        plt.plot(port_SPY.index, port_SPY["port"], label="Portfolio")
        plt.plot(port_SPY.index, port_SPY["SPY"], label="SPY")
        plt.title('Daily Portfolio Value and SPY', fontsize=14)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.savefig("Figure1.png")

        pass

    return allocs, cr, adr, sddr, sr


def test_code():
    """  		  	   		 	 	 			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 			  		 			     			  	 
    """

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
