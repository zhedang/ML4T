"""Analyze a portfolio.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2017, Georgia Tech Research Corporation  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332-0415  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import datetime as dt  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
import scipy.optimize as spo
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
# This is the function that will be tested by the autograder  		  	   		 	 	 			  		 			     			  	 
# The student must update this code to properly implement the functionality  		  	   		 	 	 			  		 			     			  	 
def f(weights,nps):
    port_np=np.dot(nps,weights)
    port_drts=port_np.copy()
    port_drts[1:]=port_np[1:]/port_np[:-1]-1
    port_drts[0]=0
    avg_drts=np.mean(port_drts)
    std_drts=np.std(port_drts)
    return -np.sqrt(252)*avg_drts/std_drts
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
  		  	   		 	 	 			  		 			     			  	 
    # Get daily portfolio value  		  	   		 	 	 			  		 			     			  	 
    port_val = prices_SPY  # add code here to compute daily portfolio values  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Get portfolio statistics (note: std_daily_ret = volatility)  		  	   		 	 	 			  		 			     			  	 
    cr, adr, sddr, sr = [  		  	   		 	 	 			  		 			     			  	 
        0.25,  		  	   		 	 	 			  		 			     			  	 
        0.001,  		  	   		 	 	 			  		 			     			  	 
        0.0005,  		  	   		 	 	 			  		 			     			  	 
        2.1,  		  	   		 	 	 			  		 			     			  	 
    ]  # add code here to compute stats  		  	   		 	 	 			  		 			     			  	 
    nums_row = len(syms)
    nums_col = len(prices)
    allocs_test = np.full(nums_row, 1 / nums_row)
    # construct normalized array
    normed_prices = prices.div(prices.iloc[0])
    np_prices = normed_prices.to_numpy()

    # use optimizer
    bnds = tuple((0, 1) for i in range(nums_row))
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1},)
    allocs_test = np.full(nums_row, 1 / nums_row)
    max_sharpe = spo.minimize(f, allocs_test, args=(np_prices,), bounds=bnds, constraints=cons, method='SLSQP',
                              options={'disp': True})
    # add code here to compute stats
    #allocs = max_sharpe.x
    allocs=np.array([0.25,0.25,0.25,0.25])
    portfolio_weighted_prices = np.dot(np_prices,allocs)

    portfolio_daily_returns = portfolio_weighted_prices[1:] / portfolio_weighted_prices[:-1] - 1

    cr = np.dot((np_prices[-1] - np_prices[0]), allocs)
    adr = np.mean(portfolio_daily_returns)
    sddr = np.std(portfolio_daily_returns,ddof=1)
    sr = np.sqrt(252) * adr / sddr
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	 	 			  		 			     			  	 
    if gen_plot:  		  	   		 	 	 			  		 			     			  	 
        # add code to plot here  		  	   		 	 	 			  		 			     			  	 
        df_temp = pd.concat(  		  	   		 	 	 			  		 			     			  	 
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1  		  	   		 	 	 			  		 			     			  	 
        )  		  	   		 	 	 			  		 			     			  	 
        pass  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Add code here to properly compute end value  		  	   		 	 	 			  		 			     			  	 
    ev = sv  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    return cr, adr, sddr, sr, ev  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
