""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
GT ID:  (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""

import datetime as dt
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util as ut
from indicators import *
import BagLearner as bl
import RTLearner as rl


class StrategyLearner(object):

    # constructor

    def __init__(self, verbose=False, impact=0.0, commission=0.0,window=10, N=10, YBUY=0.06, YSELL=-0.05,leaf_size=5, bags=50, indicators=['Price/SMA', 'RSI', 'CCI']):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.window = window
        self.N = N
        self.YBUY = YBUY
        self.YSELL = YSELL
        self.leaf_size = leaf_size
        self.bags = bags
        self.learner = bl.BagLearner(learner=rl.RTLearner,
                                     kwargs={"leaf_size": self.leaf_size},
                                     bags=self.bags)
        self.indicators = indicators

    def author(self):
        return "zdang31"

    def study_group(self):
        return "zdang31"

    def add_evidence(
            self,
            symbol="IBM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=100000,
            impact=None,
            commission=None
    ):
        # Get price and indicators
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        start = sd - pd.offsets.MonthEnd(3)
        date_ = pd.date_range(start, ed)
        df = ut.get_data(syms, date_, addSPY=False).dropna()
        window = self.window
        df['Price/SMA'] = df[syms[0]] / SMA_20(df[syms], window=window)
        df['RSI'] = RSI_14(df[syms], window=window)
        df['ROC'] = Momentum_14(df[syms], window=window)
        df['MACD'] = MACD(df[syms])
        df_c = ut.get_data(syms, date_, addSPY=False, colname='Close').dropna()
        df_h = ut.get_data(syms, date_, addSPY=False, colname='High').dropna()
        df_l = ut.get_data(syms, date_, addSPY=False, colname='Low').dropna()
        df['CCI'] = CCI_20(df_c, df_h, df_l, window=window)
        # make everyday indicator based on past data
        df[['Price/SMA', 'RSI', 'CCI', 'ROC', 'MACD']] = df[['Price/SMA', 'RSI', 'CCI', 'ROC', 'MACD']].shift(1)
        # filter for original date index
        df = df.loc[sd:ed]

        # Get Signal column
        impact = self.impact if impact is None else impact
        commission = self.commission if commission is None else commission
        share_qty = 2000

        N = self.N
        YBUY = self.YBUY
        YSELL = self.YSELL
        prices = df.iloc[:, 0]

        cost_per_trade = 2 * impact + (commission * 2) / (prices * share_qty)
        future_return = (prices.shift(-N) / prices) - 1.0
        adj_YBUY = YBUY + cost_per_trade
        adj_YSELL = YSELL - cost_per_trade
        conditions = [future_return > adj_YBUY, future_return < adj_YSELL]
        choices = [1, -1]
        df['Signal'] = np.select(conditions, choices, default=0)

        # Prepare training data
        X_train = df[self.indicators].values
        Y_train = df['Signal'].values

        # Initialize and train the BagLearner
        self.learner.add_evidence(X_train, Y_train)

    # this method should use the existing policy and test it against new data
    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=100000,
    ):

        # 1. Get Data and Calculate Indicators (same process as add_evidence)
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        start = sd - pd.offsets.MonthEnd(3)
        date_ = pd.date_range(start, ed)
        df = ut.get_data(syms, date_, addSPY=False).dropna()
        window = self.window
        df['Price/SMA'] = df[syms[0]] / SMA_20(df[syms], window=window)
        df['RSI'] = RSI_14(df[syms], window=window)
        df['ROC'] = Momentum_14(df[syms], window=window)
        df['MACD'] = MACD(df[syms])
        df_c = ut.get_data(syms, date_, addSPY=False, colname='Close').dropna()
        df_h = ut.get_data(syms, date_, addSPY=False, colname='High').dropna()
        df_l = ut.get_data(syms, date_, addSPY=False, colname='Low').dropna()
        df['CCI'] = CCI_20(df_c, df_h, df_l, window=window)
        # make everyday indicator based on past data
        df[['Price/SMA', 'RSI', 'CCI', 'ROC', 'MACD']] = df[['Price/SMA', 'RSI', 'CCI', 'ROC', 'MACD']].shift(1)
        # filter for original date index
        df = df.loc[sd:ed]

        # Filter for the specific testing date range
        df_test = df.loc[sd:ed].copy()  # Use .copy()

        # 2. Prepare Test Features (X_test)
        X_test = df[self.indicators].values

        # 3. Query the Learner for Signals
        Y_pred_signal = self.learner.query(X_test)  # Get predicted signals (1, -1, 0)

        # 4. Convert Signals to Trades (holding +1000, -1000, or 0)
        trades_df = pd.DataFrame(0, index=df_test.index, columns=['Trade'])
        current_holding = 0  # Start with no position

        # Iterate through the predicted signals day by day
        for i in range(len(df_test)):
            signal = Y_pred_signal[i]
            trade_amount = 0

            if signal == 1:  # Predicted Buy Signal
                if current_holding == 0:
                    trade_amount = 1000
                    current_holding = 1000
                elif current_holding == -1000:
                    trade_amount = 2000  # Buy 2000 to go from -1000 to +1000
                    current_holding = 1000
                # else (if current_holding == 1000), do nothing, already long

            elif signal == -1:  # Predicted Sell Signal
                if current_holding == 0:
                    trade_amount = -1000
                    current_holding = -1000
                elif current_holding == 1000:
                    trade_amount = -2000  # Sell 2000 to go from +1000 to -1000
                    current_holding = -1000
                # else (if current_holding == -1000), do nothing, already short

            # If signal is 0, do nothing, trade_amount remains 0, current_holding remains unchanged

            trades_df.iloc[i, 0] = trade_amount  # Assign trade amount for the day

        return trades_df[['Trade']]  # Return only the 'Trade' column as required