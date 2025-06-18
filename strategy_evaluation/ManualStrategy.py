import pandas as pd
import datetime as dt
import numpy as np
import util as ut
from indicators import *


class ManualStrategy:
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def testPolicy(self, symbol='IBM',
                   sd=dt.datetime(2009, 1, 1),
                   ed=dt.datetime(2010, 1, 1),
                   sv=100000):
        """
        Tests the manual strategy and returns a DataFrame with trades.
        Legal values: +1000 (BUY), -1000 (SELL), 0 (HOLD),
        +2000/-2000 allowed for short-to-long or long-to-short transitions.
        """

        """ 
        First, collect price and all indicator data
        """
        # take more dates bach to 3 months ago for rolling
        start = sd - pd.offsets.MonthEnd(3)
        end = ed
        date_ = pd.date_range(start, end)
        # get indicators
        df = ut.get_data([symbol], date_, addSPY=False).dropna()
        df['Price/SMA'] = df[[symbol][0]] / SMA_20(df[[symbol]])
        df['RSI'] = RSI_14(df[[symbol]])
        df['ROC'] = Momentum_14(df[[symbol]])
        df['MACD'] = MACD(df[[symbol]])
        df_c = ut.get_data([symbol], date_, addSPY=False, colname='Close').dropna()
        df_h = ut.get_data([symbol], date_, addSPY=False, colname='High').dropna()
        df_l = ut.get_data([symbol], date_, addSPY=False, colname='Low').dropna()
        df['CCI'] = CCI_20(df_c, df_h, df_l)
        # make everyday indicator based on past data
        df[['Price/SMA', 'RSI', 'CCI', 'ROC', 'MACD']] = df[['Price/SMA', 'RSI', 'CCI', 'ROC', 'MACD']].shift(1)
        # filter for original date index
        df = df.loc[sd:ed]

        """
        Second, generate signals
        """
        buy_condition = (df['Price/SMA'] < 0.8) | ((df['RSI'] < 25) & (df['CCI'] < -150))
        sell_condition = (df['Price/SMA'] > 1.0) | ((df['RSI'] > 75) & (df['CCI'] > 130))
        df['Signal'] = 0
        df.loc[buy_condition, 'Signal'] = 1
        df.loc[sell_condition, 'Signal'] = -1

        """
        Last, turn signals into trade df
        """
        trades_df = pd.DataFrame(0, index=df.index, columns=['Trade'])
        current_holding = 0  # Start with no position

        # Iterate through the predicted signals day by day
        for i in range(len(df)):
            signal = df['Signal'].iloc[i]
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

    def author(self):
        return "zdang31"  # Replace with your actual GT username

    def study_group(self):
        return "zdang31"  # Replace or extend with other usernames as needed

