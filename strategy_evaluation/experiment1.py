import datetime as dt
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util as ut
from indicators import *
import ManualStrategy as ms
import StrategyLearner as sl

def author():
    return "zdang31"

def study_group():
    return "zdang31"
def compute_portval(trades, prices, start_val=100000, commission=9.95, impact=0.005):
    df_trades = trades.copy()
    df_trades['Holdings'] = df_trades['Trade'].cumsum()
    df_trades['Price'] = prices
    df_trades['Impact Price'] = prices * (1 + impact * np.sign(df_trades['Trade']))
    df_trades['Cash'] = -df_trades['Impact Price'] * df_trades['Trade']
    df_trades['Cash'] -= (df_trades['Trade'] != 0) * commission
    df_trades['Cash'] = df_trades['Cash'].cumsum() + start_val
    df_trades['Value'] = df_trades['Holdings'] * df_trades['Price']
    df_trades['Portfolio'] = df_trades['Cash'] + df_trades['Value']
    return df_trades['Portfolio'] / df_trades['Portfolio'].iloc[0]
def bnchmk(symbol, date, sv=100000, commission=9.95, impact=0.005):
    sym =[symbol]
    price_df = ut.get_data(sym, date)
    shares = 1000
    cash=sv
    # Adjust first purchase with impact and commission
    first_price = price_df.iloc[0][symbol] * (1 + impact)
    cash -= shares * first_price + commission

    # Portfolio value each day
    port_val = price_df[symbol] * shares + cash
    return port_val / port_val.iloc[0]
def test_code():
    symbol='JPM'
    date_in=pd.date_range('2008-01-01', '2009-12-31')
    date_out=pd.date_range('2010-01-01', '2011-12-31')
    start_val=100000
    commission=9.95
    impact=0.005
    learner = sl.StrategyLearner(impact=0.005,commission=9.95)
    #np.random.seed(gtid())
    learner.add_evidence(symbol=symbol, sd=date_in[0], ed=date_in[-1])
    """
    In Sample Manual vs Strategy
    """
    # Calculate in sample manual portval
    manual=ms.ManualStrategy(verbose=True, impact=0.005, commission=9.95)
    price_in=ut.get_data([symbol],date_in,addSPY=False).dropna()
    trade_in = manual.testPolicy(
    symbol='JPM',
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 12, 31),
    sv=100000)
    manual_in=compute_portval(trade_in,price_in[symbol])

    # Calculate in sample strategy learner protval
    trades_in = learner.testPolicy(symbol=symbol, sd=date_in[0], ed=date_in[-1])
    prices_in = ut.get_data([symbol], pd.date_range(date_in[0], date_in[-1]), addSPY=False)[symbol]
    strategy_in = compute_portval(trades_in, prices_in)

    # Calculate benchmark
    benchmark_in=bnchmk(symbol,date_in)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(manual_in, color='red', label='Manual Strategy')
    plt.plot(benchmark_in, color='purple', label='Benchmark')
    plt.plot(strategy_in, color='blue', label='Strategy Learner')
    # Labels and legend
    plt.title('Manual Strategy vs Strategy Learner (In-Sample)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Chart3.png')
    plt.close()

    """
    Out of Sample Manual vs Strategy
    """
    # Calculate out of sample manual portval
    price_out=ut.get_data([symbol],date_out,addSPY=False).dropna()
    trade_out = manual.testPolicy(
    symbol='JPM',
    sd=dt.datetime(2010, 1, 1),
    ed=dt.datetime(2011, 12, 31),
    sv=100000)
    manual_out=compute_portval(trade_out,price_out[symbol])

    # Calculate out of sample strategy learner protval
    trades_out = learner.testPolicy(symbol=symbol, sd=date_out[0], ed=date_out[-1])
    prices_out = ut.get_data([symbol], pd.date_range(date_out[0], date_out[-1]), addSPY=False)[symbol]
    strategy_out = compute_portval(trades_out, prices_out)

    # Calculate benchmark
    benchmark_out=bnchmk(symbol,date_out)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(manual_out, color='red', label='Manual Strategy')
    plt.plot(benchmark_out, color='purple', label='Benchmark')
    plt.plot(strategy_out, color='blue', label='Strategy Learner')
    # Labels and legend
    plt.title('Manual Strategy vs Strategy Learner (Out-Of-Sample)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Chart4.png')
    plt.close()
if __name__ == "__main__":
    test_code()