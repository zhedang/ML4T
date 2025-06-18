import datetime as dt
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def calculate_sharpe_ratio(portvals, risk_free_rate=0.0, samples_per_year=252):
    # Calculate daily returns
    daily_rets = (portvals / portvals.shift(1)) - 1.0
    daily_rets = daily_rets.iloc[1:]  # Remove first NaN

    # Compute metrics
    mean_daily_ret = np.mean(daily_rets)
    std_daily_ret = np.std(daily_rets)

    # Annualize and calculate Sharpe
    sharpe = (mean_daily_ret - risk_free_rate) / std_daily_ret
    return sharpe * np.sqrt(samples_per_year)
def test_code():
    symbol='JPM'
    date_in=pd.date_range('2008-01-01', '2009-12-31')
    start_val=100000
    commission=9.95

    learner_1 = sl.StrategyLearner(impact=0, commission=0)
    learner_2 = sl.StrategyLearner(impact=0.008, commission=0)
    learner_3 = sl.StrategyLearner(impact=0.016, commission=0)
    #np.random.seed(gtid())
    learner_1.add_evidence(symbol=symbol, sd=date_in[0], ed=date_in[-1])
    learner_2.add_evidence(symbol=symbol, sd=date_in[0], ed=date_in[-1])
    learner_3.add_evidence(symbol=symbol, sd=date_in[0], ed=date_in[-1])

    prices_in = ut.get_data([symbol], pd.date_range(date_in[0], date_in[-1]), addSPY=False)[symbol]

    # Learner 1
    trades_in_1 = learner_1.testPolicy(symbol=symbol, sd=date_in[0], ed=date_in[-1])
    strategy_in_1 = compute_portval(trades_in_1, prices_in)
    daily_rets_1 = ((strategy_in_1 / strategy_in_1.shift(1)) - 1.0)[1:]
    avg_rt_1 = np.mean(daily_rets_1)
    std_rt_1 = np.std(daily_rets_1)
    sharpe_ratio_1 = calculate_sharpe_ratio(strategy_in_1)
    num_trades_in_1 = trades_in_1.value_counts()[1:].sum()

    # Learner 2
    trades_in_2 = learner_2.testPolicy(symbol=symbol, sd=date_in[0], ed=date_in[-1])
    strategy_in_2 = compute_portval(trades_in_2, prices_in)
    daily_rets_2 = ((strategy_in_2 / strategy_in_2.shift(1)) - 1.0)[1:]
    avg_rt_2 = np.mean(daily_rets_2)
    std_rt_2 = np.std(daily_rets_2)
    sharpe_ratio_2 = calculate_sharpe_ratio(strategy_in_2)
    num_trades_in_2 = trades_in_2.value_counts()[1:].sum()

    # Learner 3
    trades_in_3 = learner_3.testPolicy(symbol=symbol, sd=date_in[0], ed=date_in[-1])
    strategy_in_3 = compute_portval(trades_in_3, prices_in)
    daily_rets_3 = ((strategy_in_3 / strategy_in_3.shift(1)) - 1.0)[1:]
    avg_rt_3 = np.mean(daily_rets_3)
    std_rt_3 = np.std(daily_rets_3)
    sharpe_ratio_3 = calculate_sharpe_ratio(strategy_in_3)
    num_trades_in_3 = trades_in_3.value_counts()[1:].sum()

    # Prepare plot data
    impacts = ['Impact: 0', 'Impact: 0.008', 'Impact: 0.016']
    avg_ret = [avg_rt_1, avg_rt_2, avg_rt_3]
    stds = [std_rt_1, std_rt_2, std_rt_3]
    sharpe = [sharpe_ratio_1, sharpe_ratio_2, sharpe_ratio_3]
    trades = [num_trades_in_1, num_trades_in_2, num_trades_in_3]
    dates = strategy_in_1.index

    # Plot
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, height_ratios=[3, 2])

    # Top plot
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(dates, strategy_in_1, label='StrategyLearner - Impact: 0', color='blue')
    ax0.plot(dates, strategy_in_2, label='StrategyLearner - Impact: 0.008', color='red')
    ax0.plot(dates, strategy_in_3, label='StrategyLearner - Impact: 0.016', color='purple')
    ax0.set_title('Experiment 2: Market Impact Effect on StrategyLearner Performance')
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Normalized Portfolio Value')
    ax0.legend()
    ax0.grid(True)

    # Bar 1: Average Daily Return
    ax1 = fig.add_subplot(gs[1, 0])
    bars1 = ax1.bar(impacts, avg_ret, color=['blue', 'red', 'purple'])
    ax1.set_title('Average Daily Return')
    ax1.set_ylim(top=max(avg_ret) * 1.2)
    for bar, val in zip(bars1, avg_ret):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.01, f"{val:.5f}", ha='center')

    # Bar 2: Std of Daily Return
    ax_std = fig.add_subplot(gs[1, 1])
    bars_std = ax_std.bar(impacts, stds, color=['blue', 'red', 'purple'])
    ax_std.set_title('Std of Daily Return')
    ax_std.set_ylim(top=max(stds) * 1.2)
    for bar, val in zip(bars_std, stds):
        ax_std.text(bar.get_x() + bar.get_width() / 2, val * 1.01, f"{val:.5f}", ha='center')

    # Bar 3: Sharpe Ratio
    ax2 = fig.add_subplot(gs[1, 2])
    bars2 = ax2.bar(impacts, sharpe, color=['blue', 'red', 'purple'])
    ax2.set_title('Sharpe Ratio')
    ax2.set_ylim(top=max(sharpe) * 1.2)
    for bar, val in zip(bars2, sharpe):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.01, f"{val:.3f}", ha='center')

    # Bar 4: Total Trades
    ax3 = fig.add_subplot(gs[1, 3])
    bars3 = ax3.bar(impacts, trades, color=['blue', 'red', 'purple'])
    ax3.set_title('Total Trades')
    ax3.set_ylim(top=max(trades) * 1.2)
    for bar, val in zip(bars3, trades):
        ax3.text(bar.get_x() + bar.get_width() / 2, val * 1.01, f"{val:.0f}", ha='center')

    plt.tight_layout()
    plt.savefig('Chart5.png')
    plt.close()

if __name__ == "__main__":
    test_code()