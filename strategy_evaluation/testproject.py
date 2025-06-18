import pandas as pd
import datetime as dt
import numpy as np
import util as ut
from indicators import *
import ManualStrategy as ms
import StrategyLearner as sl
import matplotlib.pyplot as plt
import experiment1 as e1
import experiment2 as e2
def author():
    return "zdang31"

def study_group():
    return "zdang31"
def gtid():
    return 904080678
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


"""
Chart_1 In Sample Manual vs Benchmark
"""
symbol='JPM'
date_in=pd.date_range('2008-01-01', '2009-12-31')
date_out=pd.date_range('2010-01-01', '2011-12-31')
start_val=100000
commission=9.95
impact=0.005
manual=ms.ManualStrategy(verbose=True, impact=0.005, commission=9.95)
price_in=ut.get_data([symbol],date_in,addSPY=False).dropna()
trades_in = manual.testPolicy(
symbol='JPM',
sd=dt.datetime(2008, 1, 1),
ed=dt.datetime(2009, 12, 31),
sv=100000)
manual_in=compute_portval(trades_in,price_in[symbol])
benchmark_in=bnchmk(symbol,date_in)
# Identify LONG and SHORT entry points
long_entries = trades_in[trades_in['Trade'] > 0].index
short_entries = trades_in[trades_in['Trade'] < 0].index
# Plot
plt.figure(figsize=(12, 6))
plt.plot(manual_in, color='red', label='Manual Strategy')
plt.plot(benchmark_in, color='purple', label='Benchmark')

# Add vertical lines and labels in legend
for i, date in enumerate(long_entries):
    plt.axvline(x=date, color='blue', linestyle='--', linewidth=1,
                label='Long Entry' if i == 0 else "")

for i, date in enumerate(short_entries):
    plt.axvline(x=date, color='black', linestyle='--', linewidth=1,
                label='Short Entry' if i == 0 else "")

# Labels and legend
plt.title('Manual Strategy vs Benchmark (In-Sample)')
plt.xlabel('Date')
plt.ylabel('Normalized Portfolio Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Chart1.png')
plt.close()

"""
Chart_2 Out of Sample Manual vs Benchmark
"""
price_out=ut.get_data([symbol],date_out,addSPY=False).dropna()
trades_out = manual.testPolicy(
symbol='JPM',
sd=dt.datetime(2010, 1, 1),
ed=dt.datetime(2011, 12, 31),
sv=100000)
manual_out=compute_portval(trades_out,price_out[symbol])
benchmark_out=bnchmk(symbol,date_out)
# Identify LONG and SHORT entry points
long_entries_ = trades_out[trades_out['Trade'] > 0].index
short_entries_ = trades_out[trades_out['Trade'] < 0].index
# Plot
plt.figure(figsize=(12, 6))
plt.plot(manual_out, color='red', label='Manual Strategy')
plt.plot(benchmark_out, color='purple', label='Benchmark')

# Add vertical lines and legend labels
for i, date in enumerate(long_entries_):
    plt.axvline(x=date, color='blue', linestyle='--', linewidth=1,
                label='Long Entry' if i == 0 else "")

for i, date in enumerate(short_entries_):
    plt.axvline(x=date, color='black', linestyle='--', linewidth=1,
                label='Short Entry' if i == 0 else "")

# Labels and legend
plt.title('Manual Strategy vs Benchmark (Out-of-Sample)')
plt.xlabel('Date')
plt.ylabel('Normalized Portfolio Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Chart2.png')
plt.close()

"""
Performance Summary Table
"""
# Calculate daily returns using traditional method
def daily_returns(port_vals):
    daily_rets = (port_vals.diff() / port_vals.shift(1)).dropna()
    return daily_rets

# Calculate metrics
def calculate_metrics(port_vals, start_val=100000):
    # Convert normalized portfolio values to real values
    port_vals_real = port_vals * start_val
    daily_rets = daily_returns(port_vals_real)
    # Cumulative return as dollar amount (final value - initial value)
    cum_return_dollars = (port_vals_real.iloc[-1] - start_val)/start_val
    std_daily_rets = daily_rets.std()
    mean_daily_rets = daily_rets.mean()
    return cum_return_dollars, std_daily_rets, mean_daily_rets

# Compute metrics
manual_in_cum, manual_in_std, manual_in_mean = calculate_metrics(manual_in, start_val)
bench_in_cum, bench_in_std, bench_in_mean = calculate_metrics(benchmark_in, start_val)
manual_out_cum, manual_out_std, manual_out_mean = calculate_metrics(manual_out, start_val)
bench_out_cum, bench_out_std, bench_out_mean = calculate_metrics(benchmark_out, start_val)

# Create table
metrics_table = pd.DataFrame({
    'Period': ['In-Sample', 'In-Sample', 'Out-of-Sample', 'Out-of-Sample'],
    'Strategy': ['Manual', 'Benchmark', 'Manual', 'Benchmark'],
    'Cumulative Return': [manual_in_cum, bench_in_cum, manual_out_cum, bench_out_cum],
    'Std Dev Daily Returns': [manual_in_std, bench_in_std, manual_out_std, bench_out_std],
    'Mean Daily Returns': [manual_in_mean, bench_in_mean, manual_out_mean, bench_out_mean]
})
metrics_table['Cumulative Return'] = metrics_table['Cumulative Return'].round(6)
metrics_table['Std Dev Daily Returns'] = metrics_table['Std Dev Daily Returns'].round(6)
metrics_table['Mean Daily Returns'] = metrics_table['Mean Daily Returns'].round(6)
metrics_table.to_csv('metrics_table.csv', index=False)
np.random.seed(gtid())
"""
Run Experiment 1
"""
e1.test_code()

"""
Run Experiment 2
"""
e2.test_code()

