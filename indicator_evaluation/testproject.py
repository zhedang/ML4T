import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data
from indicators import SMA_20, RSI_14, ROC_14, CCI_20, MACD
import TheoreticallyOptimalStrategy as tos
def author():
    return "zdang31"
def study_group():
    return "zdang31"


if __name__ == "__main__":
    #Costomize parameters
    sv=100000
    #Take more months for rolling averages
    start_date = dt.datetime(2007, 11, 1)  		  	   		 	 	 			  		 			     			  	 
    end_date = dt.datetime(2009, 12, 31)
    symbol = "JPM"
    dates = pd.date_range(start=start_date, end=end_date)
    JPM=get_data([symbol],dates,addSPY=False).dropna()
    first_trading_day_2008 = JPM[JPM.index >= "2008-01-01"].index[0]

    """
    TOS
    """
    TOS_JPM=JPM.copy()
    TOS_JPM=TOS_JPM.loc[first_trading_day_2008:]
    benchmark=(TOS_JPM['JPM']*1000+(sv-TOS_JPM['JPM'].iloc[0]*1000))/sv
    optimal_trades=tos.testPolicy(symbol=symbol,sd=dt.datetime(2008, 1, 1),ed=end_date,sv = 100000)
    optimal_holdings= optimal_trades[symbol].cumsum()
    cash = sv - (optimal_trades[symbol] * TOS_JPM['JPM']).cumsum()
    optimal_value = cash + (optimal_holdings * TOS_JPM['JPM'])
    optimal_normalized = optimal_value / optimal_value.iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(benchmark.index, benchmark, color='purple', label='Benchmark (JPM)')
    plt.plot(optimal_normalized.index, optimal_normalized, color='red', label='Optimal Portfolio')
    plt.title('Benchmark vs Optimal Portfolio (Normalized to 1.0)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("Benchmark_VS_Optimal.png")
    # Calculate daily returns
    benchmark_daily_returns = (benchmark/benchmark.shift(1))-1
    optimal_daily_returns = (optimal_normalized/optimal_normalized.shift(1))-1

    # Cumulative returns
    benchmark_cum_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1)
    optimal_cum_return = (optimal_normalized.iloc[-1] / optimal_normalized.iloc[0] - 1)

    # Standard deviation of daily returns
    benchmark_stdev = benchmark_daily_returns.std()
    optimal_stdev = optimal_daily_returns.std()

    # Mean of daily returns
    benchmark_mean = benchmark_daily_returns.mean()
    optimal_mean = optimal_daily_returns.mean()

    # Create table as a DataFrame
    table_data = {
        'Metric': ['Cumulative Return', 'Stdev of Daily Returns', 'Mean of Daily Returns'],
        'Benchmark': [benchmark_cum_return, benchmark_stdev, benchmark_mean],
        'Optimal Portfolio': [optimal_cum_return, optimal_stdev, optimal_mean]
    }
    table_df = pd.DataFrame(table_data)
    table_df['Benchmark'] = table_df['Benchmark'].map(lambda x: f"{x:.6f}")
    table_df['Optimal Portfolio'] = table_df['Optimal Portfolio'].map(lambda x: f"{x:.6f}")
    # Save table
    table_df.to_csv("p6_results.txt", sep=" ", index=False)


    """
    Indicator: SMA_20
    """
    df_SMA=JPM.copy()
    df_SMA['SMA_20']=SMA_20(JPM)
    df_SMA['Price_SMA_Ratio'] = df_SMA['JPM'] / df_SMA['SMA_20']
    df_SMA = df_SMA.loc[first_trading_day_2008:]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    # Plot 1: Price and SMA
    ax1.plot(df_SMA.index, df_SMA['JPM'], label='Price', color='blue')
    ax1.plot(df_SMA.index, df_SMA['SMA_20'], label='SMA', color='red')
    ax1.set_title('Price & SMA: JPM (2008-01-01 to 2009-12-31)')
    ax1.set_ylabel('Price & SMA')
    ax1.grid(True)
    # Plot 2: Price / SMA Ratio
    ax2.plot(df_SMA.index, df_SMA['Price_SMA_Ratio'], label='Price / SMA', color='blue')
    ax2.axhline(y=1.0, color='gray', linestyle='--', label='Buy/Sell Threshold')

    # Fill areas for Price Above/Below SMA
    ax2.fill_between(df_SMA.index, df_SMA['Price_SMA_Ratio'], 1.0, where=df_SMA['Price_SMA_Ratio'] > 1.0, 
                    color='green', alpha=0.3, label='Price Above SMA')
    ax2.fill_between(df_SMA.index, df_SMA['Price_SMA_Ratio'], 1.0, where=df_SMA['Price_SMA_Ratio'] < 1.0, 
                    color='red', alpha=0.3, label='Price Below SMA')

    ax2.set_title('Price / SMA: JPM (2008-01-01 to 2009-12-31)')
    ax2.set_ylabel('Price / SMA')
    ax2.grid(True)
    # Move legends to the right side of the plot
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    # Adjust layout to prevent overlap and accommodate external legends
    plt.tight_layout()
    #Save Image
    plt.savefig("SMA_20.png")

    """
    Indicator: RSI_14
    """
    df_RSI=JPM.copy()
    df_RSI['RSI']=RSI_14(JPM)
    df_RSI = df_RSI.loc[first_trading_day_2008:]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    # Plot 1: JPM Price
    ax1.plot(df_RSI.index, df_RSI['JPM'], label='Price', color='blue')
    ax1.set_title('Price: JPM (2008-01-01 to 2009-12-31)')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    # Plot 2: RSI with overbought/oversold levels
    ax2.plot(df_RSI.index, df_RSI['RSI'], label='RSI', color='blue')
    ax2.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
    ax2.set_title('RSI: JPM (2008-01-01 to 2009-12-31)')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)  # RSI range
    ax2.legend()
    ax2.grid(True)
    # Move legends to the right side of the plot
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    #Save Image
    plt.savefig("RSI_14.png")

    """
    Indicator: ROC_14
    """
    df_mmtm=JPM.copy()
    df_mmtm['Momentum']=ROC_14(JPM)
    df_mmtm = df_mmtm.loc[first_trading_day_2008:]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    # Plot 1: JPM Price
    ax1.plot(df_mmtm.index, df_mmtm['JPM'], label='Price', color='blue')  # Purple to match your RSI style
    ax1.set_title('Price: JPM (2008-01-01 to 2009-12-31)')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    # Plot 2: Momentum
    ax2.plot(df_mmtm.index, df_mmtm['Momentum'], label='Rate of Change', color='blue')
    ax2.axhline(y=0, color='gray', linestyle='--', label='Zero Line')
    ax2.set_title('Rate of Change(%): JPM (2008-01-01 to 2009-12-31)')
    ax2.set_ylabel('Rate of Change(%)')
    ax2.legend()
    ax2.grid(True)
    ax2.fill_between(df_mmtm.index, df_mmtm['Momentum'], 0, where=df_mmtm['Momentum'] > 0, 
                    color='green', alpha=0.3, label='ROC>0')
    ax2.fill_between(df_mmtm.index, df_mmtm['Momentum'], 0, where=df_mmtm['Momentum'] < 0, 
                    color='red', alpha=0.3, label='ROC<0')
    # Move legends to the right side of the plot
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    #Save Image
    plt.savefig("ROC_14.png")

    """
    Indicator: CCI_20
    """
    JPM_close=get_data([symbol],dates,addSPY=False,colname="Close").dropna()
    JPM_high=get_data([symbol], dates, addSPY=False, colname="High").dropna()
    JPM_low=get_data([symbol], dates, addSPY=False, colname="Low").dropna()
    df_CCI = JPM.copy()
    df_CCI['Close']=JPM_close
    df_CCI['High']=JPM_high
    df_CCI['Low']=JPM_low
    df_CCI['CCI']=CCI_20(JPM_close,JPM_high,JPM_low)
    df_CCI = df_CCI.loc[first_trading_day_2008:]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    # Plot 1: JPM Price
    ax1.plot(df_CCI.index, df_CCI['Close'], label='Price', color='blue')  
    ax1.set_title('Price: JPM (2008-01-01 to 2009-12-31)')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    # Plot 2: CCI
    ax2.plot(df_CCI.index, df_CCI['CCI'], label='CCI', color='blue')
    ax2.axhline(y=100, color='red', linestyle='--', label='Overbought (100)')
    ax2.axhline(y=-100, color='green', linestyle='--', label='Oversold (-100)')
    ax2.set_title('CCI: JPM (2008-01-01 to 2009-12-31)')
    ax2.set_ylabel('CCI')
    ax2.legend()
    ax2.grid(True)
    # Move legends to the right side of the plot
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    #Save Image
    plt.savefig("CCI_20.png")

    """
    Indicator: MACD
    """
    df_MACD=JPM.copy()
    # 12-day EMA
    df_MACD['EMA_12'] = df_MACD[symbol].ewm(span=12, adjust=False).mean()
    # 26-day EMA
    df_MACD['EMA_26'] = df_MACD[symbol].ewm(span=26, adjust=False).mean()
    # MACD Line
    df_MACD['MACD'] = df_MACD['EMA_12'] - df_MACD['EMA_26']
    # 9-day EMA of MACD (Signal Line)
    df_MACD['Signal'] = df_MACD['MACD'].ewm(span=9, adjust=False).mean()
    # Histogram
    df_MACD['Histogram'] = MACD(JPM)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    # Plot 1: JPM Price with EMA12 and EMA26
    ax1.plot(df_MACD.index, df_MACD[symbol], label='Price', color='blue')
    ax1.set_title('Price: JPM (2008-01-01 to 2009-12-31)')
    ax1.set_ylabel('Price')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)  # Move legend to the right
    ax1.grid(True)
    # Plot 2: MACD Components
    ax2.plot(df_MACD.index, df_MACD['MACD'], label='MACD', color='blue')
    ax2.plot(df_MACD.index, df_MACD['Signal'], label='Signal', color='orange')
    ax2.bar(df_MACD.index, df_MACD['Histogram'], label='Histogram', color='gray', alpha=0.5, width=1.0)
    ax2.axhline(y=0, color='black', linestyle='--', label='Zero Line')
    ax2.set_title('MACD: JPM (2008-01-01 to 2009-12-31)')
    ax2.set_ylabel('MACD')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)  # Move legend to the right
    ax2.grid(True)
    # Move legends to the right side of the plot
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    # Adjust layout to prevent overlap and accommodate external legends
    plt.tight_layout()
    #Save Image
    plt.savefig("MACD.png")

