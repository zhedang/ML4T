import pandas as pd
import numpy as np

# Returns one column dataframe, 20 days Simple Moving Average
def SMA_20(df_,window=20):
    df=df_.copy()
    df['SMA_20'] = df.rolling(window=window).mean()
    return df['SMA_20']

def RSI_14(df_,window=14):
    df=df_.copy()
    df['Change']=df.diff()
    df["Gain"] = np.where(df["Change"] > 0, df["Change"], 0)
    df["Loss"] = np.where(df["Change"] < 0, -df["Change"], 0)
    df["Avg Gain"] = df["Gain"].rolling(window=window, min_periods=1).mean()
    df["Avg Loss"] = df["Loss"].rolling(window=window, min_periods=1).mean()
    # Compute RS 
    df["RS"] = df["Avg Gain"] / df["Avg Loss"]
    # Compute RSI
    df["RSI"] = 100 - (100 / (1 + df["RS"]))
    return df["RSI"]

def Momentum_14(df_,window=14):
    df=df_.copy()
    df["Momentum"] = (df / df.shift(window)) -1
    return df['Momentum']

def CCI_20(close,high,low,window=20,constant=0.015):
    df = pd.concat([close, high, low], axis=1)
    df.columns = ["Close", "High", "Low"]
    df['TP']=(df['Close']+df['High']+df['Low'])/3
    df["SMA_TP"] = df["TP"].rolling(window=window).mean()
    df["Deviation"] = abs(df["TP"] - df["SMA_TP"])
    df["Mean_Deviation"] = df["Deviation"].rolling(window=window).mean()
    df["CCI"] = (df["TP"] - df["SMA_TP"]) / (constant * df["Mean_Deviation"])
    return df["CCI"]

def MACD(df_):
    # Calculate MACD
    # 12-day EMA
    df=df_.copy()
    df['EMA_12'] = df_.ewm(span=12, adjust=False).mean()
    # 26-day EMA
    df['EMA_26'] = df_.ewm(span=26, adjust=False).mean()
    # MACD Line
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    # 9-day EMA of MACD (Signal Line)
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Histogram
    df['Histogram'] = df['MACD'] - df['Signal']
    return df['Histogram']
def author():
    return "zdang31"
def study_group():
    return "zdang31"