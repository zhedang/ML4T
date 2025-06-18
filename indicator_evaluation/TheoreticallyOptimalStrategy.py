import pandas as pd
import numpy as np
import datetime as dt  
from util import get_data

def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
    dates = pd.date_range(start=sd, end=ed)
    df=get_data([symbol],dates,addSPY=False)
    df=df.dropna()
    df['next_day']=df[symbol].shift(-1)
    df['position']= np.where(df[symbol]<df['next_day'], 1000, np.where(df[symbol]>df['next_day'],-1000,0))
    df.loc[df.index[-1], 'position'] = df.loc[df.index[-2], 'position']
    df['position_']=df['position'].shift(1).fillna(0)
    df['orders']= df['position']-df['position_']
    df_trades = df['orders'].to_frame(name=symbol)    
    return df_trades

def author():
    return "zdang31"
def study_group():
    return "zdang31"
