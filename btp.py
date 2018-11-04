
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import operator
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
import re
from dateutil import parser
from backtest import Strategy, Portfolio

def getStock(symbol, start, end):
    """
    Downloads Stock from Yahoo Finance.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    df =  data.get_data_yahoo(symbol, start, end)

    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + symbol
    df['Return_%s' %symbol] = df['AdjClose_%s' %symbol].pct_change()
    
    return df


def getStockFromQuandl(symbol, name, start, end):
    """
    Downloads Stock from Quandl.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    import Quandl
    df =  Quandl.get(symbol, trim_start = start, trim_end = end, authtoken="your token")

    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + name
    df['Return_%s' %name] = df['AdjClose_%s' %name].pct_change()
    
    return df

def getStockDataFromWeb(fout, start_string, end_string):
    """
    Collects predictors data from Yahoo Finance and Quandl.
    Returns a list of dataframes.
    """
    start = parser.parse(start_string)
    end = parser.parse(end_string)
    
    nasdaq = getStock('^IXIC', start, end)
    frankfurt = getStock('^GDAXI', start, end)
    london = getStock('^FTSE', start, end)
    paris = getStock('^FCHI', start, end)
    hkong = getStock('^HSI', start, end)
    nikkei = getStock('^N225', start, end)
    australia = getStock('^AXJO', start, end)
    
    djia = getStockFromQuandl("YAHOO/INDEX_DJI", 'Djia', start_string, end_string) 
    
    out =  pd.io.data.get_data_yahoo(fout, start, end)
    out.columns.values[-1] = 'AdjClose'
    out.columns = out.columns + '_Out'
    out['Return_Out'] = out['AdjClose_Out'].pct_change()
    
    return [out, nasdaq, djia, frankfurt, london, paris, hkong, nikkei, australia]


def addFeatures(dataframe, adjclose, returns, n):
    """
    operates on two columns of dataframe:
    - n >= 2
    - given Return_* computes the return of day i respect to day i-n. 
    - given AdjClose_* computes its moving average on n days

    """
    
    return_n = adjclose[9:] + "Time" + str(n)
    dataframe[return_n] = dataframe[adjclose].pct_change(n)
    
    roll_n = returns[7:] + "RolMean" + str(n)
    dataframe[roll_n] = pd.rolling_mean(dataframe[returns], n)

def applyRollMeanDelayedReturns(datasets, delta):
    """
    applies rolling mean and delayed returns to each dataframe in the list
    """
    for dataset in datasets:
        columns = dataset.columns    
        adjclose = columns[-2]
        returns = columns[-1]
        for n in delta:
            addFeatures(dataset, adjclose, returns, n)
    
    return datasets    

def mergeDataframes(datasets, index, cut):
    """
    merges datasets in the list 
    """
    subset = []
    subset = [dataset.iloc[:, index:] for dataset in datasets[1:]]
    
    first = subset[0].join(subset[1:], how = 'outer')
    finance = datasets[0].iloc[:, index:].join(first, how = 'left') 
    finance = finance[finance.index > cut]
    return finance


def applyTimeLag(dataset, lags, delta):
    """
    apply time lag to return columns selected according  to delta.
    Days to lag are contained in the lads list passed as argument.
    Returns a NaN free dataset obtained cutting the lagged dataset
    at head and tail
    """
    
    dataset.Return_Out = dataset.Return_Out.shift(-1)
    maxLag = max(lags)

    columns = dataset.columns[::(2*max(delta)-1)]
    for column in columns:
        for lag in lags:
            newcolumn = column + str(lag)
            dataset[newcolumn] = dataset[column].shift(lag)

    return dataset.iloc[maxLag:-1,:]

data=getStock('^FCHI',datetime.datetime(2017, 1, 1),datetime.datetime(2018, 9, 1))
