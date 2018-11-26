import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from  matplotlib import pyplot  as plt
from matplotlib import style
import datetime
import pandas
import time
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn import utils
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
#import Image
import pickle

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
from math import sqrt



def main():
	st=time.time()
	style.use('ggplot')

	#df = quandl.get("WIKI/GOOGL")
	names =['Date','Open',  'High',  'Low', 'Close', 'Adj. Close', 'Volume']
	dataset = pandas.read_csv('/Users/rashmisahu/Desktop/rashmi/sem_7/btp_sem7/tatasteel.csv',names=names)
	df=pandas.DataFrame(dataset)
	df = df[['Open',  'High',  'Low', 'Close', 'Adj. Close', 'Volume']]
	print (df)
	#df.drop_duplicates(inplace=True)
	df['HL_PCT'] = (df['High'] - df['Low']) / df['Adj. Close'] * 100.0
	df['PCT_change'] = (df['Adj. Close'] - df['Open']) / df['Open'] * 100.0

	df = df[[ 'Open',  'High',  'Low','Close','Adj. Close', 'HL_PCT', 'PCT_change', 'Volume']]
	forecast_col = 'Adj. Close'
	df.fillna(value=-99999, inplace=True)
	forecast_out = int(math.ceil(0.01 * len(df)))
	df['label'] = df[forecast_col].shift(-forecast_out)
	df['monthly_return']=(df['Adj. Close'].shift(-1)-df['Adj. Close'])/df['Adj. Close']


	df['Avg']=pandas.Series(np.random.randn(dataset['Adj. Close'].count()), index=df.index)
	df['execution_time']=pandas.Series(np.random.randn(dataset['Adj. Close'].count()), index=df.index)
	df['Moving Avg']=pandas.Series(np.random.randn(dataset['Adj. Close'].count()), index=df.index)
	df['CH_avg']=pandas.Series(np.random.randn(dataset['Adj. Close'].count()), index=df.index)
	for index,row in df.iterrows():
		start = time.time()
		row['Avg']=row[['High', 'Low']].mean()
		row['Moving Avg']=row[['Open', 'Adj. Close']].mean()
		row['CH_avg']=row[['Close', 'High']].mean()
		#print (row['Avg'])
		df['Avg'][index]=row['Avg']
		df['Moving Avg'][index]=row['Moving Avg']
		stop = time.time()
		duration = (stop-start)*1000.
		df['execution_time'][index]=duration

	print (df)

	ed=time.time()

	print ((ed-st)*1000)


	#df.drop(['Adj. Close'])
	#print (df['Adj. Close'])
	df=df.drop(['Adj. Close'],1)
	df=df.drop(['monthly_return'],1)
	df=df.drop(['CH_avg'],1)
	df=df.drop(['execution_time'],1)
	df=df.drop(['Close'],1)
	print (df)
	X = np.array(df.drop(['label'], 1))
	#X = preprocessing.scale(X)
	X_lately = X[-forecast_out:]
	X = X[:-forecast_out]

	df.dropna(inplace=True)

	y = np.array(df['label'])

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

	return X_train, X_test, y_train, y_test