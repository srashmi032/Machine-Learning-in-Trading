
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
#import Image
import pickle

def main():
	st=time.time()
	style.use('ggplot')

	#df = quandl.get("WIKI/GOOGL")
	names =['Date','Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']
	dataset = pandas.read_csv('/Users/rashmisahu/Desktop/rashmi/sem_7/btp_sem7/tatasteel.csv',names=names)
	df=pandas.DataFrame(dataset)
	df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
	df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
	df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

	df = df[['Adj. High',  'Adj. Low','Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
	forecast_col = 'Adj. Close'
	df.fillna(value=-99999, inplace=True)
	forecast_out = int(math.ceil(0.01 * len(df)))
	df['label'] = df[forecast_col].shift(-forecast_out)


	df['Avg']=pandas.Series(np.random.randn(dataset['Adj. Close'].count()), index=df.index)
	df['execution_time']=pandas.Series(np.random.randn(dataset['Adj. Close'].count()), index=df.index)
	df['Moving Avg']=pandas.Series(np.random.randn(dataset['Adj. Close'].count()), index=df.index)

	for index,row in df.iterrows():
		start = time.time()
		row['Avg']=row[['Adj. Open', 'Adj. Close']].mean()
		row['Moving Avg']=row[['Adj. High', 'Adj. Low']].mean()
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
	X = np.array(df.drop(['label'], 1))
	X = preprocessing.scale(X)
	X_lately = X[-forecast_out:]
	X = X[:-forecast_out]

	df.dropna(inplace=True)

	y = np.array(df['label'])

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

	return X_train, X_test, y_train, y_test
	