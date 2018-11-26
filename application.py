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



def stock():
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
		row['Avg']=row[['Open', 'Adj. Close']].mean()
		row['Moving Avg']=row[['High', 'Low']].mean()
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
	print (df)
	X = np.array(df.drop(['label'], 1))
	#X = preprocessing.scale(X)
	X_lately = X[-forecast_out:]
	X = X[:-forecast_out]

	df.dropna(inplace=True)

	y = np.array(df['label'])

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

	print ("\nX_train:\n")
	print(X_train)
	#print (X_train.shape)
	print ("\nX_test:\n")
	print(X_test)
	#print (X_test.shape)

	print("LinearRegression:")


	clf = LinearRegression(n_jobs=-1)
	clf.fit(X_train, y_train)
	confidence = clf.score(X_test, y_test)

	print (confidence)
	

	
	

	prediction=clf.predict(X_test)

	print(prediction)
	#pyplot.scatter(X_test, y_test)
	
	rms = sqrt(mean_squared_error(y_test, prediction))
	print ("rms")
	print (rms)

	plt.scatter(X_test[:,9],prediction)
	#par = np.polyfit(X_test[:,9],prediction, 1, full=True)
	#plt.plot(X_test[:,9],par)
	#plt.plot(X_test,prediction)
	plt.show()
	plt.plot(y_test)
	plt.plot(prediction)
	plt.savefig('LinearRegression.png')
	plt.show()
	
	#Image.open('LinearRegression.png').save('LinearRegression.jpg','JPEG')
	
	filename = 'model1.sav'
	pickle.dump(clf, open(filename, 'wb'))

	print("Polynomial Regression:")


	poly = PolynomialFeatures(2)

	X_new=X_test[:,3]
	m,n=np.shape(X_test)
	X_new=X_new.reshape(m,1)

	#y_new=y_test[:,3]
	m,n=np.shape(X_test)
	y_test=y_test.reshape(m,1)

	X_transform = poly.fit_transform(X_train)
	X_test_new=poly.fit_transform(X_test)
	#print ("X transform:")
	#print (X_transform)
	clf=LinearRegression()

	clf.fit(X_transform,y_train) 
	confidence = clf.score(X_test_new,y_test)

	print (confidence)
	prediction= clf.predict(X_test_new)

	plt.scatter(X_test[:,9],prediction)

	rms = sqrt(mean_squared_error(y_test, prediction))
	print ("rms")
	print (rms)
	#plt.plot(X_test,prediction)
	plt.show()
	print(prediction)
	plt.plot(y_test)
	plt.plot(prediction)
	
	plt.savefig('Poly_regression.png')
	plt.show()
	#plt.plot(X_test,prediction,color='blue', linewidth=3)

	#plt.show()'



	#print("Logistic Regression")

	lab_enc = preprocessing.LabelEncoder()
	training_scores_encoded = lab_enc.fit_transform(y_train)
	#print(training_scores_encoded)
	#print(utils.multiclass.type_of_target(y_train))
	#print(utils.multiclass.type_of_target(y_train.astype('float')))
	#print(utils.multiclass.type_of_target(training_scores_encoded))


	lab_enc = preprocessing.LabelEncoder()
	training_scores_encoded_test = lab_enc.fit_transform(y_test)
	#print(training_scores_encoded)
	#print(utils.multiclass.type_of_target(y_test))
	#print(utils.multiclass.type_of_target(y_test.astype('float')))
	#print(utils.multiclass.type_of_target(training_scores_encoded_test))


	print("Logistic Regression")
	clf = LogisticRegression()
	clf.fit(X_train,training_scores_encoded)
	# Testing
	confidence = clf.score(X_test, training_scores_encoded_test)

	print("confidence: ", confidence)

	prediction=clf.predict(X_test)

	print(prediction)
	#pyplot.scatter(X_test, y_test)
	plt.scatter(X_test[:,9],prediction)

	rms = sqrt(mean_squared_error(y_test, prediction))
	print ("rms")
	print (rms)

	plt.show()
	plt.plot(y_test)
	plt.plot(prediction)
	
	plt.savefig('LogisticRegression.png')
	plt.show()

	print("SVC")
	clf = SVC()
	clf.fit(X_train,training_scores_encoded)
	# Testing
	confidence = clf.score(X_test, training_scores_encoded_test)

	print("confidence: ", confidence)

	prediction=clf.predict(X_test)

	print(prediction)
	#pyplot.scatter(X_test, y_test)
	plt.scatter(X_test[:,9],prediction)


	rms = sqrt(mean_squared_error(y_test, prediction))
	print ("rms")
	print (rms)

	plt.savefig('Svc.png')
	plt.show()

	plt.plot(y_test)
	plt.plot(prediction)
	plt.savefig('Svc.png')
	plt.show()

	print("KNN")

	clf = KNeighborsClassifier()
	clf.fit(X_train,training_scores_encoded)
	# Testing
	confidence = clf.score(X_test, training_scores_encoded_test)

	print("confidence: ", confidence)

	prediction=clf.predict(X_test)

	print(prediction)
	#pyplot.scatter(X_test, y_test)
	plt.scatter(X_test[:,9],prediction)

	rms = sqrt(mean_squared_error(y_test, prediction))
	print ("rms")
	print (rms)

	plt.show()
	plt.plot(y_test)
	plt.plot(prediction)
	
	plt.savefig('Knn.png')
	plt.show()


	print("Ridge Regression:")

	clf = Ridge(alpha=0.5)
	clf.fit(X_train,y_train)
	confidence = clf.score(X_test, y_test)

	print (confidence)

	prediction=clf.predict(X_test)

	print(prediction)
	#pyplot.scatter(X_test, y_test)
	plt.scatter(X_test[:,9],prediction)

	rms = sqrt(mean_squared_error(y_test, prediction))
	print ("rms")
	print (rms)

	#plt.plot(X_test,prediction)
	plt.show()

	plt.plot(y_test)
	plt.plot(prediction)
	
	plt.savefig('Ridge_regression.png')
	plt.show()

	print ("Lasso Regression:")

	clf = Lasso(alpha=0.5)
	clf.fit(X_train,y_train)
	confidence = clf.score(X_test, y_test)

	print (confidence)

	prediction=clf.predict(X_test)

	print(prediction)
	#pyplot.scatter(X_test, y_test)
	plt.scatter(X_test[:,9],prediction)

	rms = sqrt(mean_squared_error(y_test, prediction))
	print ("rms")
	print (rms)

	
	#plt.plot(X_test,prediction)
	plt.show()
	plt.plot(y_test)
	plt.plot(prediction)
	
	plt.savefig('lasso_regression.png')
	plt.show()

if __name__=="__main__":
	stock()
