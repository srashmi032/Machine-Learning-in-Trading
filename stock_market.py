import pandas as pd
#import quandl, math
#from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from  matplotlib import pyplot  as plt
from matplotlib import style
import numpy as np
import datetime
import pandas as pd
import time
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
import pickle
from sklearn.neighbors import KNeighborsClassifier

n225=pd.read_csv("n225.csv")
dji=pd.read_csv("dji.csv")
hsi=pd.read_csv("hsi.csv")
bsesn=pd.read_csv("bsesn.csv")
ixic=pd.read_csv("ixic.csv")
def read_data():
	

	df=pd.DataFrame({"n225":n225['Adj Close'],
		"dji":dji['Adj Close'],
		"hsi":hsi['Adj Close'],
		"bsesn":bsesn['Adj Close'],
		"ixic":ixic['Adj Close']})

	#print (df.head(5))
	return df

def calc_return(df):
	df_ret=df.apply(lambda x: x / x[1])
	df_ret.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
	plt.show()
	return df_ret

def calc_change(df):
	#df_ret=df.apply(lambda x: x / x[1])
	df_ret=df.apply(lambda x: np.log(x) - np.log(x.shift(1)))
	print (df_ret.head())
	df_ret.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
	plt.show()


def train(df):
	X = np.array(df.drop(['Max return Stock'], 1))
	y=np.array(df['Max return Stock'])

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

	print ("\nX_train:\n")
	print(X_train)
	#print (X_train.shape)
	print ("\nX_test:\n")
	print(X_test)
	#print (X_test.shape)

	print("KNN:")


	clf = KNeighborsClassifier()
	clf.fit(X_train,y_train)
	# Testing
	confidence = clf.score(X_test, y_test)

	print("confidence: ", confidence)

	prediction=clf.predict(X_test)

	print(prediction)

	filename = 'stock_model.sav'
	pickle.dump(clf, open(filename, 'wb'))

if __name__=="__main__":
	st=time.time()
	df=read_data()
	print(df.head(5))

	#df_copy=df[['n225','dji','hsi','bsesn','ixic']]


	#df['Max return Stock']=pd.Series(np.random.randn(df['n225'].count()), index=df.index)
	
	ret=calc_return(df)

	print (ret.head())
	calc_change(df)

	df['Max return Stock']="none"
	for index,row in ret.iterrows():
		#print (row)
		val=(ret.loc[index]).argmax()
		#print (val)
		row['Max return Stock']=val

		df['Max return Stock'][index]=row['Max return Stock']

	print (df.head(10))
	df.fillna(value=-99999, inplace=True)
	ed=time.time()

	print ((ed-st)*1000)
	train(df)
