
import pandas as pd
#from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from  matplotlib import pyplot  as plt
from matplotlib import style

import pandas
import time

from sklearn.externals import joblib
import pickle


def pre_upload(X):
	model = joblib.load('/Users/rashmisahu/btp_7/model1.sav')
	st=time.time()

	#X= [[ 0.59443145, 0.19306051,  1.34318381,  0.24895324,  0.59443145, -0.11324157]]
	target=model.predict(X)

	print (target[0])

	ed=time.time()

	print ((ed-st)*1000)

	return target[0]
