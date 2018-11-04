
import pandas as pd
#from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from  matplotlib import pyplot  as plt
from matplotlib import style

import pandas
import time

from sklearn.externals import joblib
import pickle



model = joblib.load('stock_model.sav')
st=time.time()

X= [[19925.179688, 21349.630859, 27323.990234, 32514.939453,  6140.419922]]
target=model.predict(X)

print (target[0])

ed=time.time()

print ((ed-st)*1000)
