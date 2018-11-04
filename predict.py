
import pandas as pd
#from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from  matplotlib import pyplot  as plt
from matplotlib import style

import pandas
import time

from sklearn.externals import joblib
import pickle



model = joblib.load('model.sav')
st=time.time()

X= [[ 1.04993431e-01, -1.49009684e-02 , 7.83758717e-01, -8.97792216e-02,
  -2.43525037e-01 ,-1.72502547e-01 ,-8.64587320e-01  ,4.19424368e-02]]
target=model.predict(X)

print (target[0])

ed=time.time()

print ((ed-st)*1000)
