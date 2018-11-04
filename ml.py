import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm
import numpy as np
from sklearn import utils
from  matplotlib import pyplot 
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class1']
#dataset = pandas.read_csv('/Users/rashmisahu/Desktop/IRIS.csv',names=names)


names = ['Date', 'Open', 'High', 'Low', 'Close','Adj_Close','Volume']
dataset = pandas.read_csv('/Users/rashmisahu/Desktop/rashmi/sem_7/btp_sem7/tatasteel.csv',names=names)
dataset.isnull().any()

#dataset = dataset.fillna(lambda x: x.median())
dataset = dataset.fillna(method='ffill')

del dataset['Date']

print(dataset.shape)

print(dataset.head(20))

print(dataset.describe())

#print(dataset.groupby('Adj_Close').size())

#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

#dataset.hist()
#plt.show()

#scatter_matrix(dataset)
#plt.show()


y=dataset.Adj_Close
x=dataset.drop('Adj_Close', axis=1)


X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)

X_train[np.isnan(X_train)] = np.median(X_train[~np.isnan(X_train)])
y_train[np.isnan(y_train)] = np.median(y_train[~np.isnan(y_train)])
X_test[np.isnan(X_test)] = np.median(X_test[~np.isnan(X_test)])
y_test[np.isnan(y_test)] = np.median(y_test[~np.isnan(y_test)])

print ("\nX_train:\n")
print(X_train.head())
print (X_train.shape)
print ("\nX_test:\n")
print(X_test.head())
print (X_test.shape)

print (y_test)


lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y_train)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train.astype('float')))
print(utils.multiclass.type_of_target(training_scores_encoded))


lab_enc = preprocessing.LabelEncoder()
training_scores_encoded_test = lab_enc.fit_transform(y_test)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(y_test))
print(utils.multiclass.type_of_target(y_test.astype('float')))
print(utils.multiclass.type_of_target(training_scores_encoded_test))

print("Linear Regression")
clf = LinearRegression()
clf.fit(X_train,training_scores_encoded)
# Testing
confidence = clf.score(X_test, y_test)

print("confidence: ", confidence)

prediction=clf.predict(X_test)

print(prediction)
#pyplot.scatter(X_test, y_test)
pyplot.plot(X_test,prediction)
pyplot.show()

print("Logistic Regression")
clf = LogisticRegression()
clf.fit(X_train,training_scores_encoded)
# Testing
confidence = clf.score(X_test, training_scores_encoded_test)

print("confidence: ", confidence)

prediction=clf.predict(X_test)

print(prediction)
#pyplot.scatter(X_test, y_test)
pyplot.plot(X_test,prediction)
pyplot.show()


print("SVC")
clf = SVC()
clf.fit(X_train,training_scores_encoded)
# Testing
confidence = clf.score(X_test, training_scores_encoded_test)

print("confidence: ", confidence)

prediction=clf.predict(X_test)

print(prediction)
#pyplot.scatter(X_test, y_test)
pyplot.plot(X_test,prediction)
pyplot.show()




