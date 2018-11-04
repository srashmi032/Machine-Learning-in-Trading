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
import numpy as np
import time

from sklearn.linear_model import LinearRegression

from sklearn import preprocessing
from sklearn import utils

names = ['Date', 'Open', 'High', 'Low', 'Close','Adj Close','Volume']
dataset = pandas.read_csv('/Users/rashmisahu/Desktop/rashmi/sem_7/btp_sem7/tatasteel.csv',names=names)

df = pandas.DataFrame(dataset)
df.drop(df.index[0])
dataset=df
del dataset['Date']
print(dataset.shape)

print(dataset.head(20))

dataset['Percent_Change'] = (dataset['Close'].astype(float).pct_change())



print(dataset.head())
df = pandas.DataFrame(dataset)

df['Avg']=pandas.Series(np.random.randn(dataset['Close'].count()), index=df.index)
df['execution_time']=pandas.Series(np.random.randn(dataset['Close'].count()), index=df.index)



for index,row in df.iterrows():
	start = time.time()
	row['Avg']=row[['Open', 'Close']].mean()
	#print (row['Avg'])
	df['Avg'][index]=row['Avg']
	stop = time.time()
	duration = (stop-start)*1000.
	df['execution_time'][index]=duration

df['Percent_Change'][0]=-100
#dataset['Avg']=df['Avg']
dataset=df
print(dataset.head(20))

#y=dataset.execution_time
#x=dataset.drop('execution_time', axis=1)

x=dataset[['Open', 'High', 'Low', 'Close','Volume']]
y=dataset[['Adj Close']]
#divide dataset into training and test set

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)

print ("\nX_train:\n")
print(x_train.head())
print (x_train.shape)
print ("\nX_test:\n")
print(x_test.head())
print (x_test.shape)

print (y_test)




lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y_train)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train.astype('float')))
print(utils.multiclass.type_of_target(training_scores_encoded))


knn=LinearRegression()
knn.fit(x_train,training_scores_encoded)
#acc_train=accuracy_score(y_train,knn.predict(x_train))
#acc_test=accuracy_score(y_test,knn.predict(y_test))
acc_train=knn.score(x_test, y_test)
print ("accuracy_score")

print (acc_train)


seed = 7
scoring = 'accuracy'


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, x_train, training_scores_encoded, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

	#print model.predict(y_test)





	#testing linear regression

plt.boxplot(results)
plt.show()



