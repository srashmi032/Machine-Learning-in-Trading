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

from sklearn import preprocessing
from sklearn import utils
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class1']
dataset = pandas.read_csv('/Users/rashmisahu/Desktop/IRIS.csv',names=names)


#names = ['Date', 'Open', 'High', 'Low', 'Close','Adj_Close','Volume']
#dataset = pandas.read_csv('/Users/rashmisahu/Desktop/btp_sem7/tata.csv',names=names)


del dataset['Date']

print(dataset.shape)

print(dataset.head(20))

print(dataset.describe())

print(dataset.groupby('class1').size())

#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

#dataset.hist()
#plt.show()

#scatter_matrix(dataset)
#plt.show()


y=dataset.class1
x=dataset.drop('class1', axis=1)


X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)

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
	cv_results = model_selection.cross_val_score(model, X_train,y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	#print model.predict([620.231995,620.898010,611.18103,615.992004,605.205933,])



plt.boxplot(results)
plt.show()



