from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.core.files import File
from .models import Adj_close,Document
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from trading.forms import DocumentForm
from trading.predict import *
from trading.predict_upload1 import *

import pandas as pd
#from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from  matplotlib import pyplot  as plt
from matplotlib import style
import math
import pandas as pd
import time
import csv
import sys
from sklearn.externals import joblib
import pickle
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
import datetime
import importlib

from sklearn.metrics import mean_squared_error
from math import sqrt
# Create your views here.

def upload_form(request):
	return render(request, 'trading/upload.html')

def upload_csv(request):
	return render(request, 'trading/upload_csv.html')

#def showcsv(request):

	#saved=False
	#if request.method == "POST":
		#file = request.FILES['csv'] 
		#decoded_file = file.read().decode('utf-8').splitlines()
		#data = csv.DictReader(decoded_file)
		#list1 = []
		#saved = True
		#for row in data:
		#	list1.append(row)

		#print (list1)
		#context={'saved':saved}

		#return render(request, 'trading/saved_csv.html',context)
def showcsv(request):

	saved=False
	profile=Document()
	if request.method == "POST":
		MyProfileForm = DocumentForm(request.POST, request.FILES)
		profile.document = request.FILES['csv']
		#profile.landscape = request.FILES['canvas']
		profile.save()
		saved = True

		print (profile.document.name)
		namefile=profile.document.name
		filename=namefile.split('.')
		filename1=filename[0]
		real_file=filename1.split('/')
		file_realname=real_file[1]
		print (file_realname)

		sys.path.insert(0, '/Users/rashmisahu/btp_7/documents')
		#__import__(file_realname) 
		action = importlib.import_module(file_realname,package=None)

		X_train, X_test, y_train, y_test=action.main()


		max_acc=-1;
		max_acc_model="none"


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

		rms = sqrt(mean_squared_error(y_test, prediction))
		print ("rms")
		print (rms)
		plt.plot(y_test)
		plt.plot(prediction)
		plt.savefig('assets/LinearRegression.png')
		plt.clf()

		if confidence>max_acc:
			max_acc=confidence
			max_acc_classifier=clf
			max_acc_model="LinearRegression"

		#filename = 'model1.sav'
		#pickle.dump(clf, open(filename, 'wb'))

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

		print(prediction)

		rms = sqrt(mean_squared_error(y_test, prediction))
		print ("rms")
		print (rms)
		plt.plot(y_test)
		plt.plot(prediction)
		plt.savefig('assets/Poly_regression.png')
		plt.clf()

		if confidence>max_acc:
			max_acc=confidence
			max_acc_classifier=clf
			max_acc_model="Polynomial Regression"

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

		rms = sqrt(mean_squared_error(y_test, prediction))
		print ("rms")
		print (rms)
		plt.plot(y_test)
		plt.plot(prediction)
		plt.savefig('assets/LogisticRegression.png')
		plt.clf()


		print("SVC")
		clf = SVC()
		clf.fit(X_train,training_scores_encoded)
		# Testing
		confidence = clf.score(X_test, training_scores_encoded_test)

		print("confidence: ", confidence)

		prediction=clf.predict(X_test)

		print(prediction)

		rms = sqrt(mean_squared_error(y_test, prediction))
		print ("rms")
		print (rms)
		plt.plot(y_test)
		plt.plot(prediction)
		plt.savefig('assets/svc.png')
		plt.clf()


		if confidence>max_acc:
			max_acc=confidence
			max_acc_classifier=clf
			max_acc_model="SVC"

		print("KNN")

		clf = KNeighborsClassifier()
		clf.fit(X_train,training_scores_encoded)
		# Testing
		confidence = clf.score(X_test, training_scores_encoded_test)

		print("confidence: ", confidence)

		prediction=clf.predict(X_test)

		print(prediction)

		rms = sqrt(mean_squared_error(y_test, prediction))
		print ("rms")
		print (rms)
		plt.plot(y_test)
		plt.plot(prediction)
		plt.savefig('assets/knn.png')
		plt.clf()



		if confidence>max_acc:
			max_acc=confidence
			max_acc_classifier=clf
			max_acc_model="KNN"

		print("Ridge Regression:")

		clf = Ridge(alpha=1.0)
		clf.fit(X_train,y_train)
		confidence = clf.score(X_test, y_test)

		print (confidence)

		prediction=clf.predict(X_test)

		print(prediction)

		rms = sqrt(mean_squared_error(y_test, prediction))
		print ("rms")
		print (rms)
		plt.plot(y_test)
		plt.plot(prediction)
		plt.savefig('assets/Ridge_regression.png')
		plt.clf()


		if confidence>max_acc:
			max_acc=confidence
			max_acc_classifier=clf
			max_acc_model="Ridge Regression"


		print ("Lasso Regression:")

		clf = Lasso(alpha=1.0)
		clf.fit(X_train,y_train)
		confidence = clf.score(X_test, y_test)

		print (confidence)

		prediction=clf.predict(X_test)

		print(prediction)

		rms = sqrt(mean_squared_error(y_test, prediction))
		print ("rms")
		print (rms)
		plt.plot(y_test)
		plt.plot(prediction)
		plt.savefig('assets/lasso_regression.png')
		plt.clf()


		if confidence>max_acc:
			max_acc=confidence
			max_acc_classifier=clf
			max_acc_model="Lasso Regression"
			
			
			
		filename = 'model2.sav'
		pickle.dump(max_acc_classifier, open(filename, 'wb'))


			
		

	context={'saved':saved}
	return render(request, 'trading/saved_csv.html',context)



def predict(request):
	saved = False
	
	profile=Adj_close()
	if request.method == "POST":
		#MyProfileForm = ProfileForm(request.POST, request.FILES)
		
		

		
			
			
		profile.adj_open = request.POST.get("Adjusted Open Price")
		profile.moving_avg = request.POST.get('Moving Average')
		profile.adj_high = request.POST.get('Adjusted High Price')
		profile.adj_low = request.POST.get('Adjusted Low Price')
		profile.hl_pct = request.POST.get('HL_PCT')
		profile.pct_change = request.POST.get('PCT_change')
		profile.adj_vol = request.POST.get('Adjusted Volume')
		profile.avg = request.POST.get('Average')
		#profile.runtime = request.POST.get('Execution_time')
				#profile.landscape = request.FILES['canvas']
		profile.save()


		saved = True
			
			
		

		#list1=[[]]
		list2=[[]]
		list2[0].append(float(profile.adj_open));
		list2[0].append(float(profile.adj_high));
		list2[0].append(float(profile.adj_low));
		list2[0].append(float(profile.hl_pct));
		list2[0].append(float(profile.pct_change));
		list2[0].append(float(profile.adj_vol));
		list2[0].append(float(profile.avg));
		list2[0].append(float(profile.moving_avg));
		#list2[0].append(float(profile.runtime));
		#list1=np.array([list2])

		#print (list2)
		print(list2)


		label=pre(list2)
		context={
			'profile1':profile,
			
			'label':label,
			'saved':saved}
		return render(request, 'trading/saved.html',context)


def predictupload(request):
	saved = False
	
	profile=Adj_close()
	if request.method == "POST":
		#MyProfileForm = ProfileForm(request.POST, request.FILES)
		
		

		
			
			
		profile.adj_open = request.POST.get("Adjusted Open Price")
		profile.moving_avg = request.POST.get('Moving Average')
		profile.adj_high = request.POST.get('Adjusted High Price')
		profile.adj_low = request.POST.get('Adjusted Low Price')
		profile.hl_pct = request.POST.get('HL_PCT')
		profile.pct_change = request.POST.get('PCT_change')
		profile.adj_vol = request.POST.get('Adjusted Volume')
		profile.avg = request.POST.get('Average')
		#profile.runtime = request.POST.get('Execution_time')
				#profile.landscape = request.FILES['canvas']
		profile.save()


		saved = True
			
			
		

		#list1=[[]]
		list2=[[]]
		list2[0].append(float(profile.adj_open));
		list2[0].append(float(profile.adj_high));
		list2[0].append(float(profile.adj_low));
		list2[0].append(float(profile.hl_pct));
		list2[0].append(float(profile.pct_change));
		list2[0].append(float(profile.adj_vol));
		list2[0].append(float(profile.avg));
		list2[0].append(float(profile.moving_avg));
		#list2[0].append(float(profile.runtime));
		#list1=np.array([list2])

		#print (list2)
		print(list2)


		label=pre_upload(list2)
		context={
			'profile1':profile,
			
			'label':label,
			'saved':saved}
		return render(request, 'trading/saved.html',context)



		


