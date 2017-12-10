import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import csv

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc
from sklearn import model_selection

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize

from sklearn.externals import joblib

# ParamSearch

from sklearn.model_selection import GridSearchCV

# Loading Data

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.alzconnected
agingcare = db.agingcare


excluded_categories = ["Alzheimers-Dementia", "Sleep-Disorders", "Frauds-Scams", "Parkinsons-Disease", "Cancer", "Physical-Wellbeing", "Hearing-Loss", "Heart-Disease", "Diabetes", "Vision-Eye-Diseases", "Lung-Disease", "Arthritis", "Medicare-Open-Enrollment", "Caregiving-News", "Osteoporosis"]

y = []
x = []
for post in agingcare.find({"resource_topic": {"$nin": excluded_categories}}, {"resource_topic":1, "text": 1, "question": 1, "question_body":1}):
	if post['text'] and post['question'] and post['question_body']:
		tmp = []
		tmp.append(post['text'])
		tmp.append(post['question'])
		tmp.append(post['question_body'])

		y.append(post['resource_topic'])
		x.append(''.join(tmp))

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

## Tf-Idf vectorization
tfidf_vect = TfidfVectorizer(lowercase=True, stop_words='english', sublinear_tf=True)
x_train = tfidf_vect.fit_transform(x_train)
x_test = tfidf_vect.transform(x_test)

## Naive Bayes Param Selection

params = { 'alpha': [1e-10, 0.2, 0.4, 0.6, 0.8, 1.0] }

## Naive Bayes classifier
clf_NB = BernoulliNB()

clf_NB = GridSearchCV(clf_NB, params)
clf_NB.fit(x_train, y_train)

with open('NB_CV_results.csv', 'wb+') as f:
	writer = csv.writer(f)
	writer.writerow(clf_NB.cv_results_['params'])
	writer.writerow(clf_NB.cv_results_['mean_test_score'])
	writer.writerow([clf_NB.best_params_])
	writer.writerow([clf_NB.best_score_])

print('testing SVC')


## SVM classifier

# params = { 'kernel': [cosine_similarity, 'rbf'], 'C': [1] }
# params = { 'kernel': [cosine_similarity, 'rbf'], 'C': [0.1, 1, 10, 100, 1000], 'gamma': [1e-3, 1e-4] }
params = { 'kernel': [cosine_similarity], 'C': [0.1, 1, 10, 100, 1000], 'gamma': [1e-3, 1e-4] }

clf_SVC = SVC()
clf_SVC = GridSearchCV(clf_SVC, params)
clf_SVC.fit(x_train, y_train)

with open('SVC_CV_results.csv', 'wb+') as f:
	writer = csv.writer(f)
	writer.writerow(clf_SVC.cv_results_['params'])
	writer.writerow(clf_SVC.cv_results_['mean_test_score'])
	writer.writerow([clf_SVC.best_params_])
	writer.writerow([clf_SVC.best_score_])

