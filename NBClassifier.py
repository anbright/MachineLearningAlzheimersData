import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, roc_curve, auc
from sklearn import model_selection

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize

from sklearn.externals import joblib

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.alzconnected
agingcare = db.agingcare


excluded_categories = ["Alzheimers-Dementia", "Sleep-Disorders", "Frauds-Scams", "Parkinsons-Disease", "Cancer", "Physical-Wellbeing", "Hearing-Loss", "Heart-Disease", "Diabetes", "Vision-Eye-Diseases", "Lung-Disease", "Arthritis", "Medicare-Open-Enrollment", "Caregiving-News", "Osteoporosis"]

y = []
x = []
for post in agingcare.find({"resource_topic": {"$nin": excluded_categories}}, {"resource_topic":1, "text": 1, "question": 1, "question_body":1}).limit(1000):
	if post['text'] and post['question'] and post['question_body']:
		tmp = []
		tmp.append(post['text'])
		tmp.append(post['question'])
		tmp.append(post['question_body'])

		y.append(post['resource_topic'])
		x.append(''.join(tmp))

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

print("n=", len(x))

## Tf-Idf vectorization
tfidf_vect = TfidfVectorizer(lowercase=True, stop_words='english', sublinear_tf=True)
x_train = tfidf_vect.fit_transform(x_train)
x_test = tfidf_vect.transform(x_test)

## Dimensionality reduction
# svd = TruncatedSVD(n_components=500, random_state=42)
# x_train = svd.fit_transform(x_train)
# x_test = svd.transform(x_test)

## Naive Bayes classifier
clf_NB = BernoulliNB()
clf_NB.fit(x_train, y_train)

y_hat = clf_NB.predict(x_test)

print("NB accuracy: ", accuracy_score(y_test, y_hat))
print("NB recall: ", recall_score(y_test, y_hat, average='weighted'))
print("NB precision: ", precision_score(y_test, y_hat, average='weighted'))

print "y_test[0]=", y_test[0]
print "y_hat[0]=", y_hat[0]
print "NB confusion:"
print confusion_matrix(y_test, y_hat)

joblib.dump(tfidf_vect, 'vectorizer_2.pkl')
joblib.dump(clf_NB, 'NB.pkl')
