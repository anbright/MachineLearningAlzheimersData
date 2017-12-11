import numpy as np
import matplotlib
matplotlib.use('TkAgg')

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

from pymongo import MongoClient

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize

client = MongoClient('localhost', 27017)
db = client.alzconnected
agingcare = db.agingcare
excluded_categories = ["Alzheimers-Dementia", "Sleep-Disorders", "Frauds-Scams", "Parkinsons-Disease", "Cancer", "Physical-Wellbeing", "Hearing-Loss", "Heart-Disease", "Diabetes", "Vision-Eye-Diseases", "Lung-Disease", "Arthritis", "Medicare-Open-Enrollment", "Caregiving-News", "Osteoporosis"]

stemmer = SnowballStemmer("english", ignore_stopwords=True)

# def tokenize(text):
# 	tokens = word_tokenize(text)
# 	stems = [stemmer.stem(item) for item in text]
# 	return stems

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

print(len(x))

## Tf-Idf vectorization
# tfidf_vect = TfidfVectorizer(lowercase=True, stop_words='english', sublinear_tf=True, tokenizer=tokenize)
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

print(accuracy_score(y_hat, y_test))
print(recall_score(y_hat, y_test, average='weighted'))
print(precision_score(y_hat, y_test, average='weighted'))

## SVM Classifier

clf_SVC = SVC(kernel=cosine_similarity, C=10)
clf_SVC.fit(x_train, y_train)

y_hat = clf_SVC.predict(x_test)
print(accuracy_score(y_hat, y_test))
print(recall_score(y_hat, y_test, average='weighted'))
print(precision_score(y_hat, y_test, average='weighted'))

joblib.dump(tfidf_vect, 'vectorizer.pkl')
joblib.dump(clf_SVC, 'SVC.pkl')
joblib.dump(clf_NB, 'NB.pkl')