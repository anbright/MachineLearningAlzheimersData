import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc
from sklearn import model_selection

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from sklearn.externals import joblib
from scipy import interp

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

n_rows = 29096

y = []
x = []
for post in agingcare.find({"resource_topic": {"$nin": excluded_categories}}, {"resource_topic":1, "text": 1, "question": 1, "question_body":1}).limit(35000):
	if post['text'] and post['question'] and post['question_body']:
		tmp = []
		tmp.append(post['text'])
		tmp.append(post['question'])
		tmp.append(post['question_body'])

		y.append(post['resource_topic'])
		x.append(''.join(tmp))

# x = np.asarray(x)
# y = np.asarray(y)

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
clf_NB = MultinomialNB(alpha=0.2)
trained_NB = clf_NB.fit(x_train, y_train)

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

# joblib.dump(tfidf_vect, 'vectorizer.pkl')
# joblib.dump(clf_SVC, 'SVC.pkl')
# joblib.dump(clf_NB, 'NB.pkl')

## ROC Curves
lw = 2
plt.figure()

categories = np.unique(y)
n_classes = categories.shape[0]
y_test_binary = label_binarize(y_test, classes=categories)

fpr = dict()
tpr = dict()
roc_auc = dict()

# NB
decision_NB = clf_NB.predict_proba(x_test)

for i in range(n_classes):
	fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], decision_NB[:, i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


plt.plot(fpr["macro"], tpr["macro"], color='darkorange', linewidth=4, label='Naive Bayes')

## ROC for SVC

decision_SVC = clf_SVC.decision_function(x_test)

for i in range(n_classes):
	fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], decision_SVC[:, i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.plot(fpr["macro"], tpr["macro"], color='navy', linewidth=4, label='SVC')


# Finish plotting

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()
