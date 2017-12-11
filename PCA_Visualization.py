import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn.externals import joblib
from pymongo import MongoClient

from sklearn.decomposition import TruncatedSVD

client = MongoClient('localhost', 27017)
db = client.alzconnected
agingcare = db.agingcare
categories = ["Bathing-Hygiene", "End-of-Life-Hospice", "Elder-Law"]

y = []
x = []
for post in agingcare.find({"resource_topic": {"$in": categories}}, {"resource_topic":1, "text": 1, "question": 1, "question_body":1}):
	if post['text'] and post['question'] and post['question_body']:
		tmp = []
		tmp.append(post['text'])
		tmp.append(post['question'])
		tmp.append(post['question_body'])

		y.append(post['resource_topic'])
		x.append(''.join(tmp))

x = np.asarray(x)
y = np.asarray(y)

print(len(x))

vect = joblib.load('models/vectorizer.pkl')
x_vect = vect.transform(x)

print("running PCA")

pca = TruncatedSVD(n_components=2)
X_r = pca.fit(x_vect).transform(x_vect)

print(X_r.shape)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, categories, categories):
	plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.show()
