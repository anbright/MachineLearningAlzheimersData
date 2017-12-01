import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from sklearn.externals import joblib

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.alzconnected
threads = db.threads

x = []
y = []

for thread in threads.find().limit(10):
	text = ''.join([message['text'] for message in thread['messages']])
	x.append(text)

# pp.pprint(x)

vect = joblib.load('vectorizer.pkl')
NB = joblib.load('NB.pkl')
SVC = joblib.load('SVC.pkl')

x_vect = vect.transform(x)

y_hat_NB = NB.predict(x_vect)
y_hat_SVC = SVC.predict(x_vect)

pp.pprint(y_hat_NB)
pp.pprint(y_hat_SVC)