import numpy as np
from pymongo import MongoClient
import os.path
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text
from sklearn.model_selection import GridSearchCV
import pickle

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print '\n'

NUM_EXAMPLES = 29000
NUM_TOPICS = 10
dataArr = None
print 'LDA parameter hypertuning:'
print 'Getting data...'
if os.path.isfile('data.pkl'):
  with open('data.pkl', 'rb') as input:
    dataArr = pickle.load(input)
else:
  client = MongoClient('mongodb://user:cis519@ds157325.mlab.com:57325/cis519')
  db = client['cis519']

  data = db.alzconnected.find({}).limit(NUM_EXAMPLES)
  dataArr = NUM_EXAMPLES*['']
  for index, answer in enumerate(list(data)):
    if 'title' in answer:
      dataArr[index] = unicodedata.normalize('NFKD', answer['title']).encode('ascii','ignore') + '\n'
    for message in answer['messages']:
      if 'text' in message:
        processedMsg = unicodedata.normalize('NFKD', message['text']).encode('ascii','ignore')
        if type(processedMsg) is str:
          dataArr[index] = dataArr[index] + '\n' + processedMsg
  with open('data.pkl', 'wb') as output:
    pickle.dump(dataArr, output, pickle.HIGHEST_PROTOCOL)

print 'Processing data...'

stop_words = text.ENGLISH_STOP_WORDS.union(['just', 'know', 'http', 'like', 'time', 'good'])
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(dataArr)
'''lda_clf = Pipeline([('count_vect', CountVectorizer(stop_words='english', lowercase=True)),
                    ('tfidf', TfidfVectorizer(norm='l2')),
                    ('clf', lda)])'''

tf_feature_names = tf_vectorizer.get_feature_names()

print 'Running Gridsearch...'
#tuned_parameters = [{'n_components': [5, 6]}]

tuned_parameters = [{'n_components': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'learning method': ['batch', 'online'], 'learning_decay': [0.51, 0.6, 0.7, 0.8, 0.9, 1.0], 'learning_offset': [1, 5, 10, 20, 100], 'batch_size': [64, 128, 256]}]

clf = GridSearchCV(LatentDirichletAllocation(), tuned_parameters, cv=5, verbose=3)
clf.fit(tf)

print 'Best params set found on development set:'
print clf.best_params_
print 'Grid scores on development set:'
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))