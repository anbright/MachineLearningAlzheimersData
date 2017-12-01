import numpy as np
from pymongo import MongoClient
import os.path
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text
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
print 'LDA:'
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

stop_words = text.ENGLISH_STOP_WORDS.union(['just', 'know', 'http', 'like', 'time', 'good'])
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(dataArr)
'''lda_clf = Pipeline([('count_vect', CountVectorizer(stop_words='english', lowercase=True)),
                    ('tfidf', TfidfVectorizer(norm='l2')),
                    ('clf', lda)])'''

tf_feature_names = tf_vectorizer.get_feature_names()

if os.path.isfile('ldaModel.pkl'):
  with open('ldaModel.pkl', 'rb') as input:
    lda = pickle.load(input)
else:
  lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online', learning_offset=50, random_state=519)
  print 'Fitting Model...'
  lda.fit(tf)
  with open('ldaModel.pkl', 'wb') as output:
    pickle.dump(lda, output, pickle.HIGHEST_PROTOCOL)

print ('\nTopics in LDA model:')
print_top_words(lda, tf_feature_names, 5)

training_features = lda.transform(tf)
print tf[0].shape
print training_features[0]
print '\n'
print training_features[1]