import numpy as np
from pymongo import MongoClient
import os.path
import unicodedata
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import text
from sklearn.decomposition import NMF

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

print 'Extracting tf-idf features for NMF...'
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(dataArr)

print 'Fitting the NMF model with tf-idf features...'
nmf = NMF(n_components=NUM_TOPICS, random_state=519, alpha=.1, l1_ratio=.5).fit(tfidf)

print '\nTopics in NMF model:'
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)