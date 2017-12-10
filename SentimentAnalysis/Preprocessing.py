"""
Loads in testing and training data and converts them to a word2vec representation.

The word2vec representation we are using is GloVe (https://nlp.stanford.edu/projects/glove/) 
which is a word embedding model trained by stanford. They have trained millions of words into a 
100 dimensional space. Word embedding models are an attempted to represent the semantic richness 
of words and group together synonyms.

The sentiment analysis data is from IMDB:
http://ai.stanford.edu/~amaas/data/sentiment/

"""

import numpy as np
# import tensorflow as tf
from os import listdir
from os.path import isfile, join
import io
from nltk.tokenize import RegexpTokenizer

# maximum document word length
maxSeq = 275

## Loading in the GloVe model
words = np.load('words.npy')
vectors = np.load('vectors.npy')

words.tolist()
words = [word.decode('UTF-8') for word in words]

## Loading in the reviews

positiveFiles = []
for f in listdir('positive/'):
	if isfile(join('positive/', f)):
		positiveFiles.append(f)

negativeFiles = []
for f in listdir('negative/'):
	if isfile(join('negative/', f)):
		negativeFiles.append(f)

# Vector to be saved out
words2vec = np.zeros((25000, maxSeq), dtype='int32')

# Tokenizing and cleaning text
tokenizer = RegexpTokenizer(r'\w+')

file_num = 0

print('Transforming Positive Data')

for pf in positiveFiles:
	with open('positive/' + pf, 'r') as f:
		index = 0
		l = f.readline().lower()
		tokens = tokenizer.tokenize(l)
		for token in tokens:
			try:
				words2vec[file_num][index] = words.index(word)
			except ValueError:
				words2vec[file_num][index] = 399999 ## unknown word 
			index += 1
			if index >= maxSeq:
				break
		file_num += 1

print('Transforming Negative Data')

for nf in negativeFiles:
	with open('negative/' + nf, 'r') as f:
		index = 0
		l = f.readline().lower()
		tokens = tokenizer.tokenize(l)
		for token in tokens:
			try:
				words2vec[file_num][index] = words.index(word)
			except ValueError:
				words2vec[file_num][index] = 399999 ## unknown word 
			index += 1
			if index >= maxSeq:
				break
		file_num += 1

np.save('words2vec', words2vec)

