import numpy as np
import random, sys
import os
import cPickle

import keras
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers import Merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split



## Hyper Parameters:

num_train_samples = 30000
num_val_samples = 3000
num_unknown_words = 20
maxlen = 50

rnn_size = 512 
rnn_layers = 3 
activation_rnn_size = 40

## Loading words
with open('data/vocabulary-embedding.pkl', 'rb') as f:
	embedding, idx2word, word2idx, glove_idx2idx = cPickle.load(f)

vocab_size, embedding_size = embedding.shape

with open('data/vocabulary-embedding.data.pkl') as f:
	x, y = cPickle.load(f)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=num_val_samples)

# Free up space in memory
del x
del y

## Formatting word embeddings

# empty has an index of 0 in the above word embedding
empty = 0
idx2word[empty] = '_'
# end of sentence (<eos>) has an index of 0 in the above word embedding
eos = 1
idx2word[eos] = '~'

# adding space in the embedding space for words that are not in the trained vector
for k in range(num_unknown_words):
	idx2word[vocab_size-1-k] = '<{}>'.format(k)

size_out_of_vector = vocab_size - num_unknown_words

# Adds a '~' to all out of vector words
for k in range(size_out_of_vector, len(idx2word)):
	idx2word[k] = idx2word[k]+'~'

## Creates the model in Keras

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=maxlen, weights=[embedding], mask_zero=True, name='embedding_1'))

for i in range(rnn_layers):
	lstm = LSTM(rnn_size, return_sequences=True, name='lstm_%d'%(i+1))
	model.add(lstm)

from keras.layers.core import Lambda
import keras.backend as K


## Reference:  https://github.com/llSourcell/How_to_make_a_text_summarizer/blob/master/vocabulary-embedding.ipynb
def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
    desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
    head_activations, head_words = head[:,:,:n], head[:,:,n:]
    desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
    
    # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
    # activation for every head word and every desc word
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
    # make sure we dont use description words that are masked out
    activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)
    
    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies,(-1,maxlend))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

    # for every head word compute weighted average of desc words
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
    return K.concatenate((desc_avg_word, head_words))


class SimpleContext(Lambda):
    def __init__(self,**kwargs):
        super(SimpleContext, self).__init__(simple_context,**kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:, maxlend:]
    
    def get_output_shape_for(self, input_shape):
        nb_samples = input_shape[0]
        n = 2*(rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)

model.add(SimpleContext(name='simplecontext_1'))
model.add(TimeDistributed(Dense(vocab_size, name = 'timedistributed_1')))

model.add(Activation('softmax', name='activation_1'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# # Trains the encoder
# encoder = build_model(embeddings)
# encoder.compile(loss='categorical_crossentropy', optimizer='adam')
# encoder.save_weights('data/embeddings.pkl', overwrite=True)

# # Trains the decoder
# with open('data/embeddings.pkl', 'rb') as f:
# 	embeddings = pkl.load(f)

# decoder = build_model(embeddings)
# Data generator

"""Data generator generates batches of inputs and outputs/labels for training. The inputs are each made from two parts. The first maxlend words are the original description, followed by `eos` followed by the headline which we want to predict, except for the last word in the headline which is always `eos` and then `empty` padding until `maxlen` words.
For each, input, the output is the headline words (without the start `eos` but with the ending `eos`) padded with `empty` words up to `maxlenh` words. The output is also expanded to be y-hot encoding of each word.
To be more realistic, the second part of the input should be the result of generation and not the original headline.
Instead we will flip just `nflips` words to be from the generator, but even this is too hard and instead
implement flipping in a naive way (which consumes less time.) Using the full input (description + eos + headline) generate predictions for outputs. For nflips random words from the output, replace the original word with the word with highest probability from the prediction.
"""

def flip_headline(x, nflips=None, model=None, debug=False):
    """given a vectorized input (after `pad_sequences`) flip some of the words in the second half (headline)
    with words predicted by the model
    """
    if nflips is None or model is None or nflips <= 0:
        return x

    batch_size = len(x)
    assert np.all(x[:, maxlend] == eos)
    probs = model.predict(x, verbose=0, batch_size=batch_size)
    x_out = x.copy()
    for b in range(batch_size):
        # pick locations we want to flip
        # 0...maxlend-1 are descriptions and should be fixed
        # maxlend is eos and should be fixed
        flips = sorted(random.sample(range(maxlend + 1, maxlen), nflips))
        if debug and b < debug:
            print(b)
        for input_idx in flips:
            if x[b, input_idx] == empty or x[b, input_idx] == eos:
                continue
            # convert from input location to label location
            # the output at maxlend (when input is eos) is feed as input at maxlend+1
            label_idx = input_idx - (maxlend + 1)
            prob = probs[b, label_idx]
            w = prob.argmax()
            if w == empty:  # replace accidental empty with oov
                w = oov0
            if debug and b < debug:
                print('{} => {}'.format(idx2word[x_out[b, input_idx]], idx2word[w]),)
            x_out[b, input_idx] = w
        if debug and b < debug:
            print()
    return x_out

def conv_seq_labels(xds, xhs, nflips=None, model=None, debug=False):
    """description and hedlines are converted to padded input vectors. headlines are one-hot to label"""
    batch_size = len(xhs)
    assert len(xds) == batch_size
    x = [vocab_fold(lpadd(xd) + xh) for xd, xh in zip(xds, xhs)]  # the input does not have 2nd eos
    x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')
    x = flip_headline(x, nflips=nflips, model=model, debug=debug)

    y = np.zeros((batch_size, maxlenh, vocab_size))
    for i, xh in enumerate(xhs):
        xh = vocab_fold(xh) + [eos] + [empty] * maxlenh  # output does have a eos at end
        xh = xh[:maxlenh]
        y[i, :, :] = np_utils.to_categorical(xh, vocab_size)

    return x, y

def gen(Xd, Xh, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False, seed=seed):
    """yield batches. for training use nb_batches=None
    for validation generate deterministic results repeating every nb_batches
    while training it is good idea to flip once in a while the values of the headlines from the
    value taken from Xh to value generated by the model.
    """
    c = nb_batches if nb_batches else 0
    while True:
        xds = []
        xhs = []
        if nb_batches and c >= nb_batches:
            c = 0
        new_seed = random.randint(0, 2e10)
        random.seed(c + 123456789 + seed)
        for b in range(batch_size):
            t = random.randint(0, len(Xd) - 1)

            xd = Xd[t]
            s = random.randint(min(maxlend, len(xd)), max(maxlend, len(xd)))
            xds.append(xd[:s])

            xh = Xh[t]
            s = random.randint(min(maxlenh, len(xh)), max(maxlenh, len(xh)))
            xhs.append(xh[:s])

        # undo the seeding before we yield inorder not to affect the caller
        c += 1
        random.seed(new_seed)

        yield conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)

def test_gen(gen, n=5):
    Xtr, Ytr = next(gen)
    for i in range(n):
        assert Xtr[i, maxlend] == eos
        x = Xtr[i, :maxlend]
        y = Xtr[i, maxlend:]
        yy = Ytr[i, :]
        yy = np.where(yy)[1]
        prt('L', yy)
        prt('H', y)
        if maxlend:
            prt('D', x)

r = next(gen(X_train, Y_train, batch_size=batch_size))
valgen = gen(X_test, Y_test, nb_batches=3, batch_size=batch_size)

# Train
history = {}
traingen = gen(X_train, Y_train, batch_size=batch_size, nflips=args.nflips, model=model)
valgen = gen(X_test, Y_test, nb_batches=nb_val_samples // batch_size, batch_size=batch_size)

callbacks = [TensorBoard(
    log_dir=os.path.join(config.path_logs, str(time.time())),
    histogram_freq=2, write_graph=False, write_images=False)]

h = model.fit_generator(
    traingen, samples_per_epoch=nb_train_samples,
    nb_epoch=args.epochs, validation_data=valgen, nb_val_samples=nb_val_samples,
    callbacks=callbacks,
)
for k, v in h.history.items():
    history[k] = history.get(k, []) + v
with open(os.path.join(config.path_models, 'history.pkl'.format(FN)), 'wb') as fp:
    pickle.dump(history, fp, -1)
model.save_weights(FN1_filename, overwrite=True)
gensamples(skips=2, batch_size=batch_size, k=10, temperature=args.temperature)


