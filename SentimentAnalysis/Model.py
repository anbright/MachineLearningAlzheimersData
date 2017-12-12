import numpy as np
import tensorflow as tf
import datetime
import utils
import csv

units = 64
iterations = 75000

maxLength = 250
batchSize = 24 

words2vec = np.load('words2vec.npy')
vectors = np.load('vectors.npy')


## creating the model
def createModel(dropout, lstmUnits, iterations):
  tf.reset_default_graph()

  labels = tf.placeholder(tf.float32, [batchSize, 2], name='labels')
  input_data = tf.placeholder(tf.int32, [batchSize, maxLength], name='input_data')

  ## Input labels
  data = tf.Variable(tf.zeros([batchSize, maxLength, 300]),dtype=tf.float32)
  data = tf.nn.embedding_lookup(vectors,input_data)

  ## Creating layers
  lstm = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
  lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=dropout)
  value, _ = tf.nn.dynamic_rnn(lstm, data, dtype=tf.float32)

  weights = tf.Variable(tf.truncated_normal([units, 2]))
  bias = tf.Variable(tf.constant(0.1, shape=[2]))
  value = tf.transpose(value, [1, 0, 2])
  last = tf.gather(value, int(value.get_shape()[0]) - 1)
  prediction = (tf.matmul(last, weights) + bias)

  correct = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
  optimizer = tf.train.AdamOptimizer().minimize(loss)

  sess = tf.InteractiveSession()
  saver = tf.train.Saver()
  sess.run(tf.global_variables_initializer())

  tf.summary.scalar('Loss', loss)
  tf.summary.scalar('Accuracy', accuracy)
  merged = tf.summary.merge_all()
  logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
  writer = tf.summary.FileWriter(logdir, sess.graph)

  for i in range(iterations):
     #Next Batch of reviews
     batch, batchLabels = utils.trainBatch(batchSize, maxLength, words2vec);
     sess.run(optimizer, {input_data: batch, labels: batchLabels})
     
     #Write summary to Tensorboard
     if (i % 50 == 0):
         summary = sess.run(merged, {input_data: batch, labels: batchLabels})
         writer.add_summary(summary, i)

     #Save the network every 5,000 training iterations
     if (i % 5000 == 0 and i != 0):
         save_path = saver.save(sess, "models/lstm.ckpt", global_step=i)
         print("saved to %s" % save_path)
  writer.close()
  

def test_accuracy(dropout, lstmUnits, modelDir):
  tf.reset_default_graph()

  labels = tf.placeholder(tf.float32, [batchSize, 2], name='labels')
  input_data = tf.placeholder(tf.int32, [batchSize, maxLength], name='input_data')

  ## Input labels
  data = tf.Variable(tf.zeros([batchSize, maxLength, 300]),dtype=tf.float32)
  data = tf.nn.embedding_lookup(vectors,input_data)

  ## Creating layers
  lstm = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
  lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=dropout)
  value, _ = tf.nn.dynamic_rnn(lstm, data, dtype=tf.float32)

  weights = tf.Variable(tf.truncated_normal([units, 2]))
  bias = tf.Variable(tf.constant(0.1, shape=[2]))
  value = tf.transpose(value, [1, 0, 2])
  last = tf.gather(value, int(value.get_shape()[0]) - 1)
  prediction = (tf.matmul(last, weights) + bias)

  correct = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

  sess = tf.InteractiveSession()
  saver = tf.train.Saver()
  saver.restore(sess, tf.train.latest_checkpoint(modelDir))

  scores = []
  for k in range(10):
    batch, batchLabels = utils.testBatch(batchSize, maxLength, words2vec)
    score = sess.run(accuracy, {input_data: batch, labels: batchLabels})
    scores.append(score)

  return scores

def GridSearch(dropouts, lstmUnits):
  results = []
  for dropout in dropouts:
    for units in lstmUnits:
      createModel(dropout = dropout, lstmUnits = units, iterations = 75000)
      scores = test_accuracy(dropout = dropout, lstmUnits = units, modelDir='models')
      scores = np.asarray(scores)

      mean = np.mean(scores)
      std = np.std(scores)

      results.append({'dropout':dropout, 'num_units': units, 'mean': mean, 'std': std})

      with open('results_mid.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([mean, std, dropout, units])
      
      return results
## training and param selection
# createModel(dropout = 0.75, lstmUnits = 64, iterations = 75000)
dropouts = [0.25, 0.5, 0.75]
lstmUnits = [64, 128, 256]
results = GridSearch(dropouts, lstmUnits)
print(results)

with open('results.csv', 'w+') as f:
  writer = csv.writer(f)
  for result in results:
    mean = result['mean']
    std = result['std']
    dropout = result['dropout']
    numUnits = result['num_units']

    writer.writerow([mean, std, dropout, numUnits])
