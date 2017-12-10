from __future__ import print_function
import numpy as np
import math
import tensorflow as tf
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

pickle_file = 'notMNIST.pickle'

print("\n-------------------------------------------------------------------")
print("Opening file ...")
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
print("Done!")

print("\n-------------------------------------------------------------------")
print("Reformat ...")
image_size = 28
num_labels = 10
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print("Done !")

print("\n-------------------------------------------------------------------")
print("Build network : 1 hidden layer + ReLU + L2 regularization")

#hidden_layer_spec = np.array([1024])
#hidden_layer_spec = np.array([1024, 300])
hidden_layer_spec = np.array([1024, 300, 50])
num_hidden_layers = hidden_layer_spec.shape[0]
batch_size = 256
beta = 0.0005
initial_learning_rate = 0.05

epochs = 500

stepsPerEpoch = float(train_dataset.shape[0]) / batch_size
num_steps = int(math.ceil(float(epochs) * stepsPerEpoch))

l2Graph = tf.Graph()
with l2Graph.as_default():
  #with tf.device('/cpu:0'):
      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      weights = []
      biases = []
      for hi in range(0, num_hidden_layers + 1):
        width = image_size * image_size if hi == 0 else hidden_layer_spec[hi - 1]
        height = num_labels if hi == num_hidden_layers else hidden_layer_spec[hi]
        weights.append(tf.Variable(tf.truncated_normal([width, height], stddev=np.sqrt(2.0 /float(width))), name = "w" + `hi + 1`))
        biases.append(tf.Variable(tf.zeros([height]), name = "b" + `hi + 1`))
        print(`width` + 'x' + `height`)

      def dropoutLayer(layer, addDropoutLayer):
        if(addDropoutLayer):
          return tf.nn.dropout(layer, 0.5)
        else:
          return layer

      def logits(input, addDropoutLayer = False):
        previous_layer = input
        for hi in range(0, hidden_layer_spec.shape[0]):
          previous_layer = tf.nn.relu(tf.matmul(previous_layer, weights[hi]) + biases[hi])
          if addDropoutLayer:
            previous_layer = tf.nn.dropout(previous_layer, 0.5)
        return tf.matmul(previous_layer, weights[num_hidden_layers]) + biases[num_hidden_layers]

      # Training computation.
      train_logits = logits(tf_train_dataset, True)

      l2 = tf.nn.l2_loss(weights[0])
      for hi in range(1, len(weights)):
        l2 = l2 + tf.nn.l2_loss(weights[0])
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=tf_train_labels)) + beta * l2

      # Optimizer.
      global_step = tf.Variable(0)  # count the number of steps taken.
      learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, int(stepsPerEpoch) * 2, 0.96, staircase = True)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(train_logits)
      valid_prediction = tf.nn.softmax(logits(tf_valid_dataset))
      test_prediction = tf.nn.softmax(logits(tf_test_dataset))
      saver = tf.train.Saver()

with tf.Session(graph=l2Graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  loss_value = []
  step_value = []
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    step_value.append(step)
    loss_value.append(l)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Learning rate: " % learning_rate)
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  final_test_result = accuracy(test_prediction.eval(), test_labels)
  print("Test accuracy: %.1f%%" % final_test_result)
  save_path = saver.save(session, "l2_degrade.ckpt")
  print("Model save to " + `save_path`)

  fig = plt.figure()
  plt.plot(step_value, loss_value)
  path_image = "./loss/"
  for layer in hidden_layer_spec:
    path_image += str(layer)
    path_image += "-"
  path_image += "("
  path_image += str(initial_learning_rate)
  path_image += ")"
  path_image += str(epochs)
  path_image += "["
  path_image += str(final_test_result)
  path_image += "]"
  path_image += ".png"
  fig.savefig(path_image)
  plt.show()
