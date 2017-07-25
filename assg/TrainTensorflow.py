from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

# read data
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape(-1, image_size * image_size).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training_dataset', train_dataset.shape, train_labels.shape)
print('Test_dataset', test_dataset.shape, test_labels.shape)
print('Valid_dataset', valid_dataset.shape, valid_labels.shape)

train_subset = 10000
batch_size = 128
num_nodes1 = 1024
num_nodes2 = 512
beta = 1e-4


def prediction(tf_dataset, w1, b1, w2, b2, w3, b3):
    y1 = tf.matmul(tf_dataset, w1) + b1

    x2 = tf.nn.relu(y1)
    y2 = tf.matmul(x2, w2) + b2

    x3 = tf.nn.relu(y2)
    y3 = tf.matmul(x3, w3) + b3

    return tf.nn.softmax(y3)

graph = tf.Graph()
with graph.as_default():
    # Input and output
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Model parameters (Variables)
    w1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([num_nodes1]))

    w2 = tf.Variable(tf.truncated_normal([num_nodes1, num_nodes2], stddev=0.1))
    b2 = tf.Variable(tf.zeros(num_nodes2))

    w3 = tf.Variable(tf.truncated_normal([num_nodes2, num_labels], stddev=0.1))
    b3 = tf.Variable(tf.zeros(num_labels))

    keep_prob = tf.placeholder(tf.float32)
    # Training computation
    y1 = tf.matmul(tf_train_dataset, w1) + b1

    x2 = tf.nn.relu(y1)
    x2_drop = tf.nn.dropout(x2, keep_prob)
    y2 = tf.matmul(x2_drop, w2) + b2

    x3 = tf.nn.relu(y2)
    x3_drop = tf.nn.dropout(x3, keep_prob)
    logits = tf.matmul(x3_drop, w3) + b3

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    regularizers = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    loss += beta*regularizers

    # Optimizer
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.5, global_step, 500, 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # training
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = prediction(valid_dataset, w1, b1, w2, b2, w3, b3)
    test_prediction = prediction(test_dataset, w1, b1, w2, b2, w3, b3)

num_steps = 3001


def accuracy(predictons, labels):
    return 100.0 * np.sum(np.argmax(predictons, 1) == np.argmax(labels, 1)) / predictons.shape[0]


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5}

        _, l, predictons = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 300 == 0:
            print("Loss at step %d: %f" % (step, l))
            print("Training accuracy: %.1f%%" % accuracy(predictons, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
            print("-------------------------\n")
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
