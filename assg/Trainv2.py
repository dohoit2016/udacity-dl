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
num_channels = 1


def reformat(dataset, labels):
    dataset = dataset.reshape(-1, image_size, image_size, num_channels).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training_dataset', train_dataset.shape, train_labels.shape)
print('Test_dataset', test_dataset.shape, test_labels.shape)
print('Valid_dataset', valid_dataset.shape, valid_labels.shape)


def accuracy(predictons, labels):
    return 100.0 * np.sum(np.argmax(predictons, 1) == np.argmax(labels, 1)) / predictons.shape[0]

train_subset = 10000
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64


graph = tf.Graph()
with graph.as_default():
    # Input and output
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Model parameters (Variables)
    w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    b1 = tf.Variable(tf.zeros([depth]))

    w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[depth]))

    w3 = tf.Variable(tf.truncated_normal([image_size // 4 * image_size //4 * depth, num_hidden], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    w4 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model
    def model(data):
        conv = tf.nn.conv2d(data, w1, [1,2,2,1], padding="SAME")
        hidden = tf.nn.relu(conv + b1)
        conv = tf.nn.conv2d(hidden, w2, [1,2,2,1], padding="SAME")
        hidden = tf.nn.relu(conv + b2)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, w3) + b3)
        return tf.matmul(hidden, w4) + b4


    # Training computation
    logits = model(tf_train_dataset)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = model(tf_valid_dataset)
    test_prediction = model(tf_test_dataset)

num_steps = 1001


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

        _, l, predictons = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 300 == 0:
            print("Loss at step %d: %f" % (step, l))
            print("Training accuracy: %.1f%%" % accuracy(predictons, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
            print("-------------------------\n")
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
