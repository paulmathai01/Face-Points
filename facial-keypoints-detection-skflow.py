
import sys
import numpy as np
import pandas
import tensorflow as tf
import skflow
from sklearn import metrics
from util import *


train_dataset, train_labels = load_data()
test_dataset, _ = load_data(test=True)

# Generate a validation set.
validation_dataset = train_dataset[:VALIDATION_SIZE, ...]
validation_labels = train_labels[:VALIDATION_SIZE]
train_dataset = train_dataset[VALIDATION_SIZE:, ...]
train_labels = train_labels[VALIDATION_SIZE:]

train_size = train_labels.shape[0]
print("train size is %d" % train_size)


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


# setup exponential decay function
def exp_decay(global_step):
    return tf.train.exponential_decay(
        learning_rate=1e-3, global_step=global_step,
        decay_steps=train_size, decay_rate=0.95)


def cnn_model(X, y):
    # first conv layer will compute 32 features for each 3x3 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[3, 3],
                                    bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    # second conv layer will compute 64 features for each 2x2 patch
    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[2, 2],
                                    bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
    # third conv layer will compute 128 features for each 2x2 patch
    with tf.variable_scope('conv_layer3'):
        h_conv3 = skflow.ops.conv2d(h_pool2, n_filters=128, filter_shape=[2, 2],
                                    bias=True, activation=tf.nn.relu)
        h_pool3 = max_pool_2x2(h_conv3)
        # reshape tensor into a batch of vectors
        h_pool3_flat = tf.reshape(h_pool3, [-1, IMAGE_SIZE // 8 * IMAGE_SIZE // 8 * 128])
    # densely connected layer with 1024 neurons
    h_fc1 = skflow.ops.dnn(h_pool3_flat, [500, 500], activation=tf.nn.relu, keep_prob=0.5)
    return skflow.models.linear_regression(h_fc1, y)


estimator = skflow.Estimator(model_fn=cnn_model, early_stopping_rounds=EARLY_STOP_PATIENCE, steps=10, optimizer='Adam',learning_rate=exp_decay, continue_training=True)

title = 'learning curve for cnn'
# to plot the learning curve, continue_training must be set False
# generate_learning_curve(estimator, title, 'mean_squared_error', train_dataset, train_labels)


# Continuesly train for 100 steps & predict on test set.
for i in xrange(0, 100):
    estimator.fit(train_dataset, train_labels, logdir='log')
    score = metrics.mean_squared_error(validation_labels, estimator.predict(validation_dataset))
    print('mean squared error: {0:f}'.format(score))
    sys.stdout.flush()

test_labels = estimator.predict(test_dataset, batch_size=EVAL_BATCH_SIZE)
make_submission(test_labels)

