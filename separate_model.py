
import sys
import gc
import time
import tensorflow as tf
import skflow
from sklearn import cross_validation, metrics
from util import *


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


# setup exponential decay function
def exp_decay(global_step):
    return tf.train.exponential_decay(
        learning_rate=1e-3, global_step=global_step,
        decay_steps=30, decay_rate=0.95)


def cnn_model(X, y):
    # first conv layer will compute 32 features for each 3x3 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[3, 3], batch_norm=True,
                                    bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
        h_pool1 = tf.nn.dropout(h_pool1, 0.5, seed=SEED)
    # second conv layer will compute 64 features for each 2x2 patch
    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[2, 2], batch_norm=True,
                                    bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2 = tf.nn.dropout(h_pool2, 0.5, seed=SEED)
    # third conv layer will compute 128 features for each 2x2 patch
    with tf.variable_scope('conv_layer3'):
        h_conv3 = skflow.ops.conv2d(h_pool2, n_filters=128, filter_shape=[2, 2], batch_norm=True,
                                    bias=True, activation=tf.nn.relu)
        h_pool3 = max_pool_2x2(h_conv3)
        h_pool3 = tf.nn.dropout(h_pool3, 0.5, seed=SEED)
        # reshape tensor into a batch of vectors
        h_pool3_flat = tf.reshape(h_pool3, [-1, IMAGE_SIZE // 8 * IMAGE_SIZE // 8 * 128])
    # densely connected layer with 1024 neurons
    h_fc1 = skflow.ops.dnn(h_pool3_flat, [500, 500], activation=tf.nn.relu, keep_prob=0.5)
    return skflow.models.linear_regression(h_fc1, y)


# estimator = skflow.TensorFlowEstimator(model_fn=cnn_model, n_classes=0, batch_size=BATCH_SIZE,
#                                        early_stopping_rounds=EARLY_STOP_PATIENCE, steps=NUM_EPOCHS, optimizer='Adam',
#                                        learning_rate=exp_decay, continue_training=False)

train_dataframe = load_dataframe(test=False)
test_dataframe = load_dataframe(test=True)

test_data = extract_test_data(test_dataframe)

predicted_labels = np.empty((test_dataframe.shape[0], 0))
columns = ()

for setting in SPECIALIST_SETTINGS:
    print time.strftime('%Y-%m-%d %H:%M:%S')
    sys.stdout.flush()
    cols = setting['columns']
    flip_indices = setting['flip_indices']
    print(cols)
    sys.stdout.flush()
    columns += cols
    train_data, train_labels = extract_train_data(train_dataframe, cols=cols, flip_indices=flip_indices)
    train_data, validation_data, train_labels, validation_labels = \
        cross_validation.train_test_split(train_data, train_labels, test_size=0.05, random_state=SEED)

    estimator = skflow.TensorFlowEstimator(model_fn=cnn_model, n_classes=0, batch_size=BATCH_SIZE, num_cores=1,
                                                 early_stopping_rounds=EARLY_STOP_PATIENCE, steps=NUM_EPOCHS, optimizer='Adam',
                                                 learning_rate=exp_decay, continue_training=False)
    estimator.fit(train_data, train_labels, logdir='log')
    del train_data, train_labels
    gc.collect()
    score = metrics.mean_squared_error(validation_labels, estimator.predict(validation_data, batch_size=EVAL_BATCH_SIZE))
    del validation_data, validation_labels
    gc.collect()
    print('validation mean squared error: {0:f}'.format(score))
    sys.stdout.flush()
    test_labels = estimator.predict(test_data, batch_size=EVAL_BATCH_SIZE)
    del estimator
    gc.collect()
    predicted_labels = np.hstack([predicted_labels, test_labels])
    print time.strftime('%Y-%m-%d %H:%M:%S')
    sys.stdout.flush()


create_submission(predicted_labels, columns)
