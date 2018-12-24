
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import *

import time
import sys


def error_measure(predictions, labels):
    return np.sum(np.power(predictions - labels, 2)) / (2 * predictions.shape[0])


if __name__ == '__main__':
    train_dataset, train_labels = load_data()
    test_dataset, _ = load_data(test=True)

    # Generate a validation set.
    validation_dataset = train_dataset[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_dataset = train_dataset[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]

    train_size = train_labels.shape[0]
    print("train size is %d" % train_size)

    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))

    eval_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))

    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

    fc1_weights = tf.Variable(  # fully connected, depth 512.
                                tf.truncated_normal(
                                    [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                                    stddev=0.1,
                                    seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

    fc2_weights = tf.Variable(  # fully connected, depth 512.
                                tf.truncated_normal(
                                    [512, 512],
                                    stddev=0.1,
                                    seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[512]))

    fc3_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc3_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # conv = tf.Print(conv, [conv], "conv1: ", summarize=10)

        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # relu = tf.Print(relu, [relu], "relu1: ", summarize=10)

        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        # pool = tf.Print(pool, [pool], "pool1: ", summarize=10)

        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # conv = tf.Print(conv, [conv], "conv2: ", summarize=10)

        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        # relu = tf.Print(relu, [relu], "relu2: ", summarize=10)

        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        # pool = tf.Print(pool, [pool], "pool2: ", summarize=10)

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # reshape = tf.Print(reshape, [reshape], "reshape: ", summarize=10)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # hidden = tf.Print(hidden, [hidden], "hidden1: ", summarize=10)

        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

        hidden = tf.nn.relu(tf.matmul(hidden, fc2_weights) + fc2_biases)
        # hidden = tf.Print(hidden, [hidden], "hidden2: ", summarize=10)

        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

        # return tf.nn.tanh(tf.matmul(hidden, fc3_weights) + fc3_biases)
        return tf.matmul(hidden, fc3_weights) + fc3_biases

    train_prediction = model(train_data_node, True)

    # Minimize the squared errors
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(train_prediction - train_labels_node), 1))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
                    tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases))
    # Add the regularization term to the loss.
    loss += 1e-7 * regularizers

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = model(eval_data_node)

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    global_step = tf.Variable(0, trainable=False)

    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        1e-3,                      # Base learning rate.
        global_step * BATCH_SIZE,  # Current index into the dataset.
        train_size,                # Decay step.
        0.95,                      # Decay rate.
        staircase=True)

    # train_step = tf.train.AdamOptimizer(5e-3).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
    # train_step = tf.train.MomentumOptimizer(1e-4, 0.95).minimize(loss)
    train_step = tf.train.AdamOptimizer(learning_rate, 0.95).minimize(loss, global_step=global_step)

    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    loss_train_record = list() # np.zeros(n_epoch)
    loss_valid_record = list() # np.zeros(n_epoch)
    start_time = time.gmtime()

    # early stopping
    best_valid = np.inf
    best_valid_epoch = 0

    current_epoch = 0

    while current_epoch < NUM_EPOCHS:
        # Shuffle data
        shuffled_index = np.arange(train_size)
        np.random.shuffle(shuffled_index)
        train_dataset = train_dataset[shuffled_index]
        train_labels = train_labels[shuffled_index]

        for step in xrange(train_size / BATCH_SIZE):
            offset = step * BATCH_SIZE
            batch_data = train_dataset[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            _, loss_train, current_learning_rate = sess.run([train_step, loss, learning_rate], feed_dict=feed_dict)

        # After one epoch, make validation
        eval_result = eval_in_batches(validation_dataset, sess, eval_prediction, eval_data_node)
        loss_valid = error_measure(eval_result, validation_labels)

        print 'Epoch %04d, train loss %.8f, validation loss %.8f, train/validation %0.8f, learning rate %0.8f' % (
            current_epoch,
            loss_train, loss_valid,
            loss_train / loss_valid,
            current_learning_rate
        )
        loss_train_record.append(np.log10(loss_train))
        loss_valid_record.append(np.log10(loss_valid))
        sys.stdout.flush()

        if loss_valid < best_valid:
            best_valid = loss_valid
            best_valid_epoch = current_epoch
        elif best_valid_epoch + EARLY_STOP_PATIENCE < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(best_valid, best_valid_epoch))
            break

        current_epoch += 1

    print('train finish')
    end_time = time.gmtime()
    print time.strftime('%H:%M:%S', start_time)
    print time.strftime('%H:%M:%S', end_time)

    generate_submission(test_dataset, sess, eval_prediction, eval_data_node)

    # Show an example of comparison
    i = 0
    img = validation_dataset[i]
    lab_y = validation_labels[i]
    lab_p = eval_in_batches(validation_dataset, sess, eval_prediction, eval_data_node)[0]
    plot_sample(img, lab_p, lab_y)

    plot_learning_curve(loss_train_record, loss_valid_record)


