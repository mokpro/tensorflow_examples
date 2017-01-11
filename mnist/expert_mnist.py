from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from arg_parser import *

OPTIMIZERS_NAME_TO_CLASS_MAP = {
    'GradientDescent': tf.train.GradientDescentOptimizer(0.3),
    'Adagrad': tf.train.AdagradOptimizer(0.3),
    'Adadelta': tf.train.AdadeltaOptimizer(0.3),
    'ProximalAdagrad': tf.train.ProximalAdagradOptimizer(0.3),
    'ProximalGradientDescent': tf.train.ProximalGradientDescentOptimizer(0.3),
    'Adam': tf.train.AdamOptimizer(1e-4)
}


def main(optimizers):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Init variables
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Extra stuff for increasing accuracy
    # Layer 1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Layer 2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # Predicted Class and loss function
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # y = tf.matmul(x, W) + b
    # y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            y_conv, y_))

    # Training
    train_steps = []

    for optimizer in optimizers:
        if optimizer in OPTIMIZERS_NAME_TO_CLASS_MAP:
            train_step = OPTIMIZERS_NAME_TO_CLASS_MAP[optimizer]
            train_steps.append(train_step.minimize(cross_entropy))

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Init and run session
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    for train_step in train_steps:
        print("\n\nOptimizer: " + str(train_step.name))

        for i in range(10000):
            batch = mnist.train.next_batch(50)
            train_step.run(
                feed_dict={
                    x: batch[0],
                    y_: batch[1],
                    keep_prob: 0.5})
            if i % 10 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={
                        x: batch[0],
                        y_: batch[1],
                        keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))

        test_accuracy = accuracy.eval(
            feed_dict={
                x: mnist.test.images,
                y_: mnist.test.labels,
                keep_prob: 1.0})
        print('Test Accuracy: ' + str(test_accuracy))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':
    optimizers = arg_parser()
    main(optimizers)
