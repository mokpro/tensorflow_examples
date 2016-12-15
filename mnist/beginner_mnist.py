from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from arg_parser import *

OPTIMIZERS_NAME_TO_CLASS_MAP = {
    'GradientDescent': tf.train.GradientDescentOptimizer(0.3),
    'Adagrad': tf.train.AdagradOptimizer(0.3),
    'Adadelta': tf.train.AdadeltaOptimizer(0.3),
    'ProximalAdagrad': tf.train.ProximalAdagradOptimizer(0.3),
    'ProximalGradientDescent': tf.train.ProximalGradientDescentOptimizer(0.3)
}


def main(optimizers):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Init variables
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Setup Softmax neural net
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Training
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_steps = []

    for optimizer in optimizers:
        if optimizer in OPTIMIZERS_NAME_TO_CLASS_MAP:
            train_step = OPTIMIZERS_NAME_TO_CLASS_MAP[optimizer]
            train_steps.append(train_step.minimize(cross_entropy))

    # Init and run

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print("\n\n\n")

    for train_step in train_steps:
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        printable_accuracy = sess.run(
            accuracy,
            feed_dict={
                x: mnist.test.images,
                y_: mnist.test.labels})
        print('Optimizer: ' + str(train_step.name) +
              ', Accuracy: ' + str(printable_accuracy))

if __name__ == '__main__':
    optimizers = arg_parser()
    main(optimizers)
