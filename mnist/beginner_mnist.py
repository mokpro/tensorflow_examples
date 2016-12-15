# Get the image data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Init variables
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Setup Softmax neural net
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Training
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_steps = [
    tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy),
    tf.train.AdagradOptimizer(0.3).minimize(cross_entropy),
    tf.train.AdadeltaOptimizer(0.3).minimize(cross_entropy),
    tf.train.ProximalAdagradOptimizer(0.3).minimize(cross_entropy),
    tf.train.ProximalGradientDescentOptimizer(0.3).minimize(cross_entropy)
]

# Init and run

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for train_step in train_steps:
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(train_step.name, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
