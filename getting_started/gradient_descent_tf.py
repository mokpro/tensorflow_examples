import tensorflow as tf
import numpy as np

# Create 100 x, y points
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.42 + 0.7

# Try to find W and b to compute y_date = W * x_data + b using TF
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize mean square errirs
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

# Launch the graph

sess = tf.Session()
sess.run(init)

# Fit the line

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
