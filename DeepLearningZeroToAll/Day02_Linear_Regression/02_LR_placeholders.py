import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

# Our hypothesis xw+b
hypothesis = x * W + b
# cost/loss function // reduce_mean is average.
cost = tf.reduce_mean(tf.square(hypothesis - y))
# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# run and update graph and get results.
# launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# H = x*w + b  w -> 1, b-> 1.1
#fit the line
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={x: [1, 2, 3, 4, 5],
                                                    y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step%20 == 0:
        print(step, cost_val, W_val, b_val)

# Testing model
print(sess.run(hypothesis, feed_dict={x: [5]}))
print(sess.run(hypothesis, feed_dict={x: [2.5]}))
print(sess.run(hypothesis, feed_dict={x: [1.5, 3.5]}))
