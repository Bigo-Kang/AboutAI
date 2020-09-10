import tensorflow as tf
import matplotlib.pyplot as plt
X = [1,2,3]
Y = [1,2,3]

W = tf.placeholder(tf.float32)
# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())
# variables for plotting cost function
w_val = []
c_val = []

for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    w_val.append(curr_W)
    c_val.append(curr_cost)

sess.close()
# show the cost function
plt.plot(w_val, c_val)
plt.show()