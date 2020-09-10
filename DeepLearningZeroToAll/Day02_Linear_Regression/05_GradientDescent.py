import tensorflow as tf

X = [1, 2, 3]
Y = [7, 14, 21]

W = tf.Variable(-10.0)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print('{:3} {}'.format(step, sess.run(W)))
    sess.run(train)