import tensorflow as tf

# Build graph using TF operation
xTrain = [1, 2, 3]
yTrain = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis xw+b
hypothesis = xTrain * W + b

# cost/loss function // reduce_mean is average.
cost = tf.reduce_mean(tf.square(hypothesis - yTrain))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# run and update graph and get results.
# launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

#fit the line
for step in range(2001):
    sess.run(train)
    if step%20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))


