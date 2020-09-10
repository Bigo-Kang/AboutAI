import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
nb_classes = 7

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # -1 is everything
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(predicted, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {X: x_data, Y: y_data}
    for step in range(10001):
        sess.run(optimizer, feed_dict=feed)
        if step % 200 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict=feed)
            print("Step : {:5}\tLoss : {:.3f}\tAcc : {:.2%}".format(
                step, loss, acc
            ))

    # Let's see if we can predict.
    pred = sess.run(predicted, feed_dict={X: x_data})
    # y_data : (N,1) = flatten -> (N, ) matches pred.shape // [[1], [0]] -> [1, 0]
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Predictoin : {} True Y : {}".format(p == int(y), p, int(y)))