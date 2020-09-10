import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

nb_classes = 10

# mnist data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784], name="X")
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes], name="Y")

# W1 = tf.Variable(tf.random_normal([784, 676]), name="W1")
# b1 = tf.Variable(tf.random_normal([676]), name="b1")
# layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)

# W2 = tf.Variable(tf.random_normal([676, 676]), name='W2')
# b2 = tf.Variable(tf.random_normal([676]), name='b2')
# layer2 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)

# W3 = tf.Variable(tf.random_normal([676, nb_classes]), name='W3')
# b3 = tf.Variable(tf.random_normal([nb_classes]), name="b3")
# hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

#
with tf.name_scope("layer1") as scope1:
    W1 = tf.Variable(tf.random_normal([784, 676]), name="W1")
    b1 = tf.Variable(tf.random_normal([676]), name="b1")
    layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)

    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("L1", layer1)

with tf.name_scope("layer2") as scope2:
    W2 = tf.Variable(tf.random_normal([676, 676]), name='W2')
    b2 = tf.Variable(tf.random_normal([676]), name='b2')
    layer2 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)

    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("L2", layer2)

with tf.name_scope("layer3") as scope3:
    W3 = tf.Variable(tf.random_normal([676, nb_classes]), name='W3')
    b3 = tf.Variable(tf.random_normal([nb_classes]), name="b3")
    hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

    tf.summary.histogram("W3", W3)
    tf.summary.histogram("b3", b3)
    tf.summary.histogram("Hypothesis", hypothesis)

with tf.name_scope("cost") as cost1:
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("train") as opti:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
tf.summary.scalar("acc", accuracy)
#

# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))



# parameters
training_epochs = 3
batch_size = 100

with tf.Session() as sess:
    #
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/mnist1')
    writer.add_graph(sess.graph)
    #
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        s = tf.float32
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            c, s, _ = sess.run([cost, summary, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
        writer.add_summary(s, global_step=epoch)

        print('Epoch :', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    print('Learning finished.')
    # sess.run() == tensor.eval()

    for i in range(10):
        print("Accuracy : ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
        r = random.randint(0, mnist.test.num_examples - 1)
        print("Label : ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
        print("Prediction : ", sess.run(tf.argmax(hypothesis, 1),
                                        feed_dict={X: mnist.test.images[r:r+1]}))

        plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()