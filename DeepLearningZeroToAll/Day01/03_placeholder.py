import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
#  + provides a shortcut for tf.add(a, b)
adder = a+b
sess = tf.Session()
print(sess.run(adder, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder, feed_dict={a: [1, 3], b: [2, 4]}))
sess.close()