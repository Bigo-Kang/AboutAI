import tensorflow as tf

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2

sess = tf.Session()
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

v1 = tf.get_variable("v", [1])
v2 = tf.get_variable("v", [1])
assert v1 == v2

def foo():
  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("v", [1])
  return v

v1 = foo()  # Creates v
v2 = foo()  # Gets the same, existing v
assert v1 == v2

print(v1)
print(v2)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

sess = tf.Session()
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

x = tf.constant([[1,2],[3,4]], dtype=tf.float32)
y = tf.constant([[4,3],[3,2]], dtype=tf.float32)

x_add_y = tf.add(x, y) # [[5,5],[6,6]]
x_mul_y = tf.matmul(x, y) # [[10,7],[24,17]]
log_x = tf.log(x) # log_x => [[0,0.6931],[1.0986,1.3863]]

# Performs reduction operation across the specified axis
x_sum_1 = tf.reduce_sum(x, axis=[1], keepdims=False)  	# x_sum_1 => [3,7]
x_sum_2 = tf.reduce_sum(x, axis=[0], keepdims=True) 		# x_sum_2 => [[4],[6]]
