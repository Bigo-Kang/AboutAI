import tensorflow as tf

# c = tf.constant(4.0)
# print(c.graph)
# assert c.graph is tf.get_default_graph()
#
# a = tf.constant(3.0, dtype=tf.float32)
# b = tf.constant(4.0) # also tf.float32 implicitly
# total = a + b
# print(a)
# print(b)
# print(total)
#
# writer = tf.summary.FileWriter('tmp/logs')
# writer.add_graph(tf.get_default_graph())
# # writer = tf.summary.FileWriter("tmp/logs", tf.get_default_graph())
# writer.flush()

sess = tf.Session()
# print(sess.run(total))

cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
my_op = tf.assign(cool_numbers, cool_numbers+1)


sess.run(tf.global_variables_initializer())

print(sess.run(cool_numbers))
sess.run(my_op)
print(sess.run(cool_numbers))
sess.run(my_op)
print(sess.run(cool_numbers))