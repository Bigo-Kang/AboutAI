import tensorflow as tf

go = tf.constant("Go.")

# seart a TF session
sess = tf.Session()

#run the op and get result
print(sess.run(go))