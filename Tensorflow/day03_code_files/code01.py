import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
rank_three_tensor = tf.ones([3, 4, 5])
my_image = tf.zeros([10, 299, 299, 3])
r = tf.random_uniform(shape=[10,5], minval=-0.1, maxval=0.1, dtype=tf.float32)
integer_seq_tsr = tf.range(start=6, limit=15, delta=3)   # [6, 9, 12]
linear_tsr = tf.linspace(start=10.0, stop=12.0, num=3)   # [10.0, 11.0, 12.0]
filled_tsr = tf.fill([2, 3], 9)

rank_three_tensor = tf.ones([3, 4, 5])
print(rank_three_tensor.shape)		# [3, 4, 5]
matrix = tf.reshape(rank_three_tensor, [6, 10])     # Reshape existing content into a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])               # Reshape existing content into a 3x20
matrixAlt = tf.reshape(matrixB, [4, 3, -1])         # Reshape existing content into a 4x3x5 tensor

print(matrixAlt.shape)
zeros = tf.zeros(matrixAlt.shape[1])
print(zeros)
rank_one_tensor = tf.ones([3])
print(rank_one_tensor)

# Cast a constant integer tensor into floating point.
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
print(float_tensor.dtype)
