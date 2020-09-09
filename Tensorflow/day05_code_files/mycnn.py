import tensorflow as tf
from hyperparameters import *
import os


class MyNet:
    def __init__(self, images_batch, labels_batch, keep_prob, num_class):
        self.build(images_batch, labels_batch, keep_prob, num_class)
        self.keep_prob = keep_prob

    def con_layer(self, input_tensor, filter_size, in_channels, out_channels, layer_name):
        with tf.variable_scope(layer_name):
            filt = tf.get_variable(name="filter", shape=[filter_size, filter_size, in_channels, out_channels], dtype=tf.float32)
            bias = tf.get_variable(name='bias', shape=[out_channels], dtype=tf.float32)

            pre_activate = tf.nn.conv2d(input_tensor, filt, [1, 1, 1, 1], padding='SAME') + bias
            activations = tf.nn.relu(pre_activate)
            return activations

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def dense_layer(self, input_tensor, input_dim, output_dim, layer_name, act=True):
        with tf.variable_scope(layer_name):
            weights = tf.get_variable(name="weight", shape=[input_dim, output_dim])
            biases = tf.get_variable(name='bias', shape=[output_dim])
            preactivate = tf.matmul(input_tensor, weights) + biases
            if act:
                return tf.nn.relu(preactivate)
            return preactivate

    def build(self, images_batch, labels_batch, keep_prob, num_class):
        input_tensor_shape = images_batch.get_shape()[1:]
        # print(input_tensor_shape)
        conv1 = self.con_layer(images_batch, 5, input_tensor_shape[2], 32, 'con_layer1')
        h_pool1 = self.max_pool_2x2(conv1)

        conv2 = self.con_layer(h_pool1, 5, 32, 64, 'con_layer2')
        h_pool2 = self.max_pool_2x2(conv2)

        fc_size = input_tensor_shape[0]//4*input_tensor_shape[1]//4*64
        h_pool2_flat = tf.reshape(h_pool2, [-1, fc_size])

        fc1 = self.dense_layer(h_pool2_flat, fc_size, 1024, 'dense1')
        h_fc1_drop = tf.nn.dropout(fc1, keep_prob)

        y_pred = self.dense_layer(h_fc1_drop, 1024, num_class, 'dense2', act=False)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch))
        tf.summary.scalar('loss', self.loss)

        self.prediction = tf.argmax(y_pred, 1, output_type=tf.int32, name='prediction')

        correct_prediction = tf.equal(self.prediction,  labels_batch)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.summary = tf.summary.merge_all()
