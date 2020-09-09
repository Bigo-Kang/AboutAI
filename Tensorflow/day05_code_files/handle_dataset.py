
import numpy as np
import os
import cv2
import tensorflow as tf
from hyperparameters import *


def _distorted_parse_function(example_proto):
    features = {'label': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)

    # now return the converted data
    label = tf.cast(parsed_features['label'], tf.int32)
    image = tf.decode_raw(parsed_features['image'], tf.float32)
    image = tf.reshape(image, [ORI_IMG_HEIGHT, ORI_IMG_WIDTH, NUM_CHANNEL])
    image = tf.random_crop(image, [160, 160, NUM_CHANNEL])
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.image.per_image_standardization(image)

    return image, label


def _parse_function(example_proto):
    features = {'label': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)

    # now return the converted data
    label = tf.cast(parsed_features['label'], tf.int32)
    # image = features['train/image']
    image = tf.decode_raw(parsed_features['image'], tf.float32)
    image = tf.reshape(image, [ORI_IMG_HEIGHT, ORI_IMG_WIDTH, NUM_CHANNEL])
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.image.per_image_standardization(image)
    return image, label


def get_input(train_filenames, valid_filenames, test_filenames):

    dataset_train = tf.data.TFRecordDataset(train_filenames)
    dataset_train = dataset_train.map(_distorted_parse_function)
    dataset_train = dataset_train.shuffle(buffer_size=10000)
    dataset_train = dataset_train.batch(50)
    dataset_train = dataset_train.repeat()

    dataset_valid = tf.data.TFRecordDataset(valid_filenames)
    dataset_valid = dataset_valid.map(_parse_function)
    dataset_valid = dataset_valid.shuffle(buffer_size=10000)
    dataset_valid = dataset_valid.batch(50)
    dataset_valid = dataset_valid.repeat()

    dataset_test = tf.data.TFRecordDataset(test_filenames)
    dataset_test = dataset_test.map(_parse_function)
    dataset_test = dataset_test.shuffle(buffer_size=10000)
    dataset_test = dataset_test.batch(50)
    dataset_test = dataset_test.repeat()

    iterator = tf.contrib.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    x_image, labels_batch = iterator.get_next()

    training_init_op = iterator.make_initializer(dataset_train)
    valid_init_op = iterator.make_initializer(dataset_valid)
    test_init_op = iterator.make_initializer(dataset_test)

    # validation_init_op = iterator.make_initializer(validation_dataset)
    return x_image, labels_batch, training_init_op, valid_init_op, test_init_op
