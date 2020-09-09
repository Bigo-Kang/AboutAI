from random import shuffle
import glob
import cv2
import numpy as np
import tensorflow as tf
import sys
from hyperparameters import *

img_file_path = '../animal_images/*/*.*'

# read addresses and labels from the 'train' folder
addrs = glob.glob(img_file_path)
print(len(addrs))
labels = []

for addr in addrs:
    if 'cat' in addr:
        label = 0
    elif 'cow' in addr:
        label = 1
    elif 'dog' in addr:
        label = 2
    elif 'pig' in addr:
        label = 3
    elif 'sheep' in addr:
        label = 4
    else:
        print('Something wrong')
    labels.append(label)


# to shuffle data
shuffle_data = True  # shuffle the addresses before saving
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6 * len(addrs))]
train_labels = labels[0:int(0.6 * len(labels))]

val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
val_labels = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]

test_addrs = addrs[int(0.8 * len(addrs)):]
test_labels = labels[int(0.8 * len(labels)):]


def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (ORI_IMG_HEIGHT, ORI_IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_and_save(feature_addrs, labels, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(feature_addrs)):
        # print how many images are saved every 1000 images
        if not i % 10:
            print('Data: {}/{}'.format(i, len(feature_addrs)))
            sys.stdout.flush()

        # Load the image
        img = load_image(feature_addrs[i])
        label = labels[i]

        # Create a feature
        feature = {'label': _int64_feature(label),
                   'image': _bytes_feature(img.tostring())}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


convert_and_save(train_addrs, train_labels, 'train.tfrecords')
convert_and_save(val_addrs, val_labels, 'validation.tfrecords')
convert_and_save(test_addrs, test_labels, 'test.tfrecords')
