import numpy as np
import os
import cv2

IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CHANNEL = 3
NUM_CLASS = 5


def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


IMAGE_DIR_BASE = '../animal_images'
image_dir_list = os.listdir(IMAGE_DIR_BASE)

features = []
labels = []

for class_index, dir_name in enumerate(image_dir_list):
    image_list = os.listdir(IMAGE_DIR_BASE + os.sep + dir_name)
    # print(class_index)
    # print(dir_name)
    for file_name in image_list:
        image = load_image(IMAGE_DIR_BASE + os.sep + dir_name + os.sep + file_name)
        features.append(image.ravel())
        labels.append(class_index)



# print(len(features))   # python list

from random import shuffle
shuffle_data = True
if shuffle_data:
    c = list(zip(features, labels))
    shuffle(c)
    features, labels = zip(*c)

features = np.array(features)
labels = np.array(labels)

# print(len(labels))  # np array
# print(labels)
# print(labels.shape)
#
# image = features[0]
# image = image.reshape((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL))
# image = image.astype(np.uint8)

# cv2.imshow('Restored Image', image)
# cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()

train_features = features[0:int(0.8 * len(features))]
train_labels = labels[0:int(0.8 * len(labels))]

# val_features = features[int(0.6 * len(features)):int(0.8 * len(features))]
# val_labels = labels[int(0.6 * len(features)):int(0.8 * len(features))]

test_features = features[int(0.8 * len(features)):]
test_labels = labels[int(0.8 * len(labels)):]


BATCH_SIZE = 50


def train_data_iterator():
    batch_idx = 0
    while True:
        idxs = np.arange(0, len(train_features))
        np.random.shuffle(idxs)
        shuf_features = train_features[idxs]
        shuf_labels = train_labels[idxs]
        batch_size = BATCH_SIZE

        for batch_idx in range(0, len(train_features), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx+batch_size] / 255.
            # images_batch = images_batch.astype("float32")
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch


def data_iterator(features, labels, batch_size, shuffle_set):
    """ A simple data iterator """
    batch_idx = 0
    while True:
        if shuffle_set:
            # shuffle labels and features
            idxs = np.arange(0, len(features), dtype=np.int32)
            np.random.shuffle(idxs)
            # print(idxs)
            labels = labels[idxs]
            features = features[idxs]

        for batch_idx in range(0, len(features), batch_size):
            images_batch = features[batch_idx:batch_idx + batch_size] / 255.
            # images_batch = images_batch.astype("float32")
            labels_batch = labels[batch_idx:batch_idx + batch_size]
            # print('batch index is: {}'.format(batch_idx))
            yield images_batch, labels_batch


import tensorflow as tf

images_batch = tf.placeholder(dtype=tf.float32,
                              shape=[None, IMG_HEIGHT*IMG_WIDTH*NUM_CHANNEL])
labels_batch = tf.placeholder(dtype=tf.int32, shape=[None, ])

w1 = tf.get_variable("w1", [IMG_HEIGHT*IMG_WIDTH*NUM_CHANNEL, 1024])
b1 = tf.get_variable("b1", [1024])

fc1 = tf.nn.relu(tf.matmul(images_batch, w1) + b1)

w2 = tf.get_variable("w2", [1024, 512])
b2 = tf.get_variable("b2", [512])

fc2 = tf.nn.relu(tf.matmul(fc1, w2) + b2)

w3 = tf.get_variable("w3", [512, NUM_CLASS])
b3 = tf.get_variable("b3", [NUM_CLASS])
y_pred = tf.matmul(fc2, w3) + b3

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)

loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

iter_ = train_data_iterator()
for step in range(100):
    # get a batch of data
    images_batch_val, labels_batch_val = next(iter_)
    # print(images_batch_val.shape)

    _, loss_val = sess.run([rain_op, loss_mean], feed_dict={
                    images_batch:images_batch_val,
                    labels_batch:labels_batch_val
                    })
    print('LOSS = {}'.format(loss_val))

print('Training Finished....')
print('Test begins….')
TEST_BSIZE = 50
for i in range(int(len(test_features)/TEST_BSIZE)):
    images_batch_val = test_features[i*TEST_BSIZE:(i+1)*TEST_BSIZE]/255.
    labels_batch_val = test_labels[i*TEST_BSIZE:(i+1)*TEST_BSIZE]

    loss_val = sess.run(loss_mean, feed_dict={
                        images_batch:images_batch_val,
                        labels_batch:labels_batch_val
                        })
    print('LOSS = {}'.format(loss_val))

