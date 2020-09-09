import numpy as np
import os
import cv2

IMG_HEIGHT = 60
IMG_WIDTH = 60
NUM_CHANNEL = 3
NUM_CLASS = 5


def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - np.mean(img)) / np.std(img)
    return img


IMAGE_DIR_BASE = '../animal_images'
image_dir_list = os.listdir(IMAGE_DIR_BASE)
print(image_dir_list)

class_index = 0

features = []
labels = []

for cls_index, dir_name in enumerate(image_dir_list):
    image_list = os.listdir(IMAGE_DIR_BASE + os.sep + dir_name)
    for file_name in image_list:
        image = load_image(IMAGE_DIR_BASE + os.sep + dir_name + os.sep + file_name)
        features.append(image.ravel())
        labels.append(cls_index)


print(len(features))   # python list

from random import shuffle
shuffle_data = True
if shuffle_data:
    c = list(zip(features, labels))
    shuffle(c)
    features, labels = zip(*c)

# print(features)
# print(labels)
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
            images_batch = shuf_features[batch_idx:batch_idx+batch_size]
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch


iter_ = train_data_iterator()
for step in range(100):

    # get a batch of data
    images_batch_val, labels_batch_val = next(iter_)

    print(images_batch_val.shape)
    print(labels_batch_val.shape)