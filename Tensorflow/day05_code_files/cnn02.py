import tensorflow as tf
from hyperparameters import *
import mycnn
import handle_dataset
import os

best_val_acc = 0.0
save_path = LOG_DIR + os.sep + 'model.ckpt'


def train():
    step = 0
    for epoch in range(NUM_EPOCH):
        sess.run(training_init_op)
        for _ in range(BATCH_PER_EPOCH):
            _, loss_val, accuracy_val, _summary = sess.run([mynet.train_op, mynet.loss, mynet.accuracy, mynet.summary],
                                                                feed_dict={keep_prob: 0.5})
            step += 1
            summary_writer.add_summary(_summary, step)

        print('Train Loss and Accurary after {}-th epoch: {} {}'.format(epoch, loss_val, accuracy_val))
        validate(epoch)


def validate(epoch):
    global best_val_acc
    sess.run(vaid_init_op)
    sum_accuracy = 0.0
    for _ in range(NUM_VAL_BATCH):
        accuracy_val = sess.run(mynet.accuracy, feed_dict={keep_prob: 1.0})
        sum_accuracy += accuracy_val
    val_accuracy = sum_accuracy / NUM_VAL_BATCH
    print('Validation Accuracy after {}-th epoch is {}'.format(epoch, val_accuracy))
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        saver.save(sess, save_path, global_step=epoch)
        print('Weights are saved to ' + save_path)


def test():
    sess.run(test_init_op)
    sum_accuracy = 0.0
    for _ in range(NUM_TEST_BATCH):
        accuracy_test = sess.run(mynet.accuracy, feed_dict={keep_prob: 1.0})
        sum_accuracy += accuracy_test
    test_accuracy = sum_accuracy / NUM_TEST_BATCH
    print('Test Accuracy  is {}'.format(test_accuracy))


train_filenames = ['train.tfrecords']
valid_filenames = ['validation.tfrecords']
test_filenames = ['test.tfrecords']

images_batch, labels_batch, training_init_op, vaid_init_op, test_init_op = handle_dataset.get_input(train_filenames, valid_filenames, test_filenames)
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

mynet = mycnn.MyNet(images_batch, labels_batch, keep_prob, NUM_CLASS)

saver = tf.train.Saver()
sess = tf.Session()
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
sess.run(tf.global_variables_initializer())

train()
print('Training Finished....')
summary_writer.close()
test()
