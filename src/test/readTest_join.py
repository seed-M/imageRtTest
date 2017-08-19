import os
import tensorflow as tf

mdir = '/home/wm/tmp/data/dogvscat'
# train.txt
# val.txt
# test.txt
file = 'train'
fnames = ['train.cat0.tfrecords', 'train.cat1.tfrecords', 'train.cat2.tfrecords', 'train.cat3.tfrecords',
          'train.cat4.tfrecords', 'train.cat5.tfrecords', 'train.cat6.tfrecords', 'train.cat7.tfrecords',
          'train.cat8.tfrecords',
          'train.dog0.tfrecords', 'train.dog1.tfrecords', 'train.dog2.tfrecords', 'train.dog3.tfrecords',
          'train.dog4.tfrecords', 'train.dog5.tfrecords', 'train.dog6.tfrecords', 'train.dog7.tfrecords',
          'train.dog8.tfrecords']


def read_file(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [227, 227, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label


def input_pipeline(filenames, batch_size, read_threads, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [read_file(filename_queue) for _ in range(read_threads)]
    min_after_dequeue = 2000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch
