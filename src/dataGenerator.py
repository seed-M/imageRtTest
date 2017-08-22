import tensorflow as tf


def read_and_decode(filename_queue, shape):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, shape)
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label


def getInputs(filepathList, batch_size, num_epochs,shape):
    if not num_epochs: num_epochs = None
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filepathList, num_epochs=num_epochs)
        image, label = read_and_decode(filename_queue,shape=shape)
        min_after_dequeue=3000
        capacity=min_after_dequeue+3*batch_size
        img_batch, label_batch = tf.train.shuffle_batch([image,label],batch_size=batch_size,num_threads=2,
                capacity=capacity,min_after_dequeue=min_after_dequeue)
        return img_batch,label_batch