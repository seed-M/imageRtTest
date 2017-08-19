import os
import tensorflow as tf

mdir = '/home/wm/tmp/data'
# train.txt
# val.txt
# test.txt
file='train'
fnames=['train.cat0.tfrecords','train.cat1.tfrecords','train.cat2.tfrecords','train.cat3.tfrecords', \
        'train.cat4.tfrecords','train.cat5.tfrecords','train.cat6.tfrecords','train.cat7.tfrecords', \
        'train.cat8.tfrecords',\
        'train.dog0.tfrecords','train.dog1.tfrecords','train.dog2.tfrecords','train.dog3.tfrecords', \
        'train.dog4.tfrecords', 'train.dog5.tfrecords', 'train.dog6.tfrecords','train.dog7.tfrecords', \
        'train.dog8.tfrecords']



def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer(filename,shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [227, 227, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

img, label = read_and_decode([os.path.join(mdir,name) for name in fnames])

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],num_threads=2,
                                                batch_size=200, capacity=5000,
                                                min_after_dequeue=3000)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(300):
        val, l= sess.run([img_batch, label_batch])
        #我们也可以根据需要对val， l进行处理
        #l = to_categorical(l, 12)
        avg=sum(l)
        print(avg/len(l))
    sess.close()