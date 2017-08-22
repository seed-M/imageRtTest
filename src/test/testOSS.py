import tensorflow as tf

import argparse
import sys
import os


def read_and_decode(filename_queue):
    # 根据文件名生成一个队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'plus1': tf.FixedLenFeature([], tf.string),
                                           'value': tf.FixedLenFeature([], tf.int64),
                                       })

    plus1 = tf.cast(features['plus1'], tf.string)
    value = tf.cast(features['value'], tf.int32)

    return plus1, value


def inputs(filepathList, batch_size, num_epochs):
    if not num_epochs: num_epochs = None
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filepathList, num_epochs=num_epochs)
        image, label = read_and_decode(filename_queue)
        min_after_dequeue = 20
        capacity = min_after_dequeue + 3 * batch_size
        img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=2,
                                                        capacity=capacity, min_after_dequeue=min_after_dequeue)
        return img_batch, label_batch


def tp(x):
    return  tf.reduce_max(tf.add(x,1))

def tflist(path):
    tf_list=[]
    for parent, _, filenames in os.walk(path):
        for name in filenames:
            if os.path.splitext(name)[1] != '.tfrecords':
                continue
            tf_list.append(os.path.join(parent,name))

    return tf_list

def main(_):

    tf_list=tflist(FLAGS.f_path)
    v,x=inputs(tf_list,batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    sess = tf.Session()
    tp_op=tp(x)
    tf.summary.scalar('tp',tp_op)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.fw_path)
    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    try:
        while not coord.should_stop():
            if step%10==0:
                s,_=sess.run([merged_summary,tp_op])
                writer.add_summary(s,step)
            else:
                sess.run(tp_op)
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'f_path',
        type=str,
        help='tf file dir'
    )
    parser.add_argument(
        'fw_path',
        type=str,
        help='tensorboard file dir'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='batch_size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default='10',
        help='number of epochs'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)