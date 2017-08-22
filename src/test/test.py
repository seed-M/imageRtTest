import time
import tensorflow as tf


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

def ls(x):
    return tf.add(x,1)
def tp(x):
    return x*2

# main
pathList = ['test.tfrecords']
batch_size = 7
num_epochs = 10

v,x=inputs(pathList,batch_size=batch_size,num_epochs=num_epochs)
ls_op =ls(x)
tp_op =tp(ls_op)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables (the trained variables and the
# epoch counter).
sess.run(init_op)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
step = 0
try:
    while not coord.should_stop():
        start_time = time.time()
        vt= sess.run(tp_op)
        vl=sess.run(ls_op)
        print('step={0}'.format(step),end=' ')
        print(vt,end=',')
        print(vl,end=',')
        print(vl.shape)
        step += 1
except tf.errors.OutOfRangeError:
    print('Done training for %d epochs, %d steps.' % (num_epochs, step))
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
