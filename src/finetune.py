import tensorflow as tf
import time
import argparse
import sys
import os
from datetime import datetime
from model import FinetuneModel as fm
from dataGenerator import getInputs

# Basic model parameters as external flags.
FLAGS = None

def run_training(tf_list,skip_layer):
    with tf.Graph().as_default():
        images,labels=getInputs(tf_list,batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs,
                                shape=[227,227,3])
        fmodel=fm(x=images,keep_prob=FLAGS.keep_prob,num_classes=FLAGS.num_cls,skip_layer=skip_layer,weights_path=FLAGS.w_path)
        score,_=fmodel.inference()
        loss=fmodel.loss(labels=labels)
        tf.summary.scalar('loss', loss)
        train_op=fmodel.training(learning_rate=FLAGS.l_rate)
        accuracy=fmodel.accuracy(labels=labels)
        tf.summary.scalar('acc',accuracy)
        merged_summary = tf.summary.merge_all()
        init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

        writer = tf.summary.FileWriter(FLAGS.fw_path)
        sess=tf.Session()
        sess.run(init_op)
        fmodel.load_initial_weights(sess=sess)
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        step = 0
        try:
            while not coord.should_stop():
                start_time = time.time()
                if step%20==0:
                    s,_=sess.run([merged_summary,train_op])
                    writer.add_summary(s, step)
                    print("{} batch number: {}".format(datetime.now(), step))
                else:
                    sess.run(train_op)
                step+=1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            # Wait for threads to finish.
        coord.join(threads)
        sess.close()



def tflist(path):
    tf_list=[]
    for parent, _, filenames in os.walk(path):
        for name in filenames:
            if os.path.splitext(name)[1] != '.tfrecords':
                continue
            tf_list.append(os.path.join(parent,name))

    return tf_list

def main(_):
    skip_layer=['fc8','laten','fc7','fc6']
    tf_list=tflist(FLAGS.tf_path)
    run_training(tf_list, skip_layer)
    # try:
    #     tfrecords_idx=open(FLAGS.tfrecords_idx,'r')
    #     lines=tfrecords_idx.readlines()
    #     tf_list=[]
    #     for line in lines:
    #         tf_list.append(line.strip())
    #     run_training(tf_list,skip_layer)
    #
    # except IOError as err:
    #     print("FileError:"+str(err))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'num_cls',
      type=int,
      default=2,
      help='Number of classes.'
  )
  parser.add_argument(
      'tf_path',
      type=str,
      default='/tmp/data/tf.idx',
      help='Directory with the tfrecords file dir.'
  )
  parser.add_argument(
      'w_path',
      type=str,
      default='/tmp/data/bvlc_alexnet.npy',
      help='caffe bvlc_alexnet.npy path'
  )
  parser.add_argument(
      '--l_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=10,
      help='Number of epochs to run trainer.'
  )
  parser.add_argument(
      '--keep_prob',
      type=float,
      default=0.5,
      help='Number of dropout rate.'
  )

  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.'
  )
  parser.add_argument(
      '--fw_path',
      type=str,
      default='/tmp/data/tensorboard',
      help='path to tensorboard savedir'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)