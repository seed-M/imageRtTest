"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf
import dataGenerator as dg

from alexnet import AlexNet
from datetime import datetime
from tensorflow.contrib.data import Iterator

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
mdir = 'D:/tmp/Images/CatVsDog'
trf_names = ['train.cat0.tfrecords', 'train.cat1.tfrecords', 'train.cat2.tfrecords', 'train.cat3.tfrecords',
             'train.cat4.tfrecords', 'train.cat5.tfrecords', 'train.cat6.tfrecords', 'train.cat7.tfrecords',
             'train.cat8.tfrecords',
             'train.dog0.tfrecords', 'train.dog1.tfrecords', 'train.dog2.tfrecords', 'train.dog3.tfrecords',
             'train.dog4.tfrecords', 'train.dog5.tfrecords', 'train.dog6.tfrecords', 'train.dog7.tfrecords',
             'train.dog8.tfrecords']
valf_names = ['val.cat0.tfrecords', 'val.cat1.tfrecords', 'val.dog0.tfrecords', 'val.dog1.tfrecords']
img_shape = [227, 227, 3]

# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'laten', 'fc7']

# How often we want to write the tf.summary data to disk
display_step = 20
saveckpt_step=17500
# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "D:/tensorflow/tmp/finetune_alexnet_laten/tensorboard"
checkpoint_path = "D:/tensorflow/tmp/finetune_alexnet_laten/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu

tr_batch, tr_lable_batch = dg.getInputs([os.path.join(mdir, name) for name in trf_names], batch_size=batch_size,
                                        num_epochs=num_epochs, shape=img_shape)
val_batch, val_lable_batch = dg.getInputs([os.path.join(mdir, name) for name in valf_names], batch_size=batch_size,
                                          num_epochs=1, shape=img_shape)

# tr_data = ImageDataGenerator(train_file,
#                              mode='training',
#                              batch_size=batch_size,
#                              num_classes=num_classes,
#                              shuffle=True)
# val_data = ImageDataGenerator(val_file,
#                               mode='inference',
#                               batch_size=batch_size,
#                               num_classes=num_classes,
#                               shuffle=False)

# create an reinitializable iterator given the dataset structure
# iterator = Iterator.from_structure(tr_data.data.output_types,
#                                    tr_data.data.output_shapes)
# next_batch = iterator.get_next()

# Ops for initializing the two different iterators


# TF placeholder for graph input and output
# x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
# shape=[None]
# shape[len(shape):len(img_shape)]=img_shape
# x = tf.placeholder(tf.float32,shape)
# y = tf.placeholder(tf.float32, [None, num_classes])
# keep_prob = tf.placeholder(tf.float32)

# Initialize model
with tf.variable_scope('model') as scope:
    model= AlexNet(tr_batch, dropout_rate, num_classes, train_layers)
    tr_score = model.fc8
with tf.variable_scope(scope, reuse=True):
    val_score = AlexNet(val_batch, 1, num_classes, train_layers).fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tr_score,
                                                                  labels=tr_lable_batch))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    tr_correct_pred = tf.equal(tf.argmax(tr_score, 1), tf.argmax(tr_lable_batch, 1))
    tr_accuracy = tf.reduce_mean(tf.cast(tr_correct_pred, tf.float32))
    val_correct_pred = tf.equal(tf.argmax(val_score, 1), tf.argmax(val_lable_batch, 1))
    val_accuracy = tf.reduce_mean(tf.cast(val_correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', tr_accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
# Create a session for running operations in the Graph.
sess = tf.Session()
# Initialize the variables (the trained variables and the
# epoch counter).
sess.run(init_op)
writer.add_graph(sess.graph)
# Load the pretrained weights into the non-trainable layer
model.load_initial_weights(sess)
# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))
    step = 0;
    while not coord.should_stop():

        # Add the model graph to TensorBoard

        # Loop over number of epochs
        # for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        # Initialize iterator with the training dataset

        # get next batch of data
        # img_batch, label_batch = sess.run(next_batch)

        # And run the training op
        sess.run([train_op,merged_summary])

        # Generate summary with the current batch of data and write to file
        if step % display_step == 0:
            _,s=sess.run([train_op, merged_summary])
            writer.add_summary(s, step)
        else:
            sess.run(train_op)


        # print("{} batch number: {}".format(datetime.now(), step + 1))


        if step%saveckpt_step==0:
            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.now()))
            sess.run(val_accuracy)
            try:



            test_acc = 0.
            test_count = 0
            for _ in range(val_batches_per_epoch):
                img_batch, label_batch = sess.run(next_batch)
                acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    keep_prob: 1.})
                test_acc += acc
                test_count += 1
            test_acc /= test_count
            print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                           test_acc))
            print("{} Saving checkpoint of model...".format(datetime.now()))

            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path,
                                           'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))
