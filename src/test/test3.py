import tensorflow as tf


class tclass(object):
    def __init__(self,path):
        self.path=path
        self.create()


    def create(self):
        with tf.variable_scope('scope1') as scope:
            print()