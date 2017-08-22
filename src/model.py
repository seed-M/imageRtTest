import tensorflow as tf
from alexnet import AlexNet

class FinetuneModel(object):

    def __init__(self,x, keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT'):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path


    def inference(self):
        self.net=AlexNet(self.X,keep_prob=self.KEEP_PROB,num_classes=self.NUM_CLASSES,skip_layer=self.SKIP_LAYER)
        self.score=self.net.fc8
        self.bucket=self.net.laten
        return self.score, self.bucket

    def loss(self,labels):
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=self.score,labels=labels)
        self.loss=tf.reduce_mean(cross_entropy)
        return self.loss

    def training(self,learning_rate):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        if self.SKIP_LAYER!=None:
            var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in self.SKIP_LAYER]
            gradients = tf.gradients(self.loss, var_list)
            gradients = list(zip(gradients, var_list))
            train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)
        else:
            train_op=optimizer.minimize(self.loss, global_step=global_step)
        return train_op

    def accuracy(self,labels):
        correct_pred = tf.equal(tf.argmax(self.score, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def load_initial_weights(self,sess):
        self.net.load_initial_weights(session=sess)