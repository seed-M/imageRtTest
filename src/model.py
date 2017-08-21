import tensorflow as tf
from alexnet import AlexNet

def inference(images,keep_prob,num_classes,skip_layers):
    net=AlexNet(images,keep_prob=keep_prob,num_classes=num_classes,skip_layer=skip_layers)
    return net.laten,net.fc8

def loss(logits, labels):
