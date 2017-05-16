# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from scipy.io import loadmat

class VGG_Model(object):
    def __init__(self):
        self.param_path    = os.path.join(os.getcwd(), "model", "vgg-face.mat")
        self.data          = loadmat(self.param_path)
        self.meta          = self.data['meta']
        self.classes       = self.meta['classes']
        self.class_names   = self.classes[0][0]['description'][0][0]
        self.normalization = self.meta['normalization']
        self.layers        = self.data['layers']
        self.average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
        self.image_size    = np.squeeze(normalization[0][0]['imageSize'][0][0])

        self.used = False

    def vgg(self, input_maps, reuse=False):
        with tf.variable_scope("vgg"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            input_maps = input_maps - tf.constant(self.average_image)
            input_maps = tf.image.resize_images(input_maps, size=[self.image_size[0], self.image_size[1]])

            # read layer info
            current = input_maps
            network = {}
            for layer in self.layers[0]:
                name = self.layer[0]['name'][0][0]
                layer_type = self.layer[0]['type'][0][0]
                if layer_type == 'conv':
                    if name[:2] == 'fc':
                        padding = 'VALID'
                    else:
                        padding = 'SAME'
                    stride = self.layer[0]['stride'][0][0]
                    kernel, bias = self.layer[0]['weights'][0][0]
                    bias   = np.squeeze(bias).reshape(-1)
                    kernel = tf.constant(kernel)
                    bias   = tf.constant(bias)
                    kernel = tf.get_variable(name+"_W", initializer=kernel)
                    bias   = tf.get_variable(name+"_b", initializer=bias)
                    conv   = tf.nn.conv2d(current, kernel,
                                        strides=(1, stride[0], stride[0], 1), padding=padding)
                    current = tf.nn.bias_add(conv, bias)
                    print(name, 'stride:', stride, 'kernel size:', tf.shape(kernel))
                elif layer_type == 'relu':
                    current = tf.nn.relu(current)
                    print(name)
                elif layer_type == 'pool':
                    stride = self.layer[0]['stride'][0][0]
                    pool = self.layer[0]['pool'][0][0]
                    current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                             strides=(1, stride[0], stride[0], 1), padding='SAME')
                    print(name, 'stride:', stride)
                elif layer_type == 'softmax':
                    current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))
                    print(name)

                network[name] = current

        return network["conv4_3"], network["conv5_3"]

    def vgg_loss(self, a, b):
        if self.used == False
            conv4_a, conv5_a = self.vgg(a, reuse=False)
            self.used = True
        else:
            conv4_a, conv5_a = self.vgg(a, reuse=True)

        conv4_b, conv5_b = self.vgg(b, reuse=True)

        return tf.reduce_mean(tf.abs(conv4_a - conv4_b)) + \
               tf.reduce_mean(tf.abs(conv5_a - conv5_b))
