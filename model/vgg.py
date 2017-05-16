# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from scipy.io import loadmat

class VGG_Model(object):
    def __init__(self):
        self.param_path = os.path.join(os.getcwd(), "model", "vgg-face.mat")
        self.build_model()

    def build_model(self):
        with tf.variable_scope("vgg"):
            data = loadmat(self.param_path)

            self.input_maps = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name="input_maps")

            # read meta info
            meta          = data['meta']
            classes       = meta['classes']
            class_names   = classes[0][0]['description'][0][0]
            normalization = meta['normalization']
            average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
            image_size    = np.squeeze(normalization[0][0]['imageSize'][0][0])
            input_maps    = tf.image.resize_images(self.input_maps, size=[image_size[0], image_size[1]])

            # read layer info
            layers = data['layers']
            current = input_maps
            network = {}
            for layer in layers[0]:
                name = layer[0]['name'][0][0]
                layer_type = layer[0]['type'][0][0]
                if layer_type == 'conv':
                    if name[:2] == 'fc':
                        padding = 'VALID'
                    else:
                        padding = 'SAME'
                    stride = layer[0]['stride'][0][0]
                    kernel, bias = layer[0]['weights'][0][0]
                    # kernel = np.transpose(kernel, (1, 0, 2, 3))
                    bias = np.squeeze(bias).reshape(-1)
                    conv = tf.nn.conv2d(current, tf.constant(kernel),
                                        strides=(1, stride[0], stride[0], 1), padding=padding)
                    current = tf.nn.bias_add(conv, bias)
                    print(name, 'stride:', stride, 'kernel size:', np.shape(kernel))
                elif layer_type == 'relu':
                    current = tf.nn.relu(current)
                    print(name)
                elif layer_type == 'pool':
                    stride = layer[0]['stride'][0][0]
                    pool = layer[0]['pool'][0][0]
                    current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                             strides=(1, stride[0], stride[0], 1), padding='SAME')
                    print(name, 'stride:', stride)
                elif layer_type == 'softmax':
                    current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))
                    print(name)

                network[name] = current

        setattr(self, "network", network)
        setattr(self, "average_image", average_image)
        setattr(self, "conv4", network["conv4_3"])
        setattr(self, "conv5", network["conv5_3"])

    def get_vgg_loss(self, sess, a, b):
        a = a - self.average_image
        b = b - self.average_image

        [conv4_a, conv5_a] = sess.run([self.conv4, self.conv5], feed_dict={self.input_maps: a})
        [conv4_b, conv5_b] = sess.run([self.conv4, self.conv5], feed_dict={self.input_maps: b})

        loss_4 = np.mean(np.abs(conv4_a - conv4_b))
        loss_5 = np.mean(np.abs(conv5_a - conv5_b))
        loss = loss_4 + loss_5

        return loss
