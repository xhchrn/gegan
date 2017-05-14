# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import os
# import random
# from .utils import pad_seq, bytes_to_file, \
#     read_split_image, shift_and_resize_image, normalize_image

def get_train_dataloader(batch_size):
    image_list, label_list = get_image_label_list()
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

    input_queue = tf.train.slice_input_producer([images, labels], shuffle=True)
    image, label = read_image_label_from_disk(input_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    batch = tf.train.shuffle_batch([image, label],
                                   batch_size=batch_size,
                                   num_threads=4,
                                   capacity=capacity,
                                   min_after_dequeue=min_after_dequeue,
                                   name="TrainData")

    return batch

def read_image_label_from_disk(input_queue):
    label = input_queue[1]

    raw_image = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(raw_image, channels=3)

    image.set_shape([64, 64, 3])
    tf.to_float(image)

    return image, label

def get_image_label_list():
    ng_path    = "/home/jcm/thesis/celebA/0/"
    gl_path    = "/home/jcm/thesis/celebA/1/"

    ng_list    = [os.path.join(ng_path, filename) for filename in os.listdir(ng_path)]
    gl_list    = [os.path.join(gl_path, filename) for filename in os.listdir(ng_path)]

    image_list   = ng_list + gl_list
    label_list = [0] * len(ng_list) + [1] * len(gl_list)

    return image_list, label_list

# class PickledImageProvider(object):
#     def __init__(self, obj_path):
#         self.obj_path = obj_path
#         self.examples = self.load_pickled_examples()
# 
#     def load_pickled_examples(self):
#         with open(self.obj_path, "rb") as of:
#             examples = list()
#             while True:
#                 try:
#                     e = pickle.load(of)
#                     examples.append(e)
#                     if len(examples) % 1000 == 0:
#                         print("processed %d examples" % len(examples))
#                 except EOFError:
#                     break
#                 except Exception:
#                     pass
#             print("unpickled total %d examples" % len(examples))
#             return examples
# 
# 
# def get_batch_iter(examples, batch_size, augment):
#     # the transpose ops requires deterministic
#     # batch size, thus comes the padding
#     padded = pad_seq(examples, batch_size)
# 
#     def process(img):
#         img = bytes_to_file(img)
#         try:
#             img_A, img_B = read_split_image(img)
#             if augment:
#                 # augment the image by:
#                 # 1) enlarge the image
#                 # 2) random crop the image back to its original size
#                 # NOTE: image A and B needs to be in sync as how much
#                 # to be shifted
#                 w, h, _ = img_A.shape
#                 multiplier = random.uniform(1.00, 1.20)
#                 # add an eps to prevent cropping issue
#                 nw = int(multiplier * w) + 1
#                 nh = int(multiplier * h) + 1
#                 shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
#                 shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
#                 img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh)
#                 img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh)
#             img_A = normalize_image(img_A)
#             img_B = normalize_image(img_B)
#             return np.concatenate([img_A, img_B], axis=2)
#         finally:
#             img.close()
# 
#     def batch_iter():
#         for i in range(0, len(padded), batch_size):
#             batch = padded[i: i + batch_size]
#             labels = [e[0] for e in batch]
#             processed = [process(e[1]) for e in batch]
#             # stack into tensor
#             yield labels, np.array(processed).astype(np.float32)
# 
#     return batch_iter()
# 
# 
# class TrainDataProvider(object):
#     def __init__(self, data_dir, train_name="train.obj", val_name="val.obj", filter_by=None):
#         self.data_dir   = data_dir
#         self.filter_by  = filter_by
#         self.train_path = os.path.join(self.data_dir, train_name)
#         self.val_path   = os.path.join(self.data_dir, val_name)
#         self.train      = PickledImageProvider(self.train_path)
#         self.val        = PickledImageProvider(self.val_path)
#         if self.filter_by:
#             print("filter by label ->", filter_by)
#             self.train.examples = filter(lambda e: e[0] in self.filter_by, self.train.examples)
#             self.val.examples   = filter(lambda e: e[0] in self.filter_by, self.val.examples)
#         print("train examples -> %d, val examples -> %d" % (len(self.train.examples), len(self.val.examples)))
# 
#     def get_train_iter(self, batch_size, shuffle=True):
#         training_examples = self.train.examples[:]
#         if shuffle:
#             np.random.shuffle(training_examples)
#         return get_batch_iter(training_examples, batch_size, augment=True)
# 
#     def get_val_iter(self, batch_size, shuffle=True):
#         """
#         Validation iterator runs forever
#         """
#         val_examples = self.val.examples[:]
#         if shuffle:
#             np.random.shuffle(val_examples)
#         while True:
#             val_batch_iter = get_batch_iter(val_examples, batch_size, augment=False)
#             for labels, examples in val_batch_iter:
#                 yield labels, examples
# 
#     def compute_total_batch_num(self, batch_size):
#         """Total padded batch num"""
#         return int(np.ceil(len(self.train.examples) / float(batch_size)))
# 
#     def get_all_labels(self):
#         """Get all training labels"""
#         return list({e[0] for e in self.train.examples})
# 
#     def get_train_val_path(self):
#         return self.train_path, self.val_path
# 
# 
# class InjectDataProvider(object):
#     def __init__(self, obj_path):
#         self.data = PickledImageProvider(obj_path)
#         print("examples -> %d" % len(self.data.examples))
# 
#     def get_single_embedding_iter(self, batch_size, embedding_id):
#         examples = self.data.examples[:]
#         batch_iter = get_batch_iter(examples, batch_size, augment=False)
#         for _, images in batch_iter:
#             # inject specific embedding style here
#             labels = [embedding_id] * batch_size
#             yield labels, images
# 
#     def get_random_embedding_iter(self, batch_size, embedding_ids):
#         examples = self.data.examples[:]
#         batch_iter = get_batch_iter(examples, batch_size, augment=False)
#         for _, images in batch_iter:
#             # inject specific embedding style here
#             labels = [random.choice(embedding_ids) for i in range(batch_size)]
#             yield labels, images
# 
# 
# class NeverEndingLoopingProvider(InjectDataProvider):
#     def __init__(self, obj_path):
#         super(NeverEndingLoopingProvider, self).__init__(obj_path)
# 
#     def get_random_embedding_iter(self, batch_size, embedding_ids):
#         while True:
#             # np.random.shuffle(self.data.examples)
#             rand_iter = super(NeverEndingLoopingProvider, self) \
#                 .get_random_embedding_iter(batch_size, embedding_ids)
#             for labels, images in rand_iter:
#                 yield labels, images

