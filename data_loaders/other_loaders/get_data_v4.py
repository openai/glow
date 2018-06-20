# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
from imagenet_new.vgg_preprocessing_v4 import pillow_preprocess_image, downsample_images
from PIL import Image
import io

_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_NUM_PARALLEL_FILE_READERS = 2
_SHUFFLE_BUFFER = 1500
_NUM_PARALLEL_MAP_CALLS = 1


###############################################################################
# Data processing
###############################################################################
def parse_fn(example):
    """Parses an Example proto containing a training example of an image.
    The dataset contains serialized Example protocol buffers.
    The Example proto is expected to contain features named
    image/encoded (a JPEG-encoded string) and image/class/label (int)
    Args:
      example: scalar Tensor tf.string containing a serialized example protocol buffer.
      is_training: A boolean denoting whether the input is for training.
    Returns:
    Tuple
    with processed image tensor and one-hot-encoded label tensor.
    """
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1)
    }

    features = tf.parse_single_example(example, feature_map)
    image, label = features['image/encoded'], features['image/class/label']

    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    # Results in a 3-D int8 Tensor. This will be converted to a float later,
    # during resizing.
    #image = tf.image.decode_jpeg(image, channels=_NUM_CHANNELS)

    #image = preprocess_image(image, max_res, is_training)

    label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32) - 1

    #label = tf.one_hot(label, _NUM_CLASSES)
    return image, label

def pillow_preprocess(image, label, max_res, is_training):
    image = Image.open(io.BytesIO(image))
    image = image.convert('RGB')
    image = pillow_preprocess_image(image, max_res, is_training)
    return image, label

def input_fn(is_training, data_dir, batch_size, max_res, resolution, full_res = None):
    """Input function which provides batches for train or eval.
    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
      num_parallel_calls: The number of records that are processed in parallel.
        This can be optimized per data set but for generally homogeneous data
        sets, should be approximately the number of available CPU cores.
      multi_gpu: Whether this is run multi-GPU. Note that this is only required
        currently to handle the batch leftovers, and can be removed
        when that is handled directly by Estimator.
    Returns:
      An iterator that can be used for iteration.
    """
    if is_training:
        files = tf.data.Dataset.list_files(os.path.join(data_dir, 'train-*-of-01024'))
    else:
        files = tf.data.Dataset.list_files(os.path.join(data_dir, 'validation-*-of-00128'))

    if is_training:
        # Shuffle the input files
        files = files.shuffle(buffer_size=_NUM_TRAIN_FILES)

    # Convert to individual records
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=_NUM_PARALLEL_FILE_READERS))

    # We prefetch a batch at a time, This can help smooth out the time taken to
    # load input files as we go through shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffle the records. Note that we shuffle before repeating to ensure
        # that the shuffling respects epoch boundaries.
        dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

    dataset = dataset.repeat()

    # Parse the raw records into images and labels
    dataset = dataset.map(parse_fn, num_parallel_calls=_NUM_PARALLEL_MAP_CALLS)

    pillow_preprocess_func = lambda x,y: pillow_preprocess(x,y, max_res, is_training)
    dataset = dataset.map(lambda image, label: tuple(tf.py_func(
        pillow_preprocess_func, [image, label], [tf.uint8, label.dtype])))
    dataset = dataset.batch(batch_size)
    shape = [batch_size, max_res, max_res, _NUM_CHANNELS]
    dataset = dataset.map(lambda images, labels: downsample_images(images, labels, shape, max_res // resolution, full_res=full_res))

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path.
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()

    return iterator

def get_imagenet_data(sess, data_dir, batch_size, max_res, resolution, batch_init = None):
    #with tf.device('/cpu:0'):
    train_iterator = input_fn(True, data_dir, batch_size, max_res, resolution)
    valid_iterator = input_fn(False, data_dir, batch_size, max_res, resolution)
    full_res_iterator = input_fn(False, data_dir, batch_size, max_res, resolution, True)

    if batch_init is not None:
        data_init = make_batch(sess, train_iterator, batch_size, batch_init)
        full_res_init = make_batch(sess, full_res_iterator, batch_size, batch_init)

    return train_iterator, valid_iterator, data_init, full_res_init

def make_batch(sess, iterator, iterator_batch_size, required_batch_size):
    ib, rb = iterator_batch_size, required_batch_size
    assert rb % ib == 0
    k = rb // ib
    xs, x_ds, ys  = [], [], []
    data = iterator.get_next()
    for i in range(k):
        x, x_d, y  = sess.run(data)
        xs.append(x)
        x_ds.append(x_d)
        ys.append(y)
    x, x_d, y  = np.concatenate(xs), np.concatenate(x_ds), np.concatenate(ys)
    return {'x': x, 'x_d': x_d, 'y': y}

