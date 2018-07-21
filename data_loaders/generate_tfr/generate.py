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

"""
Generate CelebA-HQ and Imagenet datasets
For CelebA-HQ, first create original tfrecords file using https://github.com/tkarras/progressive_growing_of_gans/blob/master/dataset_tool.py
For Imagenet, first create original tfrecords file using https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py
Then, use this script to get our tfr file from those records.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from typing import Iterable

_NUM_CHANNELS = 3


_NUM_PARALLEL_FILE_READERS = 32
_NUM_PARALLEL_MAP_CALLS = 32
_DOWNSAMPLING = tf.image.ResizeMethod.BILINEAR
_SHUFFLE_BUFFER = 1024


def _int64_feature(value):
    if not isinstance(value, Iterable):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def error(msg):
    print('Error: ' + msg)
    exit(1)


def x_to_uint8(x):
    return tf.cast(tf.clip_by_value(tf.floor(x), 0, 255), 'uint8')


def centre_crop(img):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    min_side = tf.minimum(h, w)
    h_offset = (h - min_side) // 2
    w_offset = (w - min_side) // 2
    return tf.image.crop_to_bounding_box(img, h_offset, w_offset, min_side, min_side)


def downsample(img):
    return (img[0::2, 0::2, :] + img[0::2, 1::2, :] + img[1::2, 0::2, :] + img[1::2, 1::2, :]) * 0.25


def parse_image(max_res):
    def _process_image(img):
        img = centre_crop(img)
        img = tf.image.resize_images(
            img, [max_res, max_res], method=_DOWNSAMPLING)
        img = tf.cast(img, 'float32')
        resolution_log2 = int(np.log2(max_res))
        q_imgs = []
        for lod in range(resolution_log2 - 1):
            if lod:
                img = downsample(img)
            quant = x_to_uint8(img)
            q_imgs.append(quant)
        return q_imgs

    def _parse_image(example):
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
            'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                    default_value=-1)
        }
        features = tf.parse_single_example(example, feature_map)
        img, label = features['image/encoded'], features['image/class/label']
        label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32) - 1
        img = tf.image.decode_jpeg(img, channels=_NUM_CHANNELS)
        imgs = _process_image(img)
        parsed = (label, *imgs)
        return parsed

    return _parse_image


def parse_celeba_image(max_res, transpose=False):
    def _process_image(img):
        img = tf.cast(img, 'float32')
        resolution_log2 = int(np.log2(max_res))
        q_imgs = []
        for lod in range(resolution_log2 - 1):
            if lod:
                img = downsample(img)
            quant = x_to_uint8(img)
            q_imgs.append(quant)
        return q_imgs

    def _parse_image(example):
        features = tf.parse_single_example(example, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenFeature([], tf.string),
            'attr': tf.FixedLenFeature([40], tf.int64)})
        shape = features['shape']
        data = features['data']
        attr = features['attr']
        data = tf.decode_raw(data, tf.uint8)
        img = tf.reshape(data, shape)
        if transpose:
            img = tf.transpose(img, (1, 2, 0))  # CHW -> HWC
        imgs = _process_image(img)
        parsed = (attr, *imgs)
        return parsed

    return _parse_image


def get_tfr_files(data_dir, split, lgres):
    data_dir = os.path.join(data_dir, split)
    tfr_prefix = os.path.join(data_dir, os.path.basename(data_dir))
    tfr_files = tfr_prefix + '-r%02d-s-*-of-*.tfrecords' % (lgres)
    return tfr_files


def get_tfr_file(data_dir, split, lgres):
    if split:
        data_dir = os.path.join(data_dir, split)
    tfr_prefix = os.path.join(data_dir, os.path.basename(data_dir))
    tfr_file = tfr_prefix + '-r%02d.tfrecords' % (lgres)
    return tfr_file


def dump_celebahq(data_dir, tfrecord_dir, max_res, split, write):
    _NUM_IMAGES = {
        'train': 27000,
        'validation': 3000,
    }

    _NUM_SHARDS = {
        'train': 120,
        'validation': 40,
    }
    resolution_log2 = int(np.log2(max_res))
    if max_res != 2 ** resolution_log2:
        error('Input image resolution must be a power-of-two')
    with tf.Session() as sess:
        print("Reading data from ", data_dir)
        if split:
            tfr_files = get_tfr_files(data_dir, split, int(np.log2(max_res)))
            files = tf.data.Dataset.list_files(tfr_files)
            dset = files.apply(tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=_NUM_PARALLEL_FILE_READERS))
            transpose = False
        else:
            tfr_file = get_tfr_file(data_dir, "", int(np.log2(max_res)))
            dset = tf.data.TFRecordDataset(tfr_file, compression_type='')
            transpose = True

        parse_fn = parse_celeba_image(max_res, transpose)
        dset = dset.map(parse_fn, num_parallel_calls=_NUM_PARALLEL_MAP_CALLS)
        dset = dset.prefetch(1)
        iterator = dset.make_one_shot_iterator()
        _attr, *_imgs = iterator.get_next()
        sess.run(tf.global_variables_initializer())
        splits = [split] if split else ["validation", "train"]
        for split in splits:
            total_imgs = _NUM_IMAGES[split]
            shards = _NUM_SHARDS[split]
            with TFRecordExporter(os.path.join(tfrecord_dir, split), resolution_log2, total_imgs, shards) as tfr:
                for _ in tqdm(range(total_imgs)):
                    attr, *imgs = sess.run([_attr, *_imgs])
                    if write:
                        tfr.add_image(0, imgs, attr)
                if write:
                    assert tfr.cur_images == total_imgs, (
                        tfr.cur_images, total_imgs)

        #attr, *imgs = sess.run([_attr, *_imgs])


def dump_imagenet(data_dir, tfrecord_dir, max_res, split, write):
    _NUM_IMAGES = {
        'train': 1281167,
        'validation': 50000,
    }

    _NUM_FILES = _NUM_SHARDS = {
        'train': 2000,
        'validation': 80,
    }
    resolution_log2 = int(np.log2(max_res))
    if max_res != 2 ** resolution_log2:
        error('Input image resolution must be a power-of-two')

    with tf.Session() as sess:
        is_training = (split == 'train')
        if is_training:
            files = tf.data.Dataset.list_files(
                os.path.join(data_dir, 'train-*-of-01024'))
        else:
            files = tf.data.Dataset.list_files(
                os.path.join(data_dir, 'validation-*-of-00128'))

        files = files.shuffle(buffer_size=_NUM_FILES[split])

        dataset = files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=_NUM_PARALLEL_FILE_READERS))

        dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)
        parse_fn = parse_image(max_res)
        dataset = dataset.map(
            parse_fn, num_parallel_calls=_NUM_PARALLEL_MAP_CALLS)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_one_shot_iterator()

        _label, *_imgs = iterator.get_next()

        sess.run(tf.global_variables_initializer())

        total_imgs = _NUM_IMAGES[split]
        shards = _NUM_SHARDS[split]
        tfrecord_dir = os.path.join(tfrecord_dir, split)
        with TFRecordExporter(tfrecord_dir, resolution_log2, total_imgs, shards) as tfr:
            for _ in tqdm(range(total_imgs)):
                label, *imgs = sess.run([_label, *_imgs])
                if write:
                    tfr.add_image(label, imgs, [])
            assert tfr.cur_images == total_imgs, (tfr.cur_images, total_imgs)

        #label, *imgs = sess.run([_label, *_imgs])


class TFRecordExporter:
    def __init__(self, tfrecord_dir, resolution_log2, expected_images, shards, print_progress=True, progress_interval=10):
        self.tfrecord_dir = tfrecord_dir
        self.tfr_prefix = os.path.join(
            self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.resolution_log2 = resolution_log2
        self.expected_images = expected_images

        self.cur_images = 0
        self.shape = None
        self.tfr_writers = []
        self.print_progress = print_progress
        self.progress_interval = progress_interval
        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert (os.path.isdir(self.tfrecord_dir))
        tfr_opt = tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.NONE)
        for lod in range(self.resolution_log2 - 1):
            p_shard = np.array_split(
                np.random.permutation(expected_images), shards)
            img_to_shard = np.zeros(expected_images, dtype=np.int)
            writers = []
            for shard in range(shards):
                img_to_shard[p_shard[shard]] = shard
                tfr_file = self.tfr_prefix + \
                    '-r%02d-s-%04d-of-%04d.tfrecords' % (
                        self.resolution_log2 - lod, shard, shards)
                writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
            #print(np.unique(img_to_shard, return_counts=True))
            counts = np.unique(img_to_shard, return_counts=True)[1]
            assert len(counts) == shards
            print("Smallest and largest shards have size",
                  np.min(counts), np.max(counts))
            self.tfr_writers.append((writers, img_to_shard))

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for (writers, _) in self.tfr_writers:
            for writer in writers:
                writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def add_image(self, label, imgs, attr):
        assert len(imgs) == len(self.tfr_writers)
        # if self.print_progress and self.cur_images % self.progress_interval == 0:
        #     print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        for lod, (writers, img_to_shard) in enumerate(self.tfr_writers):
            quant = imgs[lod]
            size = 2 ** (self.resolution_log2 - lod)
            assert quant.shape == (size, size, 3), quant.shape
            ex = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'shape': _int64_feature(quant.shape),
                        'data': _bytes_feature(quant.tostring()),
                        'label': _int64_feature(label),
                        'attr': _int64_feature(attr)
                    }
                )
            )
            writers[img_to_shard[self.cur_images]].write(
                ex.SerializeToString())
        self.cur_images += 1

    # def add_labels(self, labels):
    #     if self.print_progress:
    #         print('%-40s\r' % 'Saving labels...', end='', flush=True)
    #     assert labels.shape[0] == self.cur_images
    #     with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
    #         np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--max_res", type=int, default=256, help="Image size")
    parser.add_argument("--tfrecord_dir", type=str,
                        required=True, help='place to dump')
    parser.add_argument("--write", action='store_true',
                        help="Whether to write")
    hps = parser.parse_args()  # So error if typo
    #dump_imagenet(hps.data_dir, hps.tfrecord_dir, hps.max_res, 'validation', hps.write)
    #dump_imagenet(hps.data_dir, hps.tfrecord_dir, hps.max_res, 'train', hps.write)
    dump_celebahq(hps.data_dir, hps.tfrecord_dir,
                  hps.max_res, 'validation', hps.write)
    dump_celebahq(hps.data_dir, hps.tfrecord_dir,
                  hps.max_res, 'train', hps.write)
