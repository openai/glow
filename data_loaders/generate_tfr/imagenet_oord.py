# Copyright 2016 Google Inc. All Rights Reserved.
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
Generate tfrecords for ImageNet 32x32 and 64x64.

# Get images
Downloaded images from http://image-net.org/small/download.php, and unzip them.
(Move one file from training to test to have 50000 test images)

# Get tfr file from images
Use this script to generate the tfr file.
python imagenet_oord.py --res [RES] --tfrecord_dir [OUTPUT_FOLDER] --write

"""

from __future__ import print_function

import os
import os.path

import scipy.io
import scipy.io.wavfile
import scipy.ndimage
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from typing import Iterable


def _int64_feature(value):
    if not isinstance(value, Iterable):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def dump(fn_root, tfrecord_dir, max_res, expected_images, shards, write):
    """Main converter function."""
    # fn_root = FLAGS.fn_root
    # max_res = FLAGS.max_res
    resolution_log2 = int(np.log2(max_res))
    tfr_prefix = os.path.join(tfrecord_dir, os.path.basename(tfrecord_dir))

    print("Checking in", fn_root)
    img_fn_list = os.listdir(fn_root)
    img_fn_list = [img_fn for img_fn in img_fn_list
                   if img_fn.endswith('.png')]
    num_examples = len(img_fn_list)
    print("Found", num_examples)
    assert num_examples == expected_images

    # Sharding
    tfr_opt = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.NONE)
    p_shard = np.array_split(np.random.permutation(expected_images), shards)
    img_to_shard = np.zeros(expected_images, dtype=np.int)
    writers = []
    for shard in range(shards):
        img_to_shard[p_shard[shard]] = shard
        tfr_file = tfr_prefix + \
            '-r%02d-s-%04d-of-%04d.tfrecords' % (
                resolution_log2, shard, shards)
        writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    # print(np.unique(img_to_shard, return_counts=True))
    counts = np.unique(img_to_shard, return_counts=True)[1]
    assert len(counts) == shards
    print("Smallest and largest shards have size",
          np.min(counts), np.max(counts))

    for example_idx, img_fn in enumerate(tqdm(img_fn_list)):
        shard = img_to_shard[example_idx]
        img = scipy.ndimage.imread(os.path.join(fn_root, img_fn))
        rows = img.shape[0]
        cols = img.shape[1]
        depth = img.shape[2]
        shape = (rows, cols, depth)
        img = img.astype("uint8")
        img = img.tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "shape": _int64_feature(shape),
                    "data": _bytes_feature(img),
                    "label": _int64_feature(0)
                }
            )
        )
        if write:
            writers[shard].write(example.SerializeToString())

    print('%-40s\r' % 'Flushing data...', end='', flush=True)
    for writer in writers:
        writer.close()

    print('%-40s\r' % '', end='', flush=True)
    print('Added %d images.' % num_examples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=32, help="Image size")
    parser.add_argument("--tfrecord_dir", type=str,
                        required=True, help='place to dump')
    parser.add_argument("--write", action='store_true',
                        help="Whether to write")
    hps = parser.parse_args()

    # Imagenet
    _NUM_IMAGES = {
        'train': 1281148,
        'validation': 50000,
    }

    _NUM_SHARDS = {
        'train': 2000,
        'validation': 80,
    }

    _FILE = {
        'train': 'train_%dx%d' % (hps.res, hps.res),
        'validation': 'valid_%dx%d' % (hps.res, hps.res),
    }

    for split in ['validation', 'train']:
        fn_root = _FILE[split]
        tfrecord_dir = os.path.join(hps.tfrecord_dir, split)
        total_imgs = _NUM_IMAGES[split]
        shards = _NUM_SHARDS[split]
        if not os.path.exists(tfrecord_dir):
            os.mkdir(tfrecord_dir)
        dump(fn_root, tfrecord_dir, hps.res, total_imgs, shards, hps.write)
