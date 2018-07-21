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

""""
LSUN dataset

# Get image files
Download the LSUN dataset as follows:
git clone https://github.com/fyu/lsun.git
cd lsun
python2.7 download.py -c [CATEGORY]
Unzip the downloaded .zip files and execute:
python2.7 data.py export [IMAGE_DB_PATH] --out_dir [LSUN_FOLDER] --flat

# Get tfr file from images
Use this script to generate the tfr file.
python lsun.py --res [RES] --category [CATEGORY] --lsun_dir [LSUN_FOLDER] --tfrecord_dir [OUTPUT_FOLDER] --write [--realnvp]
Without realnvp flag you get 256x256 centre cropped area downsampled images, with flag you get 96x96 images with realnvp preprocessing.
"""

from __future__ import print_function

import os
import os.path

import numpy
import skimage.transform
from PIL import Image
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


def centre_crop(img):
    h, w = img.shape[:2]
    crop = min(h, w)
    return img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]


def dump(fn_root, tfrecord_dir, max_res, expected_images, shards, write, realnvp=False):
    """Main converter function."""
    resolution_log2 = int(np.log2(max_res))
    tfr_prefix = os.path.join(tfrecord_dir, os.path.basename(tfrecord_dir))

    print("Checking in", fn_root)
    img_fn_list = os.listdir(fn_root)
    img_fn_list = [img_fn for img_fn in img_fn_list
                   if img_fn.endswith('.webp')]
    num_examples = len(img_fn_list)
    print("Found", num_examples)
    assert num_examples == expected_images

    tfr_opt = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.NONE)
    p_shard = np.array_split(np.random.permutation(expected_images), shards)
    img_to_shard = np.zeros(expected_images, dtype=np.int)
    writers = []
    for shard in tqdm(range(shards)):
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
        img = numpy.array(Image.open(os.path.join(fn_root, img_fn)))
        rows = img.shape[0]
        cols = img.shape[1]
        if realnvp:
            downscale = min(rows / 96., cols / 96.)
            img = skimage.transform.pyramid_reduce(img, downscale)
            img *= 255.
            img = img.astype("uint8")
        else:
            img = centre_crop(img)
            img = Image.fromarray(img, 'RGB')
            img = img.resize((max_res, max_res), Image.ANTIALIAS)
            img = np.asarray(img)
        rows = img.shape[0]
        cols = img.shape[1]
        depth = img.shape[2]
        shape = (rows, cols, depth)
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
    parser.add_argument("--category", type=str, help="LSUN category")
    parser.add_argument("--realnvp", action='store_true',
                        help="Use this flag to do realnvp preprocessing instead of our centre-crops")
    parser.add_argument("--res", type=int, default=256, help="Image size")
    parser.add_argument("--lsun_dir", type=str,
                        required=True, help="place of lsun dir")
    parser.add_argument("--tfrecord_dir", type=str,
                        required=True, help='place to dump')
    parser.add_argument("--write", action='store_true',
                        help="Whether to write")
    hps = parser.parse_args()

    # LSUN
    # CATEGORIES = ["bedroom", "bridge", "church_outdoor", "classroom", "conference_room", "dining_room", "kitchen", "living"]
    base_tfr = hps.tfrecord_dir
    res = hps.res
    for realnvp in [False, True]:
        for category in ["tower", "church_outdoor", "bedroom"]:
            hps.realnvp = realnvp
            hps.category = category
            if realnvp:
                hps.tfrecord_dir = "%s_%s/%s" % (base_tfr,
                                                 "realnvp", hps.category)
            else:
                hps.tfrecord_dir = "%s/%s" % (base_tfr, hps.category)
            print(hps.realnvp, hps.category, hps.lsun_dir, hps.tfrecord_dir)
            imgs = {
                'bedroom': 3033042,
                'bridge': 818687,
                'church_outdoor': 126227,
                'classroom': 168103,
                'conference_room': 229069,
                'dining_room': 657571,
                'kitchen': 2212277,
                'living_room': 1315802,
                'restaurant': 626331,
                'tower': 708264
            }

            _NUM_IMAGES = {
                'train': imgs[hps.category],
                'validation': 300,
            }

            _NUM_SHARDS = {
                'train': 2560,
                'validation': 1,
            }

            _FILE = {
                'train': os.path.join(hps.lsun_dir, '%s_train' % hps.category),
                'validation': os.path.join(hps.lsun_dir, '%s_val' % hps.category)

            }

            if hps.realnvp:
                res = 96
            else:
                res = hps.res

            for split in ['validation', 'train']:
                fn_root = _FILE[split]
                tfrecord_dir = os.path.join(hps.tfrecord_dir, split)
                total_imgs = _NUM_IMAGES[split]
                shards = _NUM_SHARDS[split]
                if not os.path.exists(tfrecord_dir):
                    os.mkdir(tfrecord_dir)
                dump(fn_root, tfrecord_dir, res, total_imgs,
                     shards, hps.write, hps.realnvp)
