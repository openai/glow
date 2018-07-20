import os
import tensorflow as tf
import numpy as np
import glob

_FILES_SHUFFLE = 1024
_SHUFFLE_FACTOR = 4


def parse_tfrecord_tf(record, res, rnd_crop):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([1], tf.int64)})
    # label is always 0 if uncondtional
    # to get CelebA attr, add 'attr': tf.FixedLenFeature([40], tf.int64)
    data, label, shape = features['data'], features['label'], features['shape']
    label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)
    img = tf.decode_raw(data, tf.uint8)
    if rnd_crop:
        # For LSUN Realnvp only - random crop
        img = tf.reshape(img, shape)
        img = tf.random_crop(img, [res, res, 3])
    img = tf.reshape(img, [res, res, 3])
    return img, label  # to get CelebA attr, also return attr


def input_fn(tfr_file, shards, rank, pmap, fmap, n_batch, resolution, rnd_crop, is_training):
    files = tf.data.Dataset.list_files(tfr_file)
    if ('lsun' not in tfr_file) or is_training:
        # For 'lsun' validation, only one shard and each machine goes over the full dataset
        # each worker works on a subset of the data
        files = files.shard(shards, rank)
    if is_training:
        # shuffle order of files in shard
        files = files.shuffle(buffer_size=_FILES_SHUFFLE)
    dset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=fmap))
    if is_training:
        dset = dset.shuffle(buffer_size=n_batch * _SHUFFLE_FACTOR)
    dset = dset.repeat()
    dset = dset.map(lambda x: parse_tfrecord_tf(
        x, resolution, rnd_crop), num_parallel_calls=pmap)
    dset = dset.batch(n_batch)
    dset = dset.prefetch(1)
    itr = dset.make_one_shot_iterator()
    return itr


def get_tfr_file(data_dir, split, res_lg2):
    data_dir = os.path.join(data_dir, split)
    tfr_prefix = os.path.join(data_dir, os.path.basename(data_dir))
    tfr_file = tfr_prefix + '-r%02d-s-*-of-*.tfrecords' % (res_lg2)
    files = glob.glob(tfr_file)
    assert len(files) == int(files[0].split(
        "-")[-1].split(".")[0]), "Not all tfrecords files present at %s" % tfr_prefix
    return tfr_file


def get_data(sess, data_dir, shards, rank, pmap, fmap, n_batch_train, n_batch_test, n_batch_init, resolution, rnd_crop):
    assert resolution == 2 ** int(np.log2(resolution))

    train_file = get_tfr_file(data_dir, 'train', int(np.log2(resolution)))
    valid_file = get_tfr_file(data_dir, 'validation', int(np.log2(resolution)))

    train_itr = input_fn(train_file, shards, rank, pmap,
                         fmap, n_batch_train, resolution, rnd_crop, True)
    valid_itr = input_fn(valid_file, shards, rank, pmap,
                         fmap, n_batch_test, resolution, rnd_crop, False)

    data_init = make_batch(sess, train_itr, n_batch_train, n_batch_init)

    return train_itr, valid_itr, data_init

#


def make_batch(sess, itr, itr_batch_size, required_batch_size):
    ib, rb = itr_batch_size, required_batch_size
    #assert rb % ib == 0
    k = int(np.ceil(rb / ib))
    xs, ys = [], []
    data = itr.get_next()
    for i in range(k):
        x, y = sess.run(data)
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs)[:rb], np.concatenate(ys)[:rb]
    return {'x': x, 'y': y}
