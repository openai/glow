import os
import tensorflow as tf
import numpy as np
import glob

_SMOOTH_INTERPOLATION = tf.image.ResizeMethod.BILINEAR
_DOWNSAMPLING = tf.image.ResizeMethod.AREA
_FILES_SHUFFLE = 1024
_SHUFFLE_FACTOR = 4
# _NUM_PARALLEL_FILE_READERS = 2
# _NUM_PARALLEL_MAP_CALLS = 1

def x_to_uint8(x):
    return tf.cast(tf.clip_by_value(tf.floor(x), 0, 255), 'uint8')

def batch_downsample(img, label, lowest, factor):
    _, h, w, _ = img.get_shape().as_list()
    img_f = tf.cast(img, 'float32')
    if lowest:
        img_d = 128 * tf.ones_like(img_f)[:, :h // factor, :w // factor, :]
        img_d = x_to_uint8(img_d)
    else:
        img_d = tf.image.resize_images(img_f, [h // factor, w // factor], method=_DOWNSAMPLING)
        img_d = x_to_uint8(img_d)

    return img, img_d, label

def parse_tfrecord_tf(record, max_res, res, htrans, wtrans, shift = 0):
    features = tf.parse_single_example(record, features={
        'data': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([1],tf.int64)})
    #shape = features['shape']
    data, label = features['data'], features['label']
    label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32) - shift
    data = tf.decode_raw(data, tf.uint8)
    img = tf.reshape(data, [max_res, max_res, 3])

    # random flip
    img = tf.image.random_flip_left_right(img)

    # random crop and resize
    img = tf.random_crop(img, [max_res - htrans, max_res - wtrans, 3])
    img = tf.image.resize_images(img, [max_res, max_res], method=_SMOOTH_INTERPOLATION)

    # Downsampling
    if res != max_res:
        img = tf.image.resize_images(img, [res, res], method=_DOWNSAMPLING)

    return img, label

#
# def batch_downsample(imgs, labels, lowest, factor):
#     img_f = tf.cast(images, 'float32')
#
# def downsample(img, label):
#     return img, _downsample(img), label

def input_fn(tfr_file, shards, rank, pmap, fmap, n_batch, max_res, resolution, htrans, wtrans, is_training, lowest, factor = 2):
    #
    shift = 1 if 'celeba' in tfr_file else 0
    print("Shifting labels by ", shift)
    files = tf.data.Dataset.list_files(tfr_file)

    if is_training:
        files = files.shard(shards, rank)  # each worker works on a subset of the data
        files = files.shuffle(buffer_size = _FILES_SHUFFLE) # shuffle order of files in shard

    dset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=fmap))

    if is_training:
        dset = dset.shuffle(buffer_size=n_batch * _SHUFFLE_FACTOR)
        #dset = dset.apply(tf.contrib.shuffle_and_repeat(buffer_size=(dps.batch_size * _SHUFFLE_FACTOR)))

    dset = dset.repeat()
    dset = dset.map(lambda x: parse_tfrecord_tf(x, max_res, resolution, htrans, wtrans, shift), num_parallel_calls= pmap)

    dset = dset.batch(n_batch)
    dset = dset.map(lambda img, label:batch_downsample(img, label, lowest, factor))
    #dset = dset.apply(tf.contrib.data.map_and_batch(map_fn = map_fn, batch_size = batch_size)

    dset = dset.prefetch(1)
    itr = dset.make_one_shot_iterator()
    return itr

def get_tfr_file(data_dir, split, res_lg2):
    data_dir = os.path.join(data_dir, split)
    tfr_prefix = os.path.join(data_dir, os.path.basename(data_dir))
    tfr_file = tfr_prefix + '-r%02d-s-*-of-*.tfrecords' % (res_lg2)
    #print(split, res_lg2, len(glob.glob(tfr_file)))
    # if split == 'train':
    #     exp_len = 1024
    # else:
    #     exp_len = 128
    # assert len(glob.glob(tfr_file)) == exp_len
    return tfr_file

def get_data(sess, data_dir, shards, rank, pmap, fmap, n_batch, n_init, max_res, resolution, htrans, wtrans, lowest):
    # data_dir = dps.data_dir
    # n_batch = dps.n_batch
    # n_init = dps.n_init
    # max_res = dps.max_res
    # resolution = dps.resolution
    # lowest = dps.lowest

    assert max_res == 2 ** int(np.log2(max_res))
    assert resolution == 2 ** int(np.log2(resolution))

    train_file = get_tfr_file(data_dir, 'train', int(np.log2(max_res)))
    valid_file = get_tfr_file(data_dir, 'validation', int(np.log2(max_res)))
    fres_file = get_tfr_file(data_dir, 'validation', int(np.log2(max_res)))

    init_itr = input_fn(train_file, shards, rank, pmap, fmap, n_batch, max_res, resolution, htrans, wtrans, True, False)
    train_itr = input_fn(train_file, shards, rank, pmap, fmap, n_batch, max_res, resolution, htrans, wtrans, True, lowest)
    valid_itr = input_fn(valid_file, shards, rank, pmap, fmap, n_batch, max_res, resolution, htrans, wtrans, False, lowest)

    # Modify batch size and resolution for full res
    fres_itr = input_fn(fres_file, shards, rank, pmap, fmap, 1, max_res, max_res, htrans, wtrans, False, lowest, 2*(max_res // resolution))

    data_init = make_batch(sess, init_itr, n_batch, n_init)
    fres_init = make_batch(sess, fres_itr, 1, 64)

    return train_itr, valid_itr, data_init, fres_init
#
def make_batch(sess, itr, itr_batch_size, required_batch_size):
    ib, rb = itr_batch_size, required_batch_size
    #assert rb % ib == 0
    k = int(np.ceil(rb / ib))
    xs, x_ds, ys  = [], [], []
    data = itr.get_next()
    for i in range(k):
        x, x_d, y  = sess.run(data)
        xs.append(x)
        x_ds.append(x_d)
        ys.append(y)
    x, x_d, y  = np.concatenate(xs)[:rb], np.concatenate(x_ds)[:rb], np.concatenate(ys)[:rb]
    return {'x': x, 'x_d': x_d, 'y': y}