import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
import horovod.tensorflow as hvd

# Debugging function
do_print_act_stats = True


def print_act_stats(x, _str=""):
    if not do_print_act_stats:
        return x
    if hvd.rank() != 0:
        return x
    if len(x.get_shape()) == 1:
        x_mean, x_var = tf.nn.moments(x, [0], keep_dims=True)
    if len(x.get_shape()) == 2:
        x_mean, x_var = tf.nn.moments(x, [0], keep_dims=True)
    if len(x.get_shape()) == 4:
        x_mean, x_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
    stats = [tf.reduce_min(x_mean), tf.reduce_mean(x_mean), tf.reduce_max(x_mean),
             tf.reduce_min(tf.sqrt(x_var)), tf.reduce_mean(tf.sqrt(x_var)), tf.reduce_max(tf.sqrt(x_var))]
    return tf.Print(x, stats, "["+_str+"] "+x.name)

# Allreduce methods


def allreduce_sum(x):
    if hvd.size() == 1:
        return x
    return hvd.mpi_ops._allreduce(x)


def allreduce_mean(x):
    x = allreduce_sum(x) / hvd.size()
    return x


def default_initial_value(shape, std=0.05):
    return tf.random_normal(shape, 0., std)


def default_initializer(std=0.05):
    return tf.random_normal_initializer(0., std)


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))

# wrapper tf.get_variable, augmented with 'init' functionality
# Get variable with data dependent init


@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False, trainable=True):
    w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
    if init:
        w = w.assign(initial_value)
        with tf.control_dependencies([w]):
            return w
    return w

# Activation normalization
# Convenience function that does centering+scaling


@add_arg_scope
def actnorm(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, init=False, trainable=True):
    if arg_scope([get_variable_ddi], trainable=trainable):
        if not reverse:
            x = actnorm_center(name+"_center", x, reverse)
            x = actnorm_scale(name+"_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
        else:
            x = actnorm_scale(name + "_scale", x, scale, logdet,
                              logscale_factor, batch_variance, reverse, init)
            if logdet != None:
                x, logdet = x
            x = actnorm_center(name+"_center", x, reverse)
        if logdet != None:
            return x, logdet
        return x

# Activation normalization


@add_arg_scope
def actnorm_center(name, x, reverse=False):
    shape = x.get_shape()
    with tf.variable_scope(name):
        assert len(shape) == 2 or len(shape) == 4
        if len(shape) == 2:
            x_mean = tf.reduce_mean(x, [0], keepdims=True)
            b = get_variable_ddi(
                "b", (1, int_shape(x)[1]), initial_value=-x_mean)
        elif len(shape) == 4:
            x_mean = tf.reduce_mean(x, [0, 1, 2], keepdims=True)
            b = get_variable_ddi(
                "b", (1, 1, 1, int_shape(x)[3]), initial_value=-x_mean)

        if not reverse:
            x += b
        else:
            x -= b

        return x

# Activation normalization


@add_arg_scope
def actnorm_scale(name, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, init=False, trainable=True):
    shape = x.get_shape()
    with tf.variable_scope(name), arg_scope([get_variable_ddi], trainable=trainable):
        assert len(shape) == 2 or len(shape) == 4
        if len(shape) == 2:
            x_var = tf.reduce_mean(x**2, [0], keepdims=True)
            logdet_factor = 1
            _shape = (1, int_shape(x)[1])

        elif len(shape) == 4:
            x_var = tf.reduce_mean(x**2, [0, 1, 2], keepdims=True)
            logdet_factor = int(shape[1])*int(shape[2])
            _shape = (1, 1, 1, int_shape(x)[3])

        if batch_variance:
            x_var = tf.reduce_mean(x**2, keepdims=True)

        if init and False:
            # MPI all-reduce
            x_var = allreduce_mean(x_var)
            # Somehow this also slows down graph when not initializing
            # (it's not optimized away?)

        if True:
            logs = get_variable_ddi("logs", _shape, initial_value=tf.log(
                scale/(tf.sqrt(x_var)+1e-6))/logscale_factor)*logscale_factor
            if not reverse:
                x = x * tf.exp(logs)
            else:
                x = x * tf.exp(-logs)
        else:
            # Alternative, doesn't seem to do significantly worse or better than the logarithmic version above
            s = get_variable_ddi("s", _shape, initial_value=scale /
                                 (tf.sqrt(x_var) + 1e-6) / logscale_factor)*logscale_factor
            logs = tf.log(tf.abs(s))
            if not reverse:
                x *= s
            else:
                x /= s

        if logdet != None:
            dlogdet = tf.reduce_sum(logs) * logdet_factor
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x

# Linear layer with layer norm


@add_arg_scope
def linear(name, x, width, do_weightnorm=True, do_actnorm=True, initializer=None, scale=1.):
    initializer = initializer or default_initializer()
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable("W", [n_in, width],
                            tf.float32, initializer=initializer)
        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0])
        x = tf.matmul(x, w)
        x += tf.get_variable("b", [1, width],
                             initializer=tf.zeros_initializer())
        if do_actnorm:
            x = actnorm("actnorm", x, scale)
        return x

# Linear layer with zero init


@add_arg_scope
def linear_zeros(name, x, width, logscale_factor=3):
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable("W", [n_in, width], tf.float32,
                            initializer=tf.zeros_initializer())
        x = tf.matmul(x, w)
        x += tf.get_variable("b", [1, width],
                             initializer=tf.zeros_initializer())
        x *= tf.exp(tf.get_variable("logs",
                                    [1, width], initializer=tf.zeros_initializer()) * logscale_factor)
        return x

# Slow way to add edge padding


def add_edge_padding(x, filter_size):
    assert filter_size[0] % 2 == 1
    if filter_size[0] == 1 and filter_size[1] == 1:
        return x
    a = (filter_size[0] - 1) // 2  # vertical padding size
    b = (filter_size[1] - 1) // 2  # horizontal padding size
    if True:
        x = tf.pad(x, [[0, 0], [a, a], [b, b], [0, 0]])
        name = "_".join([str(dim) for dim in [a, b, *int_shape(x)[1:3]]])
        pads = tf.get_collection(name)
        if not pads:
            if hvd.rank() == 0:
                print("Creating pad", name)
            pad = np.zeros([1] + int_shape(x)[1:3] + [1], dtype='float32')
            pad[:, :a, :, 0] = 1.
            pad[:, -a:, :, 0] = 1.
            pad[:, :, :b, 0] = 1.
            pad[:, :, -b:, 0] = 1.
            pad = tf.convert_to_tensor(pad)
            tf.add_to_collection(name, pad)
        else:
            pad = pads[0]
        pad = tf.tile(pad, [tf.shape(x)[0], 1, 1, 1])
        x = tf.concat([x, pad], axis=3)
    else:
        pad = tf.pad(tf.zeros_like(x[:, :, :, :1]) - 1,
                     [[0, 0], [a, a], [b, b], [0, 0]]) + 1
        x = tf.pad(x, [[0, 0], [a, a], [b, b], [0, 0]])
        x = tf.concat([x, pad], axis=3)
    return x


@add_arg_scope
def conv2d(name, x, width, filter_size=[3, 3], stride=[1, 1], pad="SAME", do_weightnorm=False, do_actnorm=True, context1d=None, skip=1, edge_bias=True):
    with tf.variable_scope(name):
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[3])

        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=default_initializer())
        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0, 1, 2])
        if skip == 1:
            x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1
            x = tf.nn.atrous_conv2d(x, w, skip, pad)
        if do_actnorm:
            x = actnorm("actnorm", x)
        else:
            x += tf.get_variable("b", [1, 1, 1, width],
                                 initializer=tf.zeros_initializer())

        if context1d != None:
            x += tf.reshape(linear("context", context1d,
                                   width), [-1, 1, 1, width])
    return x


@add_arg_scope
def separable_conv2d(name, x, width, filter_size=[3, 3], stride=[1, 1], padding="SAME", do_actnorm=True, std=0.05):
    n_in = int(x.get_shape()[3])
    with tf.variable_scope(name):
        assert filter_size[0] % 2 == 1 and filter_size[1] % 2 == 1
        strides = [1] + stride + [1]
        w1_shape = filter_size + [n_in, 1]
        w1_init = np.zeros(w1_shape, dtype='float32')
        w1_init[(filter_size[0]-1)//2, (filter_size[1]-1)//2, :,
                :] = 1.  # initialize depthwise conv as identity
        w1 = tf.get_variable("W1", dtype=tf.float32, initializer=w1_init)
        w2_shape = [1, 1, n_in, width]
        w2 = tf.get_variable("W2", w2_shape, tf.float32,
                             initializer=default_initializer(std))
        x = tf.nn.separable_conv2d(
            x, w1, w2, strides, padding, data_format='NHWC')
        if do_actnorm:
            x = actnorm("actnorm", x)
        else:
            x += tf.get_variable("b", [1, 1, 1, width],
                                 initializer=tf.zeros_initializer(std))

    return x


@add_arg_scope
def conv2d_zeros(name, x, width, filter_size=[3, 3], stride=[1, 1], pad="SAME", logscale_factor=3, skip=1, edge_bias=True):
    with tf.variable_scope(name):
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[3])
        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=tf.zeros_initializer())
        if skip == 1:
            x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1
            x = tf.nn.atrous_conv2d(x, w, skip, pad)
        x += tf.get_variable("b", [1, 1, 1, width],
                             initializer=tf.zeros_initializer())
        x *= tf.exp(tf.get_variable("logs",
                                    [1, width], initializer=tf.zeros_initializer()) * logscale_factor)
    return x


# 2X nearest-neighbour upsampling, also inspired by Jascha Sohl-Dickstein's code
def upsample2d_nearest_neighbour(x):
    shape = x.get_shape()
    n_batch = int(shape[0])
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    x = tf.reshape(x, (n_batch, height, 1, width, 1, n_channels))
    x = tf.concat(2, [x, x])
    x = tf.concat(4, [x, x])
    x = tf.reshape(x, (n_batch, height*2, width*2, n_channels))
    return x


def upsample(x, factor=2):
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    x = tf.image.resize_nearest_neighbor(x, [height * factor, width * factor])
    return x


def squeeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert height % factor == 0 and width % factor == 0
    x = tf.reshape(x, [-1, height//factor, factor,
                       width//factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
    x = tf.reshape(x, [-1, height//factor, width //
                       factor, n_channels*factor*factor])
    return x


def unsqueeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert n_channels >= 4 and n_channels % 4 == 0
    x = tf.reshape(
        x, (-1, height, width, int(n_channels/factor**2), factor, factor))
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, (-1, int(height*factor),
                       int(width*factor), int(n_channels/factor**2)))
    return x

# Reverse features across channel dimension


def reverse_features(name, h, reverse=False):
    return h[:, :, :, ::-1]

# Shuffle across the channel dimension


def shuffle_features(name, h, indices=None, return_indices=False, reverse=False):
    with tf.variable_scope(name):

        rng = np.random.RandomState(
            (abs(hash(tf.get_variable_scope().name))) % 10000000)

        if indices == None:
            # Create numpy and tensorflow variables with indices
            n_channels = int(h.get_shape()[-1])
            indices = list(range(n_channels))
            rng.shuffle(indices)
            # Reverse it
            indices_inverse = [0]*n_channels
            for i in range(n_channels):
                indices_inverse[indices[i]] = i

        tf_indices = tf.get_variable("indices", dtype=tf.int32, initializer=np.asarray(
            indices, dtype='int32'), trainable=False)
        tf_indices_reverse = tf.get_variable("indices_inverse", dtype=tf.int32, initializer=np.asarray(
            indices_inverse, dtype='int32'), trainable=False)

        _indices = tf_indices
        if reverse:
            _indices = tf_indices_reverse

        if len(h.get_shape()) == 2:
            # Slice
            h = tf.transpose(h)
            h = tf.gather(h, _indices)
            h = tf.transpose(h)
        elif len(h.get_shape()) == 4:
            # Slice
            h = tf.transpose(h, [3, 1, 2, 0])
            h = tf.gather(h, _indices)
            h = tf.transpose(h, [3, 1, 2, 0])
        if return_indices:
            return h, indices
        return h


def embedding(name, y, n_y, width):
    with tf.variable_scope(name):
        params = tf.get_variable(
            "embedding", [n_y, width], initializer=default_initializer())
        embeddings = tf.gather(params, y)
        return embeddings

# Random variables


def flatten_sum(logps):
    if len(logps.get_shape()) == 2:
        return tf.reduce_sum(logps, [1])
    elif len(logps.get_shape()) == 4:
        return tf.reduce_sum(logps, [1, 2, 3])
    else:
        raise Exception()


def standard_gaussian(shape):
    return gaussian_diag(tf.zeros(shape), tf.zeros(shape))


def gaussian_diag(mean, logsd):
    class o(object):
        pass
    o.mean = mean
    o.logsd = logsd
    o.eps = tf.random_normal(tf.shape(mean))
    o.sample = mean + tf.exp(logsd) * o.eps
    o.sample2 = lambda eps: mean + tf.exp(logsd) * eps
    o.logps = lambda x: -0.5 * \
        (np.log(2 * np.pi) + 2. * logsd + (x - mean) ** 2 / tf.exp(2. * logsd))
    o.logp = lambda x: flatten_sum(o.logps(x))
    o.get_eps = lambda x: (x - mean) / tf.exp(logsd)
    return o


# def discretized_logistic_old(mean, logscale, binsize=1 / 256.0, sample=None):
#    scale = tf.exp(logscale)
#    sample = (tf.floor(sample / binsize) * binsize - mean) / scale
#    logp = tf.log(tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)
#    return tf.reduce_sum(logp, [1, 2, 3])

def discretized_logistic(mean, logscale, binsize=1. / 256):
    class o(object):
        pass
    o.mean = mean
    o.logscale = logscale
    scale = tf.exp(logscale)

    def logps(x):
        x = (x - mean) / scale
        return tf.log(tf.sigmoid(x + binsize / scale) - tf.sigmoid(x) + 1e-7)
    o.logps = logps
    o.logp = lambda x: flatten_sum(logps(x))
    return o


def _symmetric_matrix_square_root(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    s, u, v = tf.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return tf.matmul(
        tf.matmul(u, tf.diag(si)), v, transpose_b=True)
