#!/usr/bin/env python

# Modified Horovod MNIST example

import tensorflow as tf
import horovod.tensorflow as hvd
import time, sys, os
import numpy as np
import zeus

learn = tf.contrib.learn

# Surpress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def init_visualizations(hps, model, logdirs):

    def decode_batch(y, eps):
        n_batch = hps.local_batch_train
        xs = []
        for i in range(int(np.ceil(len(eps) / n_batch))):
            xs.append(model.decode(y[i*n_batch:i*n_batch + n_batch], eps[i*n_batch:i*n_batch + n_batch]))
        return np.concatenate(xs)

    def draw_samples(epoch):
        if hvd.rank() != 0:
            return

        rows = 10 if hps.image_size <=64 else 4
        cols = rows
        n_batch = rows*cols
        y = np.asarray([_y % hps.n_y for _y in (list(range(cols)) * rows)], dtype='int32')

        #temperatures = [0., .25, .5, .626, .75, .875, 1.] #previously
        temperatures = [0., .25, .5, .6, .7, .8, .9, 1.]

        x_samples = []
        x_samples.append(decode_batch(y, [.0]*n_batch))
        x_samples.append(decode_batch(y, [.25]*n_batch))
        x_samples.append(decode_batch(y, [.5]*n_batch))
        x_samples.append(decode_batch(y, [.6]*n_batch))
        x_samples.append(decode_batch(y, [.7]*n_batch))
        x_samples.append(decode_batch(y, [.8]*n_batch))
        x_samples.append(decode_batch(y, [.9] * n_batch))
        x_samples.append(decode_batch(y, [1.]*n_batch))
        #previously: 0, .25, .5, .625, .75, .875, 1.
        
        for i in range(len(x_samples)):
            x_sample = np.reshape(x_samples[i], (n_batch, hps.image_size, hps.image_size, 3))
            zeus.graphics.save_raster(x_sample, logdirs[0] + 'epoch_{}_sample_{}.png'.format(epoch, i))
            #zeus.graphics.save_raster(x_sample, logdirs[0] + 'sample{}.png'.format(i), width=np.sqrt(n_samples))

    return draw_samples

# ===
# Code for getting data
# ===
def get_data(hps, sess):
    if hps.image_size == -1:
        hps.image_size = {'mnist':32,'cifar10':32,'imagenet-oord':64,'imagenet':256,'celeba':256,'lsun_realnvp':64,'lsun':256}[hps.problem]
    if hps.n_test == -1:
        hps.n_test = {'mnist':10000,'cifar10':10000,'imagenet-oord':50000,'imagenet':50000,'celeba':3000, 'lsun_realnvp':300*hvd.size(), 'lsun':300*hvd.size()}[hps.problem]
    hps.n_y = {'mnist':10, 'cifar10': 10, 'imagenet-oord':1000, 'imagenet':1000, 'celeba':1, 'lsun_realnvp':1, 'lsun':1}[hps.problem]
    if hps.data_dir == "":
        hps.data_dir = {'mnist':None,'cifar10':None,'imagenet-oord':'/mnt/host/imagenet-oord-tfr', 'imagenet':'/mnt/host/imagenet-tfr', 'celeba':'/mnt/host/celeba-reshard-tfr', 'lsun_realnvp':'/mnt/host/lsun_realnvp', 'lsun':'/mnt/host/lsun'}[hps.problem]

    if hps.problem == 'lsun_realnvp':
        hps.rnd_crop = True
    else:
        hps.rnd_crop = False

    if hps.category:
        hps.data_dir += ('/%s' % hps.category)

    s = hps.anchor_size

    hps.local_batch_train = hps.n_batch_train * s * s // (hps.image_size * hps.image_size)
    hps.local_batch_test = {64:50, 32:25, 16:10, 8:5, 4:2, 2:2, 1:1}[hps.local_batch_train]  # round down to closest divisor of 50
    hps.local_batch_init = hps.n_batch_init * s * s // (hps.image_size * hps.image_size)

    print("Rank {} Batch sizes Train {} Test {} Init {}".format(hvd.rank(), hps.local_batch_train, hps.local_batch_test, hps.local_batch_init))

    if hps.problem in ['imagenet-oord', 'imagenet', 'celeba', 'lsun_realnvp', 'lsun']:
        hps.direct_iterator = True
        import data_loaders.get_data as v
        train_iterator, test_iterator, data_init = \
            v.get_data(sess, hps.data_dir, hvd.size(), hvd.rank(), hps.pmap, hps.fmap, hps.local_batch_train, hps.local_batch_test, hps.local_batch_init, hps.image_size, hps.rnd_crop)

    elif hps.problem in ['mnist','cifar10']:
        hps.direct_iterator = False
        import mnistcifar10.get_data as v
        train_iterator, test_iterator, data_init = \
            v.get_data(hps.problem, hvd.size(), hvd.rank(), hps.dal, hps.local_batch_train, hps.local_batch_test, hps.local_batch_init,  hps.image_size)

    else:
        raise Exception()

    return train_iterator, test_iterator, data_init

def check_test_iterator(sess, test_iterator):
    ys = []
    xms = []
    if hps.direct_iterator:
        data = test_iterator.get_next()
    for it in range(hps.full_test_its):
        if hps.direct_iterator:
            x, y = sess.run(data)
        else:
            x, y = test_iterator()
        ys.append(y)
        xms.append(np.mean(x, axis=(1,2,3)))
    ys = np.concatenate(ys, axis=0)
    xms = np.concatenate(xms, axis=0)
    print(hvd.rank(), ys.shape, xms.shape, np.unique(ys, return_counts=True), np.mean(xms))

def process_results(results):
    stats = ['local_loss', 'bits_x', 'bits_y', 'pred_loss']
    assert len(stats) == results.shape[0]
    res_dict = {}
    for i in range(len(stats)):
        res_dict[stats[i]] = "{:.4f}".format(results[i])
    return res_dict

def main(hps):

    # Initialize Horovod.
    hvd.init()

    # Create tensorflow session
    sess = tensorflow_session()

    # Download and load dataset.
    tf.set_random_seed(hvd.rank() + hvd.size() * hps.seed)
    np.random.seed(hvd.rank() + hvd.size() * hps.seed)

    # Get data and set train_its and valid_its
    train_iterator, test_iterator, data_init = get_data(hps, sess)
    hps.train_its, hps.test_its, hps.full_test_its = get_its(hps)

    if hps.check_test_iterator:
        if hvd.rank() == 0:
            print(hps)
        check_test_iterator(sess, test_iterator)
        check_test_iterator(sess, test_iterator)
        exit()

    # Create log dir
    logdirs, _print = zeus.get_logdirs(['', '_ckpt'])
    if hvd.rank() == 0:
        hps.debug_logdir = logdirs[1]

    # Create model
    import model
    model = model.model(sess, hps, train_iterator, test_iterator, data_init)

    # Initialize visualization functions
    draw_samples = init_visualizations(hps, model, logdirs)

    if hvd.rank() == 0:
        _print(hps)
        _print('Starting training. Logging to', logdirs[0])
        _print('epoch n_processed n_images pps dtrain dtest dsample dtot train_results test_results msg')

    # Train
    sess.graph.finalize()
    n_processed = 0
    n_images = 0
    train_time = 0.0
    test_error_best = 1
    test_loss_best = 999999

    tcurr = time.time()
    for epoch in range(1,hps.epochs):
        if hps.debug_init:
            if hvd.rank() == 0:
                model.save(logdirs[1] + "model_init.ckpt")
            draw_samples(epoch)
            exit()

        #if hps.restore_path:
        #    draw_class(epoch)
        #    print("Done")
        #    exit()

        t0 = time.time()

        train_results = []
        for it in range(hps.train_its):

            # Set learning rate
            # lr = hps.lr
            _warmup = min(1., n_processed / (hps.n_train * hps.epochs_warmup))
            lr = hps.lr * _warmup
            if hps.lr_scalemode == 1:
                lr *= hvd.size()
            if hps.lr_scalemode == 2:
                lr *= np.sqrt(hvd.size())

            # Run a training step synchronously.
            _t0 = time.time()
            train_results += [model.train(lr)]
            if hps.verbose and hvd.rank() == 0:
                _print(n_processed, time.time()-_t0, train_results[-1])
                sys.stdout.flush()

            n_processed += hvd.size() * hps.n_batch_train  # Images seen wrt anchor resolution
            n_images += hvd.size() * hps.local_batch_train # Actual images seen at current resolution

        train_results = np.mean(np.asarray(train_results), axis=0)
        dt = time.time() - t0
        train_time += dt
        ips = hps.train_its * hvd.size() * hps.n_batch_train / dt  # Images per second wrt anchor resolution

        if epoch < 10 or (epoch < 50 and epoch % 10 == 0) or epoch % hps.epochs_full_valid == 0:
            test_results = []
            msg = ''

            t0 = time.time()
            #model.polyak_swap()
            if epoch % hps.epochs_full_valid == 0:
                # Full validation run
                for it in range(hps.full_test_its):
                    test_results += [model.test()]
                test_results = np.mean(np.asarray(test_results), axis=0)

                if hvd.rank() == 0:
                    if test_results[0] < test_loss_best:
                        test_loss_best = test_results[0]
                        model.save(logdirs[1]+"model_best_loss.ckpt")
                        msg += ' *'
                # if test_results[-1] < test_error_best:
                #     test_error_best = test_results[-1]
                #     model.save(logdirs[1]+"model_best_accuracy.ckpt")
                #     msg += ' **'
            dtest = time.time() - t0

            # Full sample uses all machines, 1 sample per machine
            t0 = time.time()
            if epoch == 1 or epoch == 10 or epoch % hps.epochs_full_sample == 0:
                draw_samples(epoch)
            dfullsample = time.time() - t0

            if hvd.rank() == 0:
                dcurr = time.time() - tcurr
                tcurr = time.time()
                _print(epoch, n_processed, n_images, "{:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(ips, dt, dtest, dfullsample, dcurr), train_results, test_results, msg, np_precision=4)

            #model.polyak_swap()


    if hvd.rank() == 0:
        _print("Finished!")


# Get number of training and validation iterations
def get_its(hps):
    # These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
    train_its = int(np.ceil(hps.n_train / (hps.n_batch_train * hvd.size())))
    test_its = int(np.ceil(hps.n_test / (hps.n_batch_train * hvd.size())))
    train_epoch = train_its * hps.n_batch_train * hvd.size()

    # Do a full validation run
    if hvd.rank() == 0:
        print(hps.n_test, hps.local_batch_test, hvd.size())
    assert hps.n_test % (hps.local_batch_test * hvd.size()) == 0
    full_test_its = hps.n_test // (hps.local_batch_test * hvd.size())

    if hvd.rank() == 0:
        print("Train epoch size: " + str(train_epoch))
    return train_its, test_its, full_test_its

'''
Create tensorflow session with horovod
'''
def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())  # Pin GPU to local rank (one GPU per process)
    sess = tf.Session(config=config)
    return sess

if __name__ == "__main__":

    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--debug", action='store_true', help="Debug mode")
    parser.add_argument("--debug_init", action='store_true', help="Debug mode")
    parser.add_argument('--restore_path', type=str, default='', help="Location of checkpoint to restore")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='imagenet', help="Problem (mnist/cifar10/imagenet")
    parser.add_argument("--category", type=str, default='', help="LSUN category")
    parser.add_argument("--data_dir", type=str, default="", help="Location of data")
    parser.add_argument("--dal", type=int, default=1, help="Data augmentation level: 0=None, 1=Standard, 2=Extra")

    # New dataloader params
    # parser.add_argument("--version", type = int, default=6)
    parser.add_argument("--check_test_iterator", action='store_true', help="Debug mode")
    parser.add_argument("--fmap", type=int, default=1, help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16, help="# Threads for parallel map")

    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int, default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int, default=-1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int, default=50, help="Minibatch size")
    parser.add_argument("--n_batch_init", type=int, default=256, help="Minibatch size for init")
    parser.add_argument("--optimizer", type=str, default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--lr_scalemode", type=int, default=0, help="Type of learning rate scaling. 0=none, 1=linear, 2=sqrt.")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1, help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--beta3", type=float, default=1., help="Adam beta3")
    parser.add_argument("--epochs", type=int, default=1000000, help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int, default=50, help="Epochs between valid")
    parser.add_argument("--gradient_checkpointing", type=int, default=1, help="Use memory saving gradients")
    parser.add_argument("--shift", type=int, default=1, help="Shift rank to resolution_id for better init")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int, default=-1, help="Image size")
    parser.add_argument("--anchor_size", type=int, default=32, help = "Anchor size for batches")
    parser.add_argument("--width", type=int, default=512, help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=48, help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00, help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8, help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=3, help="Number of levels")

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=1, help="minibatch size for sample")
    parser.add_argument("--epochs_full_sample", type=int, default=50, help="Epochs between full scale sample")
    parser.add_argument("--eps_beta", type=float, default=0.95, help="Eps beta for samples")

    # Ablation
    parser.add_argument("--learntop", action="store_true", help="Use y conditioning")
    parser.add_argument("--ycond", action="store_true", help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=0, help="Seed for ablation")
    parser.add_argument("--flow_permutation", type=int, default=2, help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--flow_coupling", type=int, default=0, help="Coupling type: 0=additive, 1=affine")
    parser.add_argument("--extra_invertible", type=int, default=0, help="Whether to put and extra inver 1x1 conv")


    #parser.add_argument("--sampling", action='store_true', help="Only sampling")
    hps = parser.parse_args() # So error if typo
    main(hps)
