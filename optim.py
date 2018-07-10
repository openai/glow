import tensorflow as tf
import tfops as Z
import horovod.tensorflow as hvd

# Optimizers

'''
Polyak averaging op
'''


def polyak(params, beta):
    #params = tf.trainable_variables()
    ema = tf.train.ExponentialMovingAverage(decay=beta, zero_debias=True)
    avg_op = tf.group(ema.apply(params))
    # Swapping op
    updates = []
    for i in range(len(params)):
        p = params[i]
        avg = ema.average(p)
        tmp = 0. + avg * 1.
        with tf.control_dependencies([tmp]):
            update1 = avg.assign(p)
            with tf.control_dependencies([update1]):
                update2 = p.assign(tmp)
                updates += [update1, update2]
    swap_op = tf.group(*updates)
    return avg_op, swap_op, ema


def adam(params, cost_or_grads, alpha=3e-4, hps=None, epsilon=1e-8):
    updates = []
    if type(cost_or_grads) is not list:
        gs = tf.gradients(cost_or_grads, params)
    else:
        gs = cost_or_grads

    beta2 = 1-1./(hps.train_its*hps.polyak_epochs)

    # all-reduce
    grads = [Z.allreduce_mean(g) for g in gs]

    t = tf.Variable(1., 'adam_t')
    alpha_t = alpha * tf.sqrt((1. - tf.pow(beta2, t))) / \
        (1. - tf.pow(hps.beta1, t))
    updates.append(t.assign_add(1))

    for w, g in zip(params, grads):
        mom2 = tf.Variable(tf.zeros(w.get_shape()), w.name + '_adam_m2')
        if hps.beta1 > 0:
            mom1 = tf.Variable(tf.zeros(w.get_shape()), w.name + '_adam_m1')
            mom1_new = hps.beta1 * mom1 + (1. - hps.beta1) * g
            updates.append(mom1.assign(mom1_new))
        else:
            mom1_new = g
        m2_new = beta2 * mom2 + (1. - beta2) * tf.square(g)
        delta_t = mom1_new / (tf.sqrt(m2_new) + epsilon)
        w_new = hps.weight_decay * w - alpha_t * delta_t
        updates.append(mom2.assign(m2_new))
        updates.append(w.assign(w_new))

    # Polyak averaging
    polyak_avg_op, polyak_swap_op, ema = polyak(params, beta2)
    train_op = tf.group(polyak_avg_op, *updates)
    return train_op, polyak_swap_op, ema


'''
Adam optimizer
Version whose learning rate could, in theory, be scaled linearly (like SGD+momentum).
(It doesn't seem to work yet, though.)
'''


def adam2(params, cost_or_grads, alpha=3e-4, hps=None, epsilon=1e-8):
    updates = []
    if type(cost_or_grads) is not list:
        gs = tf.gradients(cost_or_grads, params)
    else:
        gs = cost_or_grads

    beta2 = 1-1./(hps.train_its*hps.polyak_epochs)

    # all-reduce
    grads1 = [Z.allreduce_mean(g) for g in gs]
    grads2 = [Z.allreduce_mean(g**2) for g in gs]

    t = tf.Variable(1., 'adam_t')
    alpha_t = alpha * tf.sqrt((1. - tf.pow(beta2, t))) / \
        (1. - tf.pow(hps.beta1, t))
    updates.append(t.assign_add(1))

    for w, g1, g2 in zip(params, grads1, grads2):
        mom2 = tf.Variable(tf.zeros(w.get_shape()), w.name + '_adam_m2')
        if hps.beta1 > 0:
            mom1 = tf.Variable(tf.zeros(w.get_shape()), w.name + '_adam_m1')
            mom1_new = hps.beta1 * mom1 + (1. - hps.beta1) * g1
            updates.append(mom1.assign(mom1_new))
        else:
            mom1_new = g1
        m2_new = beta2 * mom2 + (1. - beta2) * g2
        delta_t = mom1_new / (tf.sqrt(m2_new) + epsilon)
        w_new = hps.weight_decay * w - alpha_t * delta_t
        updates.append(mom2.assign(m2_new))
        updates.append(w.assign(w_new))

    # Polyak averaging
    polyak_avg_op, polyak_swap_op, ema = polyak(params, beta2)
    train_op = tf.group(polyak_avg_op, *updates)
    return train_op, polyak_swap_op, ema


'''
Adam optimizer
Version whose learning rate could, in theory, be scaled linearly (like SGD+momentum).
It doesn't seem to work though.
'''


def adam2_old(params, cost_or_grads, lr=3e-4, mom1=0.9, mom2=0.999, epsilon=1e-8):
    updates = []
    if type(cost_or_grads) is not list:
        gs = tf.gradients(cost_or_grads, params)
    else:
        gs = cost_or_grads

    # all-reduce
    grads1 = [Z.allreduce_mean(g) for g in gs]
    grads2 = [Z.allreduce_mean(tf.square(g)) for g in gs]
    mom2 = tf.maximum(0., 1. - (hvd.size() * (1 - mom2)))

    t = tf.Variable(1., 'adam_t')
    lr_t = lr * tf.sqrt((1. - tf.pow(mom2, t))) / (1. - tf.pow(mom1, t))
    updates.append(t.assign_add(1))

    for p, g1, g2 in zip(params, grads1, grads2):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1 > 0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1 * v + (1. - mom1) * g1
            updates.append(v.assign(v_t))
        else:
            v_t = g1
        mg_t = mom2 * mg + (1. - mom2) * g2
        delta_t = v_t / (tf.sqrt(mg_t) + epsilon)
        p_t = p - lr_t * delta_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    return tf.group(*updates)


def adamax(params, cost_or_grads, alpha=3e-4, hps=None, epsilon=1e-8):
    updates = []
    if type(cost_or_grads) is not list:
        gs = tf.gradients(cost_or_grads, params)
    else:
        gs = cost_or_grads

    beta2 = 1-1./(hps.train_its*hps.polyak_epochs)

    # all-reduce
    grads = [Z.allreduce_mean(g) for g in gs]

    t = tf.Variable(1., 'adam_t')
    alpha_t = alpha * tf.sqrt((1. - tf.pow(beta2, t))) / \
        (1. - tf.pow(hps.beta1, t))
    updates.append(t.assign_add(1))

    for w, g in zip(params, grads):
        mom2 = tf.Variable(tf.zeros(w.get_shape()), w.name + '_adam_m2')
        if hps.beta1 > 0:
            mom1 = tf.Variable(tf.zeros(w.get_shape()), w.name + '_adam_m1')
            mom1_new = hps.beta1 * mom1 + (1. - hps.beta1) * g
            updates.append(mom1.assign(mom1_new))
        else:
            mom1_new = g
        m2_new = tf.maximum(beta2 * mom2, abs(g))
        delta_t = mom1_new / (m2_new + epsilon)
        w_new = hps.weight_decay * w - alpha_t * delta_t
        updates.append(mom2.assign(m2_new))
        updates.append(w.assign(w_new))

    # Polyak averaging
    polyak_avg_op, polyak_swap_op, ema = polyak(params, beta2)
    train_op = tf.group(polyak_avg_op, *updates)
    return train_op, polyak_swap_op, ema


def adam(params, cost_or_grads, alpha=3e-4, hps=None, epsilon=1e-8):
    updates = []
    if type(cost_or_grads) is not list:
        gs = tf.gradients(cost_or_grads, params)
    else:
        gs = cost_or_grads

    beta2 = 1-1./(hps.train_its*hps.polyak_epochs)

    # all-reduce
    grads = [Z.allreduce_mean(g) for g in gs]

    t = tf.Variable(1., 'adam_t')
    alpha_t = alpha * tf.sqrt((1. - tf.pow(beta2, t))) / \
        (1. - tf.pow(hps.beta1, t))
    updates.append(t.assign_add(1))

    for w, g in zip(params, grads):
        mom2 = tf.Variable(tf.zeros(w.get_shape()), w.name + '_adam_m2')
        if hps.beta1 > 0:
            mom1 = tf.Variable(tf.zeros(w.get_shape()), w.name + '_adam_m1')
            mom1_new = hps.beta1 * mom1 + (1. - hps.beta1) * g
            updates.append(mom1.assign(mom1_new))
        else:
            mom1_new = g
        m2_new = beta2 * mom2 + (1. - beta2) * tf.square(g)
        delta_t = mom1_new / (tf.sqrt(m2_new) + epsilon)
        w_new = hps.weight_decay * w - alpha_t * delta_t
        updates.append(mom2.assign(m2_new))
        updates.append(w.assign(w_new))

    # Polyak averaging
    polyak_avg_op, polyak_swap_op, ema = polyak(params, beta2)
    train_op = tf.group(polyak_avg_op, *updates)
    return train_op, polyak_swap_op, ema
