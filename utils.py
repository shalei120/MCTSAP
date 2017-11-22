import math
import numpy as np
import tensorflow as tf

def to_onehot(size, index):
    ret = np.zeros((size,))
    ret[index] = 1
    return ret

def from_onehot(v):
    return tf.argmax(v, 1)

def weight(name, shape, init='xavier', range=None):
    """ Initializes weight.
    :param name: Variable name
    :param shape: Tensor shape
    :param init: Init mode. xavier / normal / uniform / he (default is 'xavier')
    :param range:
    :return: Variable
    """
    initializer = tf.constant_initializer(0.0)

    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]

    if init == 'xavier':
        range = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-range, range)

    elif init == 'he':
        std = math.sqrt(2.0 / fan_in)
        initializer = tf.random_normal_initializer(stddev=std)

    elif init == 'normal':
        initializer = tf.random_normal_initializer(stddev=0.1)

    elif init == 'uniform':
        if range is None:
            raise ValueError("range must not be None if uniform init is used.")
        initializer = tf.random_uniform_initializer(-range, range)
    
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var

