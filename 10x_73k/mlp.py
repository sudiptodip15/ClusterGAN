import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):
    def __init__(self, x_dim=720):
        self.x_dim = x_dim
        self.name = '10x_73k/mlp/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            fc1 = tc.layers.fully_connected(
                x, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)

            fc2 = tc.layers.fully_connected(
                fc1, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc2 = leaky_relu(fc2)

            fc3 = tc.layers.fully_connected(fc2, 1,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.identity
            )
            return fc3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, z_dim = 38, x_dim = 720):
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.name = '10x_73k/mlp/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:            
            fc1 = tcl.fully_connected(
                z, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            fc2 = tcl.fully_connected(
                fc1, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc2 = leaky_relu(fc2)
            
            fc3 = tc.layers.fully_connected(
                fc2, self.x_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            return fc3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
