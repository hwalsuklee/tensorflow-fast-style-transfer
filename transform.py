# Most code in this file was borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/transform.py

import tensorflow as tf

class Transform:
    def __init__(self, mode='train'):
        if mode == 'train':
            self.reuse = None
        else:
            self.reuse = True

    def net(self, image):
        image_p = self._reflection_padding(image)
        conv1 = self._conv_layer(image_p, 32, 9, 1, name='conv1')
        conv2 = self._conv_layer(conv1, 64, 3, 2, name='conv2')
        conv3 = self._conv_layer(conv2, 128, 3, 2, name='conv3')
        resid1 = self._residual_block(conv3, 3, name='resid1')
        resid2 = self._residual_block(resid1, 3, name='resid2')
        resid3 = self._residual_block(resid2, 3, name='resid3')
        resid4 = self._residual_block(resid3, 3, name='resid4')
        resid5 = self._residual_block(resid4, 3, name='resid5')
        conv_t1 = self._conv_tranpose_layer(resid5, 64, 3, 2, name='convt1')
        conv_t2 = self._conv_tranpose_layer(conv_t1, 32, 3, 2, name='convt2')
        conv_t3 = self._conv_layer(conv_t2, 3, 9, 1, relu=False, name='convt3')
        preds = (tf.nn.tanh(conv_t3) + 1) * (255. / 2)
        return preds

    def _reflection_padding(self, net):
        return tf.pad(net,[[0, 0],[40, 40],[40, 40], [0, 0]], "REFLECT")

    def _conv_layer(self, net, num_filters, filter_size, strides, padding='SAME', relu=True, name=None):
        weights_init = self._conv_init_vars(net, num_filters, filter_size, name=name)
        strides_shape = [1, strides, strides, 1]
        net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding)
        net = self._instance_norm(net, name=name)
        if relu:
            net = tf.nn.relu(net)

        return net

    def _conv_tranpose_layer(self, net, num_filters, filter_size, strides, name=None):
        weights_init = self._conv_init_vars(net, num_filters, filter_size, transpose=True, name=name)

        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1,strides,strides,1]

        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = self._instance_norm(net, name=name)
        return tf.nn.relu(net)

    def _residual_block(self, net, filter_size=3, name=None):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        tmp = self._conv_layer(net, 128, filter_size, 1, padding='VALID', relu=True, name=name+'_1')
        return self._conv_layer(tmp, 128, filter_size, 1, padding='VALID', relu=False, name=name+'_2') + tf.slice(net, [0,2,2,0], [batch,rows-4,cols-4,channels])

    def _instance_norm(self, net, name=None):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        with tf.variable_scope(name, reuse=self.reuse):
            shift = tf.get_variable('shift', initializer=tf.zeros(var_shape), dtype=tf.float32)
            scale = tf.get_variable('scale', initializer=tf.ones(var_shape), dtype=tf.float32)
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
        return scale * normalized + shift

    def _conv_init_vars(self, net, out_channels, filter_size, transpose=False, name=None):
        _, rows, cols, in_channels = [i.value for i in net.get_shape()]
        if not transpose:
            weights_shape = [filter_size, filter_size, in_channels, out_channels]
        else:
            weights_shape = [filter_size, filter_size, out_channels, in_channels]
        with tf.variable_scope(name, reuse=self.reuse):
            weights_init = tf.get_variable('weight', shape=weights_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=tf.float32)
        return weights_init