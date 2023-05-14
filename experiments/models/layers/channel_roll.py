import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Concatenate , Add, Dot, Activation, Lambda
from tensorflow.compat.v2.math import mod
import sys 

class ChannelRoll(tf.keras.layers.Layer):
    def __init__(self):
        super(ChannelRoll, self).__init__(trainable= False)

    tf.function(jit_compile=False)
    def call(self, inputs, training=None):
        
        x = inputs[0]
        map =  tf.cast(inputs[1], tf.int32)
        filters =  tf.cast(x.shape[-1], tf.int32)

        #map  = tf.cast(map, tf.int64) 

        indicies = tf.cast(tf.ensure_shape(tf.linspace(tf.squeeze(map, -1), tf.squeeze(map + filters-1, -1), filters, axis=-1), x.shape) , tf.int32)
        indicies = mod(indicies, filters)
        indicies = tf.cast(indicies, tf.int64)
        out = tf.gather(x, indicies , batch_dims=-1)



        '''
        index = tf.squeeze(tf.sequence_mask(map, x.shape[-1]), axis=-2)
        index = tf.math.logical_not(index)

        out_left = tf.ragged.boolean_mask(x, index)

        index = tf.math.logical_not(index)

        out_right = tf.ragged.boolean_mask(x, index)
        out_shifted = tf.concat([out_left, out_right], axis=-1)

        out =  out_shifted.to_tensor(shape=x.shape) '''
        #out.set_shape(x.shape)
        return out  #self.activation(tf.math.add(x_a, x_s)) #, self.map
        
