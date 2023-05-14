import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Concatenate , Add, Dot, Activation, Lambda
from tensorflow.compat.v2.math import mod
import sys 

class GConfusion(tf.keras.layers.Layer):
    def __init__(self, prob, max, patch_size=None):
        super(GConfusion, self).__init__(trainable= False)

        self.patch_size = patch_size
        self.prob = prob
        self.max  = max


    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "stddev": self.stddev
        })
        return config

    tf.function(jit_compile=False)
    def call(self, inputs, training=None):
        
        x = inputs
        if self.patch_size is None:
            self.patch_size = x.shape[1]
        filters =  tf.cast(x.shape[-1], tf.int32)

        map_shape = [tf.shape(x)[0], int(x.shape[1]/self.patch_size), int(x.shape[2]/self.patch_size), 1]
        mask_shape = [tf.shape(x)[0], int(x.shape[1]), int(x.shape[2]), 1]

        mask =  tf.cast(tf.random.uniform(
            shape = mask_shape,
            minval = 0.0,
            maxval = 1.0
        ) > self.prob , tf.int32)
        
        map =  tf.random.uniform(
            shape = map_shape,
            minval = 0.0,
            maxval = self.max,
        )
        map  = tf.cast(map, tf.int32) 

        map = tf.repeat(map, repeats = self.patch_size, axis=1)
        map = tf.repeat(map, repeats = self.patch_size, axis=2)
        map *= mask


        indicies = tf.cast(tf.ensure_shape(tf.linspace(tf.squeeze(map, -1), tf.squeeze(map + filters-1, -1), filters, axis=-1), x.shape) , tf.int32)
        indicies = mod(indicies, filters)
        indicies = tf.cast(indicies, tf.int32)
        out = tf.gather(x, indicies , batch_dims=-1)
        
        return K.in_train_phase(out, inputs,
                                    training=training)


        '''
        index = tf.squeeze(tf.sequence_mask(map, x.shape[-1]), axis=-2)
        index = tf.math.logical_not(index)

        out_left = tf.ragged.boolean_mask(x, index)

        index = tf.math.logical_not(index)

        out_right = tf.ragged.boolean_mask(x, index)
        out_shifted = tf.concat([out_left, out_right], axis=-1)

        out =  out_shifted.to_tensor(shape=x.shape) '''
        #out.set_shape(x.shape)
        #return out  #self.activation(tf.math.add(x_a, x_s)) #, self.map
        
