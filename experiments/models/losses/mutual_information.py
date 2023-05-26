import tensorflow as tf


from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import backend 
from tensorflow.keras.layers import Input, Concatenate , Add, Dot, Activation, Lambda, BatchNormalization, LeakyReLU, Softmax, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid

from tensorflow.image import flip_up_down, flip_left_right, rot90
from tensorflow.linalg import normalize

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def mutual_information(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)


    v_pred = tf.math.reduce_mean(tf.math.reduce_variance(tf.cast(y_pred, dtype = tf.float32), axis=-1))
    
    v_all_preds = tf.math.reduce_variance(tf.math.reduce_sum(y_pred, axis=0)/tf.cast(tf.shape(y_pred)[0], dtype = tf.float32))

    print("v_pred" , v_pred.shape) 

    print("v_all_preds" , tf.math.reduce_sum(y_pred, axis=0).shape) 

    #l =  tf.math.abs(tf.math.reduce_variance(tf.cast(y_true, dtype = tf.float32)) - tf.math.reduce_variance(tf.cast(y_pred, dtype = tf.float32)))
    l = -v_all_preds

    return(l)
    #return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`