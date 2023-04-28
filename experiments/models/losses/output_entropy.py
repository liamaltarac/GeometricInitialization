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

def entropy(y_true, y_pred):
    l =  -tf.math.reduce_sum(tf.math.multiply(y_pred, tf.math.log(tf.clip_by_value(y_pred,1e-10,1.0))), axis=-1)
    print(l)
    return(l)
    #return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`