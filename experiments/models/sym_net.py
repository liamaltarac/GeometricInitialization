    #ch norm model 4
'''
From :https://github.com/LeoTungAnh/CNN-CIFAR-100/blob/main/CNN_models.ipynb

'''
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers import InputLayer

from keras.layers.core import Dense,Activation,Dropout,Flatten
from tensorflow.keras.initializers import RandomNormal, Constant, HeNormal

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal, Constant
from .layers.sym_conv2d import SymConv2D

def batchnorm_sym_cnn():

    model = Sequential()
    model.add(InputLayer(input_shape=(32,32,3)))
    model.add(SymConv2D(256,padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SymConv2D(256,padding='SAME'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(SymConv2D(256,padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SymConv2D(256,padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(SymConv2D(256,padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SymConv2D(256,padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(SymConv2D(256,padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SymConv2D(256,padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=HeNormal(seed=5)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(momentum=0.95, 
            epsilon=0.005,
            beta_initializer=RandomNormal(mean=0.0, stddev=0.05), 
            gamma_initializer=Constant(value=0.9)))
    model.add(Dense(10,activation='softmax',  kernel_initializer=HeNormal(seed=5)))
    return model        