    #ch norm model 4
'''
From :https://github.com/LeoTungAnh/CNN-CIFAR-100/blob/main/CNN_models.ipynb

'''
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten, SpatialDropout2D

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal, Constant, HeNormal
from tensorflow.keras import regularizers
from geo_init_new.geometric_initialization_heuristic import GeometricInit3x3 

import sys
sys.path.append("..")    
from .layers.geometric_confusion import GConfusion as GC
def batchnorm_cnn(k_init = 'he_normal'):

    model = Sequential()
    
    model.add(Conv2D(256,(3,3),padding='same',input_shape=(32,32,3), kernel_initializer = GeometricInit3x3(rho=0.7, beta=0.8) ) )
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256,(3,3),padding='same',  kernel_initializer = GeometricInit3x3(rho=0.7, beta=0.8) ))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    
    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = GeometricInit3x3(rho=0.6, beta=0.75) ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = GeometricInit3x3(rho=0.6, beta=0.75) ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = GeometricInit3x3(rho=0.5, beta=0.7) ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = GeometricInit3x3(rho=0.5, beta=0.7) ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = GeometricInit3x3(rho=0.4, beta=2/3) ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = GeometricInit3x3(rho=0.4, beta=2/3) ))
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
    model.add(Dense(100,activation=None,  kernel_initializer=HeNormal(seed=5)))
    model.add(Activation('softmax'))

    return model        