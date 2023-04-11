    #ch norm model 4
'''
From :https://github.com/LeoTungAnh/CNN-CIFAR-100/blob/main/CNN_models.ipynb

'''
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal, Constant

def batchnorm_cnn_mnist(k_init = 'he_normal'):

    model = Sequential()
    
    model.add(Conv2D(256,(3,3),padding='same',input_shape=(28,28,1), kernel_initializer = k_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256,(3,3),padding='same', kernel_initializer = k_init))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = k_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = k_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = k_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = k_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = k_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512,(3,3),padding='same', kernel_initializer = k_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(momentum=0.95, 
            epsilon=0.005,
            beta_initializer=RandomNormal(mean=0.0, stddev=0.05), 
            gamma_initializer=Constant(value=0.9)))
    model.add(Dense(10,activation='softmax'))
    return model        