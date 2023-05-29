    #ch norm model 4
'''
From :https://github.com/LeoTungAnh/CNN-CIFAR-100/blob/main/CNN_models.ipynb

'''
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal, Constant, HeNormal
from .layers.rot_conv2d import RotConv2D
from .layers.rot_conv2d_transpose import RotConvTranspose2D

def rot_autoencoder():

    model = Sequential()
    model.add(InputLayer(input_shape=(32,32,3)))
    model.add(RotConv2D(256,padding='SAME'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RotConv2D(256,padding='SAME')) #3 
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))  #16x16
    #model.add(Dropout(0.2))

    model.add(RotConv2D(512,padding='SAME')) #6
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RotConv2D(512,padding='SAME')) #9
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))  #8x8
    #model.add(Dropout(0.2))
    
    
    model.add(RotConv2D(512,padding='SAME')) #12  8x8
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RotConv2D(512,padding='SAME')) #15
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))  #4x4




    model.add(RotConvTranspose2D(512,padding='SAME')) #18
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RotConv2D(512,padding='SAME')) #12
    model.add(Activation('relu'))
    
    model.add(RotConvTranspose2D(512,padding='SAME')) #23
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RotConv2D(256,padding='SAME')) #16
    model.add(Activation('relu'))

    model.add(RotConvTranspose2D(256,padding='SAME')) #18
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RotConv2D(3,padding='SAME')) #20
    model.add(Activation('sigmoid'))
    return model        