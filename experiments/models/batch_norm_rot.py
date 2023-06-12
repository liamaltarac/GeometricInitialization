    #ch norm model 4
'''
From :https://github.com/LeoTungAnh/CNN-CIFAR-100/blob/main/CNN_models.ipynb

'''
from keras.models import Sequential
from keras import Input

from keras.layers import InputLayer
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten

from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, MaxPool1D
from tensorflow.keras.initializers import RandomNormal, Constant, HeNormal
from .layers.rot_conv2d import RotConv2D
def batchnorm_rot_cnn():

    input_layer=Input(shape=(32,32,3))
    x=RotConv2D(256,padding='SAME')(input_layer)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=RotConv2D(256,padding='SAME')(x) #3
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(2,2))(x)  #16x16
    x=Dropout(0.2)(x)
        
    x=RotConv2D(512,padding='SAME')(x) #6
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=RotConv2D(512,padding='SAME')(x) #9
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(2,2))(x)  #8x8
    x=Dropout(0.2)(x)

    x=RotConv2D(512,padding='SAME')(x) #13
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=RotConv2D(512,padding='SAME')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(2,2))(x) #4x4
    x=Dropout(0.2)(x)

    x=RotConv2D(512,padding='SAME')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=RotConv2D(512,padding='SAME')(x)
    x=BatchNormalization()(x)




    feature_layer= AveragePooling2D(pool_size=(4, 4), name='avg_pool_feature')(x)
    #feature_layer = Dropout(0.2)(feature_layer)
    #feature_layer= MaxPool1D(pool_size=4, name='maxpool1D')(feature_layer)
    feature_layer= Activation('softmax')(feature_layer/0.75)
    feature_layer = Dropout(0.5)(feature_layer)


    x=Activation('relu')(x)
    x=MaxPool2D(pool_size=(2,2))(x)  #2x2x512
    x = Dropout(0.2)(x)



    x=Flatten()(x)
    x=Dense(1024, kernel_initializer=HeNormal(seed=5))(x)   #1024
    x=Activation('relu')(x)
    x=Dropout(0.2)(x)
    x=BatchNormalization(momentum=0.95, 
                epsilon=0.005,
                beta_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                gamma_initializer=Constant(value=0.9))(x)
    output=Dense(100,activation='softmax',  kernel_initializer=HeNormal(seed=5))(x)
    return (input_layer, feature_layer, output)  