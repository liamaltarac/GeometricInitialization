import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K  


from keras import Input
import math
from keras.layers import InputLayer, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten

from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, MaxPool1D
from tensorflow.keras.initializers import RandomNormal, Constant, HeNormal


import tensorflow_addons as tfa
import numpy as np
import sys
from tensorflow.keras.losses import Loss
import os

class Cross_Entropy_Objective(Loss):
    def __init__(self, num_classes):
        super(Cross_Entropy_Objective, self).__init__()

        self.flat =   Flatten()

        self.dense = Dense(num_classes, activation='softmax', trainable = False, kernel_initializer=RandomNormal(stddev=tf.math.sqrt(1/num_classes)))
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy()
    #@tf.function()
    def __call__(self, y_true, y_pred, sample_weight=None):
        x = self.flat(y_pred)
        y_pred =  self.dense(x)
        ce = self.scce(tf.expand_dims(y_true, axis=-1) , y_pred)

        self.acc.update_state(tf.expand_dims(y_true, axis=-1) , y_pred)
        tf.print("Accuracy : ", self.acc.result()  )

        return(ce)
    

class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, initial_learning_rate,
            decay_steps,
            alpha,
            warmup_target,
            warmup_steps):
    
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps

    def warmup_learning_rate(self, step):
        completed_fraction = step / self.warmup_steps
        total_delta = self.warmup_target - self.initial_learning_rate
        return completed_fraction * total_delta
    
    def decayed_learning_rate(self, step):
        step = tf.math.minimum(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.warmup_target * decayed
    
    def __call__(self, step):

        return tf.cond(
            step < self.warmup_steps,
            true_fn=lambda: self.warmup_learning_rate(step), 
            false_fn=lambda: self.decayed_learning_rate(step)
        )


if __name__ == '__main__':


    sys.path.append("..")    

    from .models.batch_norm_rot import batchnorm_rot_cnn as cnn
    #from .models.batch_norm_2 import batchnorm_cnn as cnn

    from .models.layers.rot_conv2d_callback import RotConv2DCallback
    from .models.losses.output_entropy import entropy
    from .models.losses.variance_loss import var_loss

    #from .models.no_batch_norm import no_batchnorm_cnn as cnn

    #from geo_init.geometric_initialization_relu import GeometricInit3x3Relu
    #from geo_init_matthew.geometric_initialization import GeometricInit3x3 as gim
    #from geo_init_matthew.geometric_initialization_with_chi_mag import GeometricInit3x3 as gim
    from geo_init_heuristic.geometric_initialization import GeometricInit3x3 as gim

    #from geo_init_liam.geometric_initialization import GeometricInit3x3 as gim
    from .callbacks.filter_layout_logger import FLL

    from tensorflow.python.client import device_lib

    #physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.__version__ )

    input_layer, feat, output = cnn()
    model = tf.keras.Model(input_layer, output)

    #print(backbone.summary())
    #model = tf.keras.Model(input_layer, output)

    num_classes = 100
    input_shape = (32, 32, 3)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    from sklearn.model_selection import train_test_split
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Configuration for creating new images
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=93)
    train_datagen.fit(X_train)


    '''import wandb
    from wandb.keras import WandbCallback

    run = wandb.init(project="new_approach", entity="geometric_init")
    wandb.run.name = '8_layer_roation_cnn__cifar100'
    wandb.config = {
    "learning_rate": '[1e-6, 1e-4]',
    'batch_size' : '64',
    'epochs' : '10',
    "initialization": 'geo heuristic init l',
    "model": '8_layer_BatchNorm_Heuristic_Liam'
    }'''

    #rot_conv_callback = RotConv2DCallback(model = model, rotation_epochs=1, rotation_lr=1)
    #layout_callback = FLL(wandb=wandb, model=backbone, layer_filter_dict={4: [1, 10, 100], 9: [1, 10, 100], 12: [1, 10, 100]})


    '''''''''

    Stage 1 : Learning the rotation parameter (backbone)

    '''''''''

    batch_size = 64
    epochs = 10

    conv_layers = []
    other_layers = []
    for l in model.layers:
        if l.__class__.__name__ == 'RotConv2D':
            l.learn_rotation()
            conv_layers.append(l)
        if l.__class__.__name__ == 'BatchNormalization':
            print(l)
            l.trainable = True

    optimizer = tf.keras.optimizers.RMSprop(1e-2) 
    '''(tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                                                boundaries=[5*len(X_train)//64], 
                                                values=[1e-2, 1e-3]
                                                 )) '''

    model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
    )

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    '''predictions = model.predict(X_train)

    conf = confusion_matrix( tf.squeeze(y_train), tf.argmax( predictions, 1 ))
    plt.matshow(conf)
    plt.savefig("my_conf_mat_before.png")'''
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)


    history = model.fit(X_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(X_validation, y_validation),
                        callbacks=[cp_callback])#, WandbCallback(), layout_callback])



    '''''''''
    Stage 2 : Learning the mlp weights + cnn weights
    '''''''''
    
    #backbone = tf.keras.Model(input_layer, feat)
    #backbone.load_weights(checkpoint_path)
    
    #backbone.evaluate(X_train[0:600], y_train[0:600], batch_size=64)


    conv_layers = []
    other_layers = []
    for l in model.layers:
        if l.__class__.__name__ == 'RotConv2D':
            l.trainable = True
            l.learn_weights()
        else:
            other_layers.append(l)
        if l.__class__.__name__ == 'BatchNormalization':
            print(l)
            l.trainable = True


    batch_size = 64
    epochs = 40


    optimizer = tf.keras.optimizers.RMSprop(tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                                                boundaries=[10*len(X_train)//64, 20*len(X_train)//64], 
                                                values=[1e-3, 1e-4, 1e-5]
                                            )) 

    model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
    )
    #layout_callback = FLL(wandb=wandb, model=backbone, layer_filter_dict={4: [1, 10, 100], 8: [1, 10, 100], 11: [1, 10, 100]})



    #layout_callback.m = backbone
    history = model.fit(X_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(X_validation, y_validation))
                        #callbacks=[WandbCallback()]) 
    
