import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K  


from keras import Input
import math
from keras.layers import InputLayer
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

class Mutual_Information_Objective(Loss):
    def __init__(self, num_features, num_classes=10, conv_features=True):

        super(Mutual_Information_Objective, self).__init__()
        self.n_features = num_features
        self.n_classes = num_classes
        #self.prob = input_is_probability
        self.conv_features =  conv_features  #Is training using the output of convolution layer 

        self.centroid_table= tf.Variable(tf.zeros([ self.n_classes,  self.n_features]), trainable=False)
        self.prob_mat= tf.Variable(tf.zeros([ self.n_classes, self.n_features]), trainable=False)


        self.num_features =  0
        tf.print("Starting")


    def get_prob_mat(self):
        return self.prob_mat.value()

    @tf.function()
    def __call__(self, y_true, y_pred, sample_weight=None):
        #tf.print(y_true)
        #tf.print(y_pred.shape)
        #norms = tf.zeros([0, 1])
        #centroids = tf.zeros([0, 100])



        #for label in tf.range(self.n_classes):
        def body(i, label, centroids):
            #Find all latent codes in batch belonging a specific class
            idx = tf.where(y_true == [label])[:,0]
            #tf.print(label)

            if tf.equal(tf.size(idx), 0) == False:
                #tf.print(idx)
                gathered_vectors = tf.gather(y_pred,  idx)

                #tf.io.write_file("Math_Experiments/dot.csv", tf.strings.as_string(prob_mat))
                centroid = tf.reduce_mean(gathered_vectors, axis=0)
                #tf.io.write_file("Math_Experiments/dot.csv", tf.strings.as_string(centroid))
                if self.conv_features:
                    #centroid = tf.keras.layers.GlobalAveragePooling2D()(tf.expand_dims(centroid, axis=0))
                    centroid = tf.reshape(centroid, [self.n_features])
                    #tf.io.write_file("Math_Experiments/dot.csv", tf.strings.as_string(centroid))

                    #centroid = K.clip(centroid, -1, 1, )

                    '''scale = centroid.shape[-1]//self.n_classes
                    centroid = tf.keras.layers.AveragePooling1D(pool_size=scale,
                                strides=scale, padding='valid')(tf.expand_dims(centroid, axis=-1))'''
                
                #centroid = tf.reshape(centroid, [self.n_classes])

                #if self.prob is False:
                #    centroid = tf.nn.softmax(centroid)
                

                centroid_sum = tf.math.reduce_sum(centroid) + K.epsilon()
                #centroid_sum = K.clip(centroid_sum , K.epsilon(), centroid_sum )

                centroid = centroid / centroid_sum # ensure that sum=1

                #if prob is False:
                #    centroid = tf.nn.softmax(centroid)
                centroids = centroids.write(label, centroid)
            
            label += 1
            return i, label, centroids



        centroids = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=0, 
                                        dynamic_size=True, name="centroids",  clear_after_read=False) 
        

        centroids = centroids.unstack(self.centroid_table)
        i, _, centroids = tf.while_loop(lambda i, label, *_: tf.less(label, self.n_classes), body,[0,0, centroids] )

        centroid_table =  centroids.stack()
        if tf.keras.backend.learning_phase():
            tf.print("Learn")
        self.centroid_table.assign(centroid_table)
        prob_mat =(centroid_table/ self.n_classes ) + K.epsilon()
        #prob_mat = K.clip(centroid_table+ / self.n_classes , K.epsilon(), 1, ) #Ensure that marginals sum to 1 and that the zeros are not actually 0 (prevents nan)
        #prob_mat = self.centroid_table / self.n_classes 
        prob_mat = tf.reshape(prob_mat, [self.n_classes, self.n_features])

        px = tf.expand_dims(tf.math.reduce_sum(prob_mat, axis=1), axis=0)
        px = tf.reshape(px, [1,  self.n_classes])
        py = tf.expand_dims(tf.math.reduce_sum(prob_mat, axis=0), axis=0)
        py = tf.reshape(py, [1,  self.n_features])

        #tf.print(py)
        px_py = tf.linalg.matmul(px , py, transpose_a=True)  #Px * Py

        MI = tf.math.reduce_sum(tf.math.multiply(prob_mat, tf.math.log(tf.math.divide(prob_mat, px_py)))) #axis=0)
        #tf.io.write_file("dot.csv", tf.strings.as_string(MI))

        #centroids.close()
        #if tf.keras.backend.learning_phase():
        self.prob_mat.assign(prob_mat)

        #self.centroids.close()
        #tf.print(angle)

        #tf.print(norms)

        return(-MI)
    

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

    input_layer, output = cnn()
    #backbone = tf.keras.Model(input_layer, feat)

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


    import wandb
    from wandb.keras import WandbCallback

    run = wandb.init(project="new_approach", entity="geometric_init")
    wandb.run.name = '8_layer_roation_cnn__cifar100'
    wandb.config = {
    "learning_rate": '[1e-6, 1e-4]',
    'batch_size' : '64',
    'epochs' : '10',
    "initialization": 'geo heuristic init l',
    "model": '8_layer_BatchNorm_Heuristic_Liam'
    }

    #rot_conv_callback = RotConv2DCallback(model = model, rotation_epochs=1, rotation_lr=1)
    #layout_callback = FLL(wandb=wandb, model=backbone, layer_filter_dict={4: [1, 10, 100], 9: [1, 10, 100], 12: [1, 10, 100]})






    '''''''''
    Stage 2 : Learning the mlp weights + rotation
    '''''''''
    
    backbone = tf.keras.Model(input_layer, output)
    #backbone.load_weights(checkpoint_path)
    backbone.trainable = True

    #model.load_weights(checkpoint_path)
    inputs = keras.Input(shape=(32,32,3))
    x = backbone(inputs, training=False)
    x=Flatten()(x)
    x=Dense(1024, kernel_initializer=HeNormal(seed=5))(x)   #1024
    x=Activation('relu')(x)
    x=Dropout(0.2)(x)
    x=BatchNormalization(momentum=0.95, 
                epsilon=0.005,
                beta_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                gamma_initializer=Constant(value=0.9))(x)
    outputs=Dense(100,activation='softmax',  kernel_initializer=HeNormal(seed=5))(x)
    model = tf.keras.Model(inputs, outputs)

    conv_layers = []
    other_layers = []
    for l in backbone.layers:
        if l.__class__.__name__ == 'RotConv2D':
            l.learn_rotation()
            conv_layers.append(l)


    batch_size = 64
    epochs = 5




    initial_learning_rate = 1e-2
    final_learning_rate = 1e-4
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
    steps_per_epoch = int(X_train.shape[0]/batch_size)
    print("Steps per Epoch ", steps_per_epoch)

    sched_mlp = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            steps_per_epoch*1,
            t_mul=1.5,
            m_mul=0.5,
            alpha=1e-6,
    )


    #optimizer =   tf.keras.optimizers.Adam(learning_rate=sched)  #-4 best



    optimizers = [
        tf.keras.optimizers.Adam(1e-3),
        tf.keras.optimizers.Adam(1e-4)
    ]
    optimizers_and_layers = [(optimizers[0], model.layers[0:2]), (optimizers[1], model.layers[2:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

    #print(backbone.summary())
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
                        validation_data=(X_train, y_train))
                        #callbacks=[WandbCallback()]) 
    


    '''''''''

    Stage 3 : Learning all the weights 

    '''''''''
    #layout_callback = FLL(wandb=wandb, model=model, layer_filter_dict={3: [1, 10, 100], 7: [1, 10, 100], 10: [1, 10, 100], 15: [1, 10, 100]})

    batch_size = 64
    epochs = 60

    backbone.trainable = True


    conv_layers = []
    other_layers = []
    for l in backbone.layers:
        if l.__class__.__name__ == 'RotConv2D':
            l.learn_weights()
            conv_layers.append(l)
        if l.__class__.__name__ == 'BatchNormalization':
            print(l)
            l.trainable = False



    initial_learning_rate = 1e-4
    final_learning_rate = 1e-6
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
    steps_per_epoch = int(X_train.shape[0]/batch_size)
    print("Steps per Epoch ", steps_per_epoch)

    sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            steps_per_epoch*2,
            t_mul=1.5,
            m_mul=0.9,
            alpha=final_learning_rate,
    )
    optimizer =     tf.keras.optimizers.Adam(learning_rate=sched)  #-4 best


    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account
    model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
    )


    history = model.fit(X_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(X_validation, y_validation),
                        callbacks=[WandbCallback()])
    
    wandb.finish()

    '''log_gradients   = (True), 
    log_weights     = (True),
    training_data   = (X_train, y_train),
    validation_data = (X_validation, y_validation)
    input_type = "images",
    output_type = "label",
    )])'''