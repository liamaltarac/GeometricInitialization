import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K  

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
        self.prob_mat.assign(prob_mat)

        #self.centroids.close()
        #tf.print(angle)

        #tf.print(norms)

        return(-MI)
    
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
    backbone = tf.keras.Model(input_layer, feat)
    print(backbone.summary())
    model = tf.keras.Model(input_layer, output)

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
    #layout_callback = FLL(wandb=wandb, model=backbone, layer_filter_dict={3: [1, 10, 100], 7: [1, 10, 100], 10: [1, 10, 100], 15: [1, 10, 100]})
    

    '''''''''

    Stage 1 : Learning the rotation parameter

    '''''''''

    batch_size = 64
    epochs = 1


    conv_layers = []
    other_layers = []
    for l in backbone.layers:
        if l.__class__.__name__ == 'RotConv2D':
            l._train_r = True
            l._train_w = False
            #l.trainable = True
            conv_layers.append(l)
        if  l.__class__.__name__ == 'Dense':
            l.trainable = False


    '''optimizers = [
    tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    tf.keras.optimizers.RMSprop(learning_rate=1e-6)
    ]
    optimizers_and_layers = [(optimizers[0], conv_layers), (optimizers[1], other_layers)]'''

    initial_learning_rate = 1e-3
    final_learning_rate = 1e-5
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
    steps_per_epoch = int(X_train.shape[0]/batch_size)
    print("Steps per Epoch ", steps_per_epoch)

    sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            150,
            t_mul=1.0,
            m_mul=0.9,
            alpha=final_learning_rate,
            )
    
    optimizer =     tf.keras.optimizers.RMSprop(1e-2) #tfa.optimizers.MultiOptimizer(optimizers_and_layers)


    objective = Mutual_Information_Objective(512, num_classes= 100, conv_features=True)

    backbone.compile(
            optimizer=optimizer,
            loss = objective # keras.losses.SparseCategoricalCrossentropy(from_logits=False),
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


    history = backbone.fit(X_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        #validation_data=(X_validation, y_validation),
                        callbacks=[cp_callback])

    predictions = backbone.evaluate(X_validation, y_validation)



    m = objective.get_prob_mat()
    plt.matshow(m)
    plt.savefig("my_conf_mat_after.png")

    model.load_weights(checkpoint_path)

    '''''''''

    Stage 2 : Learning the mlp weights 

    '''''''''
    #layout_callback = FLL(wandb=wandb, model=model, layer_filter_dict={3: [1, 10, 100], 7: [1, 10, 100], 10: [1, 10, 100], 15: [1, 10, 100]})

    batch_size = 64
    epochs = 2

    #
    conv_layers = []
    other_layers = []
    for l in model.layers:
        if l.__class__.__name__ == 'RotConv2D':
            l._train_r = False
            l._train_w = True
            l.trainable = False
            conv_layers.append(l)
        if  l.__class__.__name__ == 'Dense':
            l.trainable = True
            other_layers.append(l)

    optimizer =   tf.keras.optimizers.RMSprop(learning_rate=1e-2)  #-4 best


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
                        validation_data=(X_validation, y_validation))
    

    '''log_gradients   = (True), 
    log_weights     = (True),
    training_data   = (X_train, y_train),
    validation_data = (X_validation, y_validation)
    input_type = "images",
    output_type = "label",
    )])'''


    '''''''''

    Stage 3 : Learning all the weights 

    '''''''''
    #layout_callback = FLL(wandb=wandb, model=model, layer_filter_dict={3: [1, 10, 100], 7: [1, 10, 100], 10: [1, 10, 100], 15: [1, 10, 100]})

    batch_size = 64
    epochs = 60

    #
    conv_layers = []
    other_layers = []
    for l in model.layers:
        if l.__class__.__name__ == 'RotConv2D':
            l._train_r = False
            l._train_w = True
            l.trainable = True
            conv_layers.append(l)
        if  l.__class__.__name__ == 'Dense':
            l.trainable = True
            other_layers.append(l)


    optimizers = [
    tf.keras.optimizers.RMSprop(learning_rate=1e-4),  #-4 best
    tf.keras.optimizers.RMSprop(learning_rate=1e-3)  #-3 best
    ]
    optimizers_and_layers = [(optimizers[0], conv_layers), (optimizers[1], other_layers)]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)


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
                        validation_data=(X_validation, y_validation))
    
    wandb.finish()

    '''log_gradients   = (True), 
    log_weights     = (True),
    training_data   = (X_train, y_train),
    validation_data = (X_validation, y_validation)
    input_type = "images",
    output_type = "label",
    )])'''