import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K  

import tensorflow_addons as tfa
import numpy as np
import sys
from tensorflow.keras.losses import Loss


if __name__ == '__main__':


    sys.path.append("..")    

    from .models.batch_norm_rot import batchnorm_rot_cnn as cnn
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
    #layout_callback = FLL(wandb=wandb, model=backbone, layer_filter_dict={3: [1, 10, 100], 7: [1, 10, 100], 10: [1, 10, 100], 15: [1, 10, 100]})
    

    '''''''''

    Stage 1 : Learning the mlp + cnn weights

    '''''''''

    batch_size = 64
    epochs = 1


    conv_layers = []
    other_layers = []
    for l in model.layers:
        if l.__class__.__name__ == 'RotConv2D':
            l._train_r = True
            l._train_w = False
            conv_layers.append(l)
            
        if  l.__class__.__name__ == 'Dense':
            l.trainable = True
            other_layers.append(l)
    optimizers = [
    tf.keras.optimizers.RMSprop(learning_rate=1e-1),
    tf.keras.optimizers.RMSprop(learning_rate=1e-3)
    ]
    optimizers_and_layers = [(optimizers[0], conv_layers), (optimizers[1], other_layers)]
    optimizer =  tfa.optimizers.MultiOptimizer(optimizers_and_layers)



    model.compile(
            optimizer=optimizer,
            loss =  keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
    )
   

    history = model.fit(X_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(X_validation, y_validation),)



    '''''''''

    Stage 2 : Learning the rotation 

    '''''''''
    #layout_callback = FLL(wandb=wandb, model=model, layer_filter_dict={3: [1, 10, 100], 7: [1, 10, 100], 10: [1, 10, 100], 15: [1, 10, 100]})

    batch_size = 64
    epochs = 54

    #
    conv_layers = []
    other_layers = []
    for l in model.layers:
        if l.__class__.__name__ == 'RotConv2D':
            l._train_r = False
            l._train_w = True
            conv_layers.append(l)
        if  l.__class__.__name__ == 'Dense':
            l.trainable = True
            other_layers.append(l)


    optimizers = [
    tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    tf.keras.optimizers.RMSprop(learning_rate=1e-4)
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