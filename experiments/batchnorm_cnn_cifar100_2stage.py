import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.initializers import RandomNormal, Constant, HeNormal, GlorotNormal

import sys

class LRLogger(tf.keras.callbacks.Callback):
    def __init__(self, optimizer):
      super(LRLogger, self).__init__()
      self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs):
      lr = self.optimizer.learning_rate(self.optimizer.iterations)
      wandb.log({"lr": lr}, commit=False)


if __name__ == '__main__':


    sys.path.append("..")    

    from .models.batch_norm import batchnorm_cnn as cnn
    #from .models.no_batch_norm import no_batchnorm_cnn as cnn

    #from geo_init.geometric_initialization_relu import GeometricInit3x3Relu
    #from geo_init_matthew.geometric_initialization import GeometricInit3x3 as gim
    #from geo_init_matthew.geometric_initialization_with_chi_mag import GeometricInit3x3 as gim
    from geo_init_heuristic.geometric_initialization import GeometricInit3x3 as gim

    #from geo_init_liam.geometric_initialization import GeometricInit3x3 as gim
    from .callbacks.filter_layout_logger import FLL

    from tensorflow.python.client import device_lib
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.__version__ )

    model = cnn(k_init=gim)

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
        #featurewise_center = True
    )

    X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=93)
    train_datagen.fit(X_train)


    import wandb
    from wandb.keras import WandbCallback

    run = wandb.init(project="new_approach", entity="geometric_init")
    wandb.run.name = '8_layer_cnn_cifar100_HEURISTIC_batchnorm_CustomSchedule_2stage' #'8_layer_cnn_cifar100_Heuristic_p=0.7_batchnorm_ReduceLRonPlateau_Vrf2=Glorot'
    wandb.config = {
    "learning_rate": '[1e-6, 1e-4]',
    'batch_size' : '64',
    'epochs' : '30',
    "initialization": 'geo heuristic init l',
    "model": '8_layer_BatchNorm_Heuristic_Liam'
    }


    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr

    layout_callback = FLL(wandb=wandb, model=model, layer_filter_dict={3: [1, 10, 100], 7: [1, 10, 100], 10: [1, 10, 100], 15: [1, 10, 100]})

    '''''''''

    Stage 1 : Learning the MLP

    '''''''''
    batch_size = 64
    epochs = 1
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    for l in model.layers:
        if l.__class__.__name__ == 'Conv2D':
            l.trainable=False

    optimizer = tf.keras.optimizers.RMSprop(tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                                                boundaries=[1*len(X_train)//64], 
                                                values=[1e-3, 1e-4]
                                            ))    
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
                        callbacks=[WandbCallback() , layout_callback, LRLogger(optimizer)]) 
    

    '''''''''

    Stage 2 : Learning the CNN + MLP 

    '''''''''
    batch_size = 64
    epochs = 20
    #
    for l in model.layers:
        if l.__class__.__name__ == 'Conv2D':
            l.trainable = True

    optimizer = tf.keras.optimizers.RMSprop(tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                                                boundaries=[5*len(X_train)//64, 10*len(X_train)//64], 
                                                values=[5e-4, 5e-5, 5e-6]
                                            ))
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
                        callbacks=[WandbCallback() , layout_callback, LRLogger(optimizer)]) 
    
    wandb.finish()
