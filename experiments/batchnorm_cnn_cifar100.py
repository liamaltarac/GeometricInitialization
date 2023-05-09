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

    X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.3, random_state=93)
    train_datagen.fit(X_train)


    import wandb
    from wandb.keras import WandbCallback

    run = wandb.init(project="new_approach", entity="geometric_init")
    wandb.run.name = '8_layer_cnn_cifar100_HEURISTIC_batchnorm_CustomSchedule' #'8_layer_cnn_cifar100_Heuristic_p=0.7_batchnorm_ReduceLRonPlateau_Vrf2=Glorot'
    wandb.config = {
    "learning_rate": '[1e-6, 1e-4]',
    'batch_size' : '64',
    'epochs' : '30',
    "initialization": 'geo heuristic init l',
    "model": '8_layer_BatchNorm_Heuristic_Liam'
    }

    '''optimizers = [
    tf.keras.optimizers.RMSprop(learning_rate=5e-5),
    tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    ]
    optimizers_and_layers = [(optimizers[0], model.layers[:-6]), (optimizers[1], model.layers[-6:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)'''
    
    #lr = 1e-4

    optimizer = tf.keras.optimizers.RMSprop(tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                                                boundaries=[10*len(X_train)//64, 20*len(X_train)//64], 
                                                values=[1e-4, 1e-5, 1e-6]
                                            ))
    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr

    lr_metric = get_lr_metric(optimizer)
    model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
    )
    batch_size = 64
    epochs = 30

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=1e-6, min_delta=0.03)

    #lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=True)

    layout_callback = FLL(wandb=wandb, model=model, layer_filter_dict={3: [1, 10, 100], 7: [1, 10, 100], 10: [1, 10, 100], 15: [1, 10, 100]})
    history = model.fit(X_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(X_validation, y_validation),
                        callbacks=[WandbCallback(), layout_callback, LRLogger(optimizer)]) 
    
    wandb.finish()

    '''log_gradients   = (True), 
    log_weights     = (True),
    training_data   = (X_train, y_train),
    validation_data = (X_validation, y_validation)
    input_type = "images",
    output_type = "label",
    )])'''