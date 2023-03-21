import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

import sys

if __name__ == '__main__':


    sys.path.append("..")    

    from .models.no_batch_norm import no_batchnorm_cnn
    from geo_init.geometric_initialization_relu import GeometricInit3x3Relu
    #from geo_init_matthew.geometric_initialization import GeometricInit3x3 as gim
    from geo_init_liam.geometric_initialization import GeometricInit3x3 as gim

    from tensorflow.python.client import device_lib
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.__version__ )

    model = no_batchnorm_cnn(k_init = gim)

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

    wandb.init(project="new_approach")
    wandb.run.name = '8_layer_cnn_cifar100_Liam3_noBN'
    wandb.config = {
    "learning_rate": '[1e-6, 1e-4]',
    'batch_size' : '64',
    'epochs' : '10',
    "initialization": "geo init m",
    "model": '8_layer_No_BatchNorm Matthew_he'
    }

    optimizers = [
    tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    ]
    optimizers_and_layers = [(optimizers[0], model.layers[:-6]), (optimizers[1], model.layers[-6:])]
    #optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

    model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
    )
    batch_size = 64
    epochs = 10

    history = model.fit(X_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(X_validation, y_validation),
                        callbacks=[WandbCallback()]) 
    '''log_gradients   = (True), 
    log_weights     = (True),
    training_data   = (X_train, y_train),
    validation_data = (X_validation, y_validation)
    input_type = "images",
    output_type = "label",
    )])'''