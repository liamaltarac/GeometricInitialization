if __name__ == '__main__':

    import tensorflow as tf
    from tensorflow import keras
    import sys
    sys.path.append("..")    

    from .models.batch_norm import batchnorm_cnn
    from geo_init.geometric_initialization_relu import GeometricInit3x3Relu

    from tensorflow.python.client import device_lib
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.__version__ )

    model = batchnorm_cnn(k_init=GeometricInit3x3Relu)

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
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_validation, y_validation))