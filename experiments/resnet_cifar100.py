if __name__ == '__main__':

    import tensorflow as tf
    from tensorflow import keras
    import sys
    sys.path.append("..")    

    from .models.resnet50 import ResNet50
    from geo_init.geometric_initialization_relu import GeometricInit3x3Relu

    from tensorflow.python.client import device_lib
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.__version__ )

    model = ResNet50(weights=None,
                  include_top=True,
                  input_shape=(32, 32, 3),
                  k_init=GeometricInit3x3Relu,
                  classes = 100)

    num_classes = 100
    input_shape = (32, 32, 3)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
    )
    batch_size = 128
    epochs = 10
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))