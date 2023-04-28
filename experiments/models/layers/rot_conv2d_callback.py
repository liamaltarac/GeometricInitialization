import tensorflow as tf
import time
import io
from keras import backend

class RotConv2DCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, rotation_epochs, rotation_lr):
        self.rot_epochs = rotation_epochs
        self.rot_lr = rotation_lr #learning rate for rotation parameter
        self.model = model
        self.w_lr = float(backend.get_value(self.model.optimizer.lr)) # weight learning rate

    def on_epoch_begin(self, epoch, logs=None):
        if self.rot_epochs == epoch:
            backend.set_value(self.model.optimizer.learning_rate, self.w_lr)
            print("Training weights now @",backend.get_value(self.model.optimizer.lr))
            for l in self.model.layers:
                l.trainable = True
                if l.__class__.__name__ == 'RotConv2D':
                    l._train_r = False

                    

    def on_train_begin(self, logs=None):
        backend.set_value(self.model.optimizer.learning_rate, self.rot_lr)
        print("Training Distrbution rotation @", backend.get_value(self.model.optimizer.lr))

        for l in self.model.layers:
            l.trainable = False
            if l.__class__.__name__ == 'RotConv2D':
                l.trainable = True
                l._train_r = True
