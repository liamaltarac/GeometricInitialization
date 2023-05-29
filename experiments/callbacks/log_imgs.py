from skimage.filters import sobel_h
from skimage.filters import sobel_v
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import time
import io
from PIL import Image

class LI(tf.keras.callbacks.Callback):

    def __init__(self, wandb, model, num_imgs,dataset,  file_type = "png" ):
        self.wandb = wandb
        self.model = model
        self.X = dataset
        self.n = num_imgs  # {0 : [1,2,3,5], 5: [10, 20] , ...}
        self.ft = file_type
        #self.epochs = int(wandb.config['epochs'])

    def on_epoch_begin(self, epoch, logs=None):


        for i in range(self.n):
            fig, ax = plt.subplots(1,2)
            fig.set_tight_layout(True)
            x = self.X[i]
            ax[0].imshow(x)
            y = self.model.predict(tf.expand_dims(x, axis=0))
            ax[1].imshow(y[0])

            buf = io.BytesIO()

            plt.savefig(buf, format=self.ft)
            buf.seek(0)
            plt.close()
            im = Image.open(buf)

            print('sample.{}'.format(self.ft))
            self.wandb.log({'sample {}'.format(str(i)): self.wandb.Image(im)})

            buf.close()
