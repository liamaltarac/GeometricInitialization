from skimage.filters import sobel_h
from skimage.filters import sobel_v
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TKAgg')
import tensorflow as tf

class FLL(tf.keras.callbacks.Callback):

    def __init__(self, wandb, model, layer_filter_dict, file_type = "png"):
        self.wandb = wandb
        self.model = model
        self.lfd = layer_filter_dict  # {0 : [1,2,3,5], 5: [10, 20] , ...}
        self.ft = file_type

    def get_filter(self, layer):
        layer = self.model.layers[layer]

        # check for convolutional layer
        if 'conv' not in layer.name:
            raise ValueError('Layer must be a conv. layer')
        # get filter weights
        filters, biases = layer.get_weights()
        print("biases shape : ", biases.shape)
        print("filters shape : ", filters.shape)

        return (filters)
        #print(layer.name, filters.shape)

    def getSobelAngle(self, f):

        s_h = sobel_h(f)
        s_v = sobel_v(f)
        return (np.arctan2(s_h,s_v))


    def getSymAntiSym(self, filter):

        mat_flip_x = np.fliplr(filter)

        mat_flip_y = np.flipud(filter)

        mat_flip_xy =  np.fliplr( np.flipud(filter))

        sum = filter + mat_flip_x + mat_flip_y + mat_flip_xy
        mat_sum_rot_90 = np.rot90(sum)
        
        return  (sum + mat_sum_rot_90) / 8, filter - ((sum + mat_sum_rot_90) / 8)


    def on_epoch_begin(self, epoch, logs=None):

        for layer, filters in self.lfd.items():
            weights = self.get_filter(layer)
            for filter in filters:
                #FILTER = [15] #list(range(t.shape[-1]))
                channels =  list(range(weights.shape[-2]))
                thetas = []

                mags = []
                anti_mags = []
                sym_mags = []
                f_vals = []

                for i, channel in enumerate(channels):
                        
                    f = weights[:,:,:, filter]
                    f = np.array(f[:,:, channel])  
                    
                    theta = self.getSobelAngle(f)
                    theta = theta[theta.shape[0]//2, theta.shape[1]//2]
                    thetas.append(theta)
                    
                    mag = np.linalg.norm(f) 
                    mags.append(mag)
                    s, a = self.getSymAntiSym(f)
                    anti_mag = np.linalg.norm(a) 
                    anti_mags.append(anti_mag)


                fig = plt.figure(figsize=(10,6))

                ax = fig.add_subplot()
                x =anti_mags*np.cos((thetas))
                y = anti_mags*np.sin((thetas))

                lim_x = np.max(np.abs(x))
                lim_y = np.max(np.abs(y))
                lim = np.max([lim_x, lim_y])
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                plt.scatter(x,y)
                ax.set_box_aspect(1)



                ax.set_box_aspect(1)
                plt.savefig('Layer_{}_Filter_{}.{}'.format(str(layer), str(filter),self.ft))
                print('Saving Layer_{}_Filter_{}.{}'.format(str(layer), str(filter),self.ft))
                plt.close()
                self.wandb.log({'Layer {}, Filter {}'.format(str(layer), str(filter)): self.wandb.Image('Layer_{}_Filter_{}.{}'.format(str(layer), str(filter),self.ft))})

