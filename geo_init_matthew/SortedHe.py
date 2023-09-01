import tensorflow as tf
from tensorflow.keras.initializers import Initializer
#from keras import backend
import tensorflow_probability as tfp
import math
#import numpy as np
from scipy.stats import wrapcauchy
import math as m
from skimage.filters import sobel_h
from skimage.filters import sobel_v
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
def getSobelAngle(f):

    s_h = sobel_h(f)
    s_v = sobel_v(f)
    return (np.arctan2(s_h,s_v))


class SortedHe(Initializer):

    def __init__(self, seed = None):
        self.seed = seed
        self.channels = None
        self.filters = None
        self.n_avg = None
        self.k = None

        self.rho = None

        #np.random.seed(self.seed)
        tf.random.set_seed(self.seed)



    def __call__(self, shape, dtype=None, **kwargs):
        self.channels = int(shape[-2])
        self.filters = int(shape[-1])
        self.k = shape[0]        


        initializer = tf.keras.initializers.HeNormal()
        values = initializer(shape=[self.k, self.k, self.channels, self.filters])
        values = tf.reshape(values, shape=[self.k, self.k, self.channels* self.filters])
        #Get filter anngles for every filter
        thetas = []
        index = []
        for i in range(self.channels*self.filters):
            
            f = values[:,:, i]
            f = np.array(f)  
            theta = getSobelAngle(f)
            theta = (theta[theta.shape[0]//2, theta.shape[1]//2] + m.pi)%m.pi
            thetas.append(theta)
            index.append(i)

        '''idx = [[[1]], [[0]]]
        p =   [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
        print(tf.shape(idx))
        print(tf.shape(p))
        v =tf.gather_nd(
                indices = idx,
                params = p)
        print(v)'''
        #values = tf.reshape(values, shape=[self.k, self.k, self.channels*self.filters ])

        index = tf.argsort(thetas, axis=-1, direction='ASCENDING')
        #index = tf.expand_dims(index, axis=2)
        #index = tf.reshape(index, shape=[self.channels*self.filters, 1, 1])
        #index = tf.reshape(index, shape=[1,1,self.channels*self.filters])

        #index  = [x for _,x in sorted(zip(thetas,index))] 
        values = tf.gather(params = values, indices = index, axis=-1)
        #index = tf.re
        print(index)

        #values = tf.squeeze(tf.gather_nd(values, index), axis=1)
        #print(values)

        values = tf.reshape(values, shape=[self.k, self.k, self.filters,  self.channels])
        values = tf.transpose(values, perm=[0, 1, 3, 2])
        print(values)

        return values
        #return (asym_filters + sym_filters ) 


    def get_config(self):  # To support serialization
        return {"seed": self.seed}


matplotlib.use('TKAgg')

def getSobelAngle(f):

    s_h = sobel_h(f)
    s_v = sobel_v(f)
    return (np.arctan2(s_h,s_v))


def getSymAntiSym(filter):

    #patches = extract_image_patches(filters, [1, k, k, 1],  [1, k, k, 1], rates = [1,1,1,1] , padding = 'VALID')
    #print(patches)
    mat_flip_x = np.fliplr(filter)

    mat_flip_y = np.flipud(filter)

    mat_flip_xy =  np.fliplr( np.flipud(filter))

    sum = filter + mat_flip_x + mat_flip_y + mat_flip_xy
    mat_sum_rot_90 = np.rot90(sum)
    
    return  (sum + mat_sum_rot_90) / 8, filter - ((sum + mat_sum_rot_90) / 8)

if __name__ == "__main__":

    gi = SortedHe(seed=50)
    filters = gi.__call__([3,3,128,128])
    FILTER = [55] #list(range(t.shape[-1]))
    CHANNEL =  list(range(filters.shape[-2]))
    #thetas = []

    print("F shape : ", filters.shape)
    mags = []
    anti_mags = []
    sym_mags = []
    f_vals = []
    thetas = []
    for i in range(128):
        for j, filter in enumerate(FILTER):
            
            f = filters[:,:,:, filter]
            f = np.array(f[:,:, i])  
            #print(f)
            f_vals.append(f)
            theta = getSobelAngle(f)
            theta = theta[theta.shape[0]//2, theta.shape[1]//2]
            thetas.append(theta)
            mag = np.linalg.norm(f) 

            mags.append(mag)
            s, a = getSymAntiSym(f)
            sym_mag = np.linalg.norm(s) 
            anti_mag = np.linalg.norm(a) 
            anti_mags.append(anti_mag)
            sym_mags.append(sym_mag)


    plt.hist(thetas, bins=16)
    plt.xticks(np.arange(0, 2*np.pi, step=1), size='small', rotation=0)    
    plt.title("Layer {}, Filter = {}, Channel orientation Distribution".format("N", FILTER[0]))
    plt.xlabel('θ (Deg)')
    plt.ylabel('Count')
    plt.show()
    print(thetas)
    t_rad = thetas
    n = len(thetas)
    r = np.sqrt(np.sum(np.cos(t_rad))**2 + np.sum(np.sin(t_rad))**2 )
    print(1 - r/n)

    plt.hist(mags, bins=32)
    plt.xticks(np.arange(np.min(mags), np.max(mags), step=45), size='small', rotation=0)    
    plt.xlabel('magnitude ')
    plt.ylabel('Count')
    plt.show()

    fig = plt.figure()

    ax = fig.add_subplot()
    x =anti_mags*np.cos((t_rad))
    y = anti_mags*np.sin((t_rad))

    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    plt.scatter(x,y)



    ax.set_box_aspect(1)
    plt.show()


    len(mags)
    print('v_r2: ',np.var(np.array(mags)**2))
    print('v_ra2: ',np.var(np.array(anti_mags)**2))
    print('v_rs2: ',np.var(np.array(sym_mags)**2))
    print('var_init: ',np.var(np.array(f_vals)))

