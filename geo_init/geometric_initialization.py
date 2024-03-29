import tensorflow as tf
from tensorflow.keras.initializers import Initializer
from keras import backend
import tensorflow_probability as tfp
import math
import numpy as np
from scipy.stats import wrapcauchy

#From  N.I Fisher - Statistical analyis of circular data p.47-48
def wrapnormal(mean, std, size):

    out = np.zeros(size)
    for i in range(out.size):
        while(1):
            u1 = np.random.uniform(low=0.0, high=1.0, size=None)
            u2 = np.random.uniform(low=0.0, high=1.0, size=None)
            z = 1.715528*(u1-0.5)/u2
            x = 0.25*z**2
            if x <= 1 - u2 or x <= -np.log(u2):
                break

        z = np.random.normal(loc=0.0, scale=1.0, size=None)
        X = std*z + mean
        out[i] = X % (2*np.pi)
    return out

class GeometricInit3x3(Initializer):

    def __init__(self, seed = None):
        self.seed = seed
        self.channels = None
        self.filters = None
        self.n_avg = None
        self.k = None
        np.random.seed(self.seed)


    def __call__(self, shape, dtype=None, **kwargs):
        self.channels = int(shape[-2])
        self.filters = int(shape[-1])
        self.k = shape[0]        
        self.n_avg = (self.channels+self.filters)/2
        

        chi = tfp.distributions.Chi(4)
        scale = np.sqrt(2)*np.sqrt(1/(2* self.n_avg * self.k**2))
        
        asym_mag_var = (scale**2) * chi.variance()
        print('asym_var :', asym_mag_var)
        magnitude = chi.sample(sample_shape=[1, self.channels, self.filters], seed = self.seed)*scale
        '''magnitude = tfp.random.rayleigh(
                        [1, self.channels, self.filters], 
                        scale=s, 
                        dtype=tf.float32,
                        seed=self.seed, 
                        name = None) '''

        locs =np.random.uniform(low=0, high=2*np.pi, size=self.filters)       
        theta_var = 1/(self.n_avg * asym_mag_var)
        print('theta_var :', theta_var)

        #print(theta_var)
        #c = 1 - theta_var
        #print(c)

        theta = np.array([], dtype=np.float32).reshape(self.channels, 0)
        for loc in locs:
            theta = np.concatenate((theta, np.array([wrapnormal(mean = loc,
                                                    std = np.sqrt(theta_var),
                                                    size = self.channels)]).T), axis=1)
        theta =  np.float32(theta)
        theta = tf.convert_to_tensor(
                    tf.expand_dims(theta, axis=0), dtype=tf.float32)

        a = -tf.math.sqrt(8.0)*tf.math.cos(theta - 9*math.pi/4)
        b = -2*tf.math.sin(theta)
        c = -tf.math.sqrt(8.0)*tf.math.sin(theta - 9*math.pi/4)
        d = -2*tf.math.cos(theta)
        
        asym_filters = tf.stack([tf.concat( [a,b,c], axis=0) , 
                        tf.concat( [d,tf.zeros([1, self.channels, self.filters]), -d], axis=0),
                        tf.concat( [-c, -b, -a], axis=0)])

        
        norm = tf.sqrt(tf.reduce_sum(tf.square(asym_filters), axis=[0,1]))  
        print("N shape ", norm.shape)
        asym_filters = tf.math.multiply((asym_filters / norm) , magnitude)




    
    

        chi = tfp.distributions.Chi(3)
        std = np.sqrt(1/(chi.variance() * self.n_avg * 33))
        print('sym_var :', (std**2) * chi.variance())

        magnitude = chi.sample(sample_shape=[1, self.channels, self.filters], seed = self.seed)  
        a = tf.random.normal([1, self.channels, self.filters], stddev = std, dtype=tf.dtypes.float32, seed = self.seed)
        b = tf.random.normal([1, self.channels, self.filters], stddev = std,  dtype=tf.dtypes.float32, seed = self.seed)
        c = tf.random.normal([1, self.channels, self.filters], stddev = std,  dtype=tf.dtypes.float32, seed = self.seed)

        sym_filters = tf.stack([tf.concat([a,b, a], axis=0), 
                                tf.concat([b, c, b], axis=0),
                                tf.concat([a, b, a], axis=0)])



        return (asym_filters + sym_filters ) * tf.math.rint(tf.random.uniform([1, self.channels, self.filters], 
                                                                  minval=-1, maxval=1, dtype=tf.dtypes.float32,
                                                                  seed=self.seed))


    def get_config(self):  # To support serialization
        return {"seed": self.seed}

from skimage.filters import sobel_h
from skimage.filters import sobel_v
import matplotlib
import matplotlib.pyplot as plt
def getSobelAngle(f):

    s_h = sobel_h(f)
    s_v = sobel_v(f)
    return (np.degrees(np.arctan2(s_h,s_v)))

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

    gi = GeometricInit3x3()
    filters = gi.__call__([3,3,256,32])
    FILTER = [15] #list(range(t.shape[-1]))
    CHANNEL =  list(range(filters.shape[-2]))

    print("F shape : ", filters.shape)
    thetas = []
    anti_mags = []
    sym_mags = []

    mags = []
    for i, channel in enumerate(CHANNEL):
        for j, filter in enumerate(FILTER):
            
            f = filters[:,:,:, filter]
            f = np.array(f[:,:, channel])  
            s, a = getSymAntiSym(f)
            theta = getSobelAngle(f)
            theta = theta[theta.shape[0]//2, theta.shape[1]//2]
            thetas.append(theta)
            sym_mag = np.linalg.norm(s) 
            anti_mag = np.linalg.norm(a) 
            anti_mags.append(anti_mag)
            sym_mags.append(sym_mag)

            mags.append(np.linalg.norm(f))

    plt.hist(thetas, bins=16)
    plt.xticks(np.arange(0, 2*np.pi, step=1), size='small', rotation=0)    
    plt.title("Layer {}, Filter = {}, Channel orientation Distribution".format("N", FILTER[0]))
    plt.xlabel('θ (Deg)')
    plt.ylabel('Count')
    plt.show()
    print(len(thetas))
    t_rad = thetas
    n = len(thetas)
    r = np.sqrt(np.sum(np.cos(t_rad))**2 + np.sum(np.sin(t_rad))**2 )
    print(1 - r/n)
    plt.hist(mags, bins=32)
    plt.xticks(np.arange(np.min(mags), np.max(mags), step=45), size='small', rotation=0)    
    plt.xlabel('magnitude ')
    plt.ylabel('Count')
    plt.show()
    len(mags)
    print(np.var(anti_mags))
    print(np.var(sym_mags))
