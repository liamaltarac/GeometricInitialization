import tensorflow as tf
from tensorflow.keras.initializers import Initializer
#from keras import backend
import tensorflow_probability as tfp
import math
#import numpy as np
from scipy.stats import wrapcauchy
import math as m


class GeometricInit3x3(Initializer):

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
        self.n_avg = (self.channels+self.filters)/2.0
        self.rho = 0.5

        #std_init = tf.math.sqrt(1/(self.n_avg)*self.k**2)  #glorot
        std_init = tf.math.sqrt(2/(self.channels*self.k**2))  # he sqrt(2 / fan_in)


        #Anti-symetric 
        theta = 2*np.pi*tf.math.ceil(tfp.distributions.Uniform(low=0, high=8).sample(sample_shape=(self.filters), seed=self.seed) )/8

        theta = tfp.distributions.Uniform(low=0, high=2*m.pi).sample(sample_shape=(self.filters), seed=self.seed)    
        R = tf.stack([tf.stack([tf.math.cos(-m.pi/4 + theta ), -tf.math.sin(-m.pi/4  + theta)], axis = -1),     
                     tf.stack([tf.math.sin(-m.pi/4  + theta),  tf.math.cos(-m.pi/4  + theta)], axis=-1)], axis= -1)
        R = tf.cast(R,  tf.dtypes.float32)
        #R = tf.reshape(R, (self.filters, 2,2))


        var_ra2 = 12*std_init**4
        var_x  = (var_ra2/(4*(1+self.rho**2)))**(1/2) #np.sqrt(3) * std_init**2
        var_y = var_x
        cov = tf.stack([var_x, self.rho*tf.math.sqrt(var_x*var_y),      
                        self.rho*tf.math.sqrt(var_x*var_y),  var_y ])

        cov = tf.cast(tf.reshape(cov, (1, 2,2)), tf.dtypes.float32)

        
        #cov = R @ cov @ tf.linalg.matrix_transpose(R)
        cov = tf.matmul(tf.matmul(R, cov), R,  transpose_b=True)


        z = tfp.distributions.MultivariateNormalTriL(loc = None,  
                            scale_tril=tf.linalg.cholesky(cov), validate_args = False, allow_nan_stats=True)
        z = z.sample(sample_shape=([1, self.channels]), seed = self.seed)


        x, y = z[0,:,:,0], z[0,:,:,1]

        ra = tf.math.sqrt(x**2 + y**2)
        theta = tf.expand_dims(tf.math.atan2(y, x), axis=0)

        a = -tf.math.sqrt(8.0)*tf.math.cos(theta - 9*math.pi/4)
        b = -2*tf.math.sin(theta)
        c = -tf.math.sqrt(8.0)*tf.math.sin(theta - 9*math.pi/4)
        d = -2*tf.math.cos(theta)
        
        asym_filters = tf.stack([tf.concat( [a,b,c], axis=0) , 
                        tf.concat( [d,tf.zeros([1, self.channels, self.filters]), -d], axis=0),
                        tf.concat( [-c, -b, -a], axis=0)])
                
        norm = tf.sqrt(tf.reduce_sum(tf.square(asym_filters), axis=[0,1]))  
        asym_filters = tf.math.multiply((asym_filters / norm) , ra)


        # Symetric math
        a = tf.random.normal([1, self.channels, self.filters], stddev = std_init/2, dtype=tf.dtypes.float32, seed = self.seed)
        b = tf.random.normal([1, self.channels, self.filters], stddev = std_init/2,  dtype=tf.dtypes.float32, seed = self.seed)
        c = tf.random.normal([1, self.channels, self.filters], stddev = std_init,  dtype=tf.dtypes.float32, seed = self.seed)

        sym_filters = tf.stack([tf.concat([a,b, a], axis=0), 
                                tf.concat([b, c, b], axis=0),
                                tf.concat([a, b, a], axis=0)])



        return (asym_filters + sym_filters ) 


    def get_config(self):  # To support serialization
        return {"seed": self.seed}

from skimage.filters import sobel_h
from skimage.filters import sobel_v
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TKAgg')

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

    gi = GeometricInit3x3(seed=5)
    filters = gi.__call__([3,3,256,200])
    FILTER = [15] #list(range(t.shape[-1]))
    CHANNEL =  list(range(filters.shape[-2]))
    thetas = []

    print("F shape : ", filters.shape)
    mags = []
    anti_mags = []
    sym_mags = []
    f_vals = []

    for i, channel in enumerate(CHANNEL):
        for j, filter in enumerate(FILTER):
            
            f = filters[:,:,:, filter]
            f = np.array(f[:,:, channel])  
            print(f)
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
    print('v_r2: ',np.var(np.array(mags)**2))
    print('v_ra2: ',np.var(np.array(anti_mags)**2))
    print('v_rs2: ',np.var(np.array(sym_mags)**2))
    print('var_init: ',np.var(np.array(f_vals)))

