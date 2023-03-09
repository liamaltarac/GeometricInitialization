import tensorflow as tf
from tensorflow.keras.initializers import Initializer
#from keras import backend
import tensorflow_probability as tfp
import math
import numpy as np
from scipy.stats import wrapcauchy


class GeometricInit3x3(Initializer):

    def __init__(self, seed = None):
        self.seed = seed
        self.channels = None
        self.filters = None
        self.n_avg = None
        self.k = None

        self.rho = None

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)



    def __call__(self, shape, dtype=None, **kwargs):
        self.channels = int(shape[-2])
        self.filters = int(shape[-1])
        self.k = shape[0]        
        self.n_avg = (self.channels+self.filters)/2
        self.rho = 1.0

        std_init = np.sqrt(1/self.n_avg)  #glorot



        #Anti-symetric 
        theta = tfp.distributions.Uniform(low=0, high=2*np.pi).sample(sample_shape=(self.filters), seed=self.seed)       
        print(theta)
        R = tf.stack([tf.math.cos(-np.pi/4 + theta ), -tf.math.sin(-np.pi/4  + theta),     
                    tf.math.sin(-np.pi/4  + theta),  tf.math.cos(-np.pi/4  + theta)])
        R = tf.reshape(R, (self.filters, 2,2))
        
        var_ra2 = 12*std_init**4
        var_x  = (var_ra2/(4*(1+self.rho**2)))**(1/2) #np.sqrt(3) * std_init**2
        var_y = var_x
        print(var_x)
        cov = tf.stack([var_x, self.rho*np.sqrt(var_x*var_y),      
                        self.rho*np.sqrt(var_x*var_y),  var_y ])
        cov = tf.cast(tf.reshape(cov, (1, 2,2)), tf.float32)
        print(cov)
        
        cov = R @ cov @ tf.linalg.matrix_transpose(R)

        print(cov)

        z = tfp.distributions.MultivariateNormalFullCovariance(tf.zeros(2),  
                            covariance_matrix = cov, validate_args = True, allow_nan_stats=False)
        z = z.sample(sample_shape=([1, self.channels, self.filters]), seed = self.seed)
        print(z.shape)

        x, y = z[:,:,0,0,0], z[:,:,0,0, 1]

        ra = tf.math.sqrt(x**2 + y**2)
        theta = tf.math.atan2(y, x)


        # Symetric math
        var_rs =  std_init**2 * (3-8/np.pi)




        
        
        #sln = solve([eq1, eq2], [ra2, rs2], exclude=[p, n, rf2], manual=True, simplify=False, rational=False, minimal=True)
        
        '''chi = tfp.distributions.Chi(4)
        scale = np.sqrt(2)*np.sqrt(1/(2* self.n_avg * self.k**2))
        
        asym_mag_var = (scale**2) * chi.variance()
        print('asym_var :', asym_mag_var)
        magnitude = chi.sample(sample_shape=[1, self.channels, self.filters], seed = self.seed)*scale'''
        '''magnitude = tfp.random.rayleigh(
                        [1, self.channels, self.filters], 
                        scale=s, 
                        dtype=tf.float32,
                        seed=self.seed, 
                        name = None) '''

        '''locs =np.random.uniform(low=0, high=2*np.pi, size=self.filters)       
        theta_var = 1/(self.n_avg * asym_mag_var)
        print('theta_var :', theta_var)

    
    

        chi = tfp.distributions.Chi(3)
        std = np.sqrt(1/(chi.variance() * self.n_avg * 33))
        print('sym_var :', (std**2) * chi.variance())

        magnitude = chi.sample(sample_shape=[1, self.channels, self.filters], seed = self.seed)  
        a = tf.random.normal([1, self.channels, self.filters], stddev = std, dtype=tf.dtypes.float32, seed = self.seed)
        b = tf.random.normal([1, self.channels, self.filters], stddev = std,  dtype=tf.dtypes.float32, seed = self.seed)
        c = tf.random.normal([1, self.channels, self.filters], stddev = std,  dtype=tf.dtypes.float32, seed = self.seed)

        sym_filters = tf.stack([tf.concat([a,b, a], axis=0), 
                                tf.concat([b, c, b], axis=0),
                                tf.concat([a, b, a], axis=0)])'''



        return x,y 
    '''(asym_filters + sym_filters ) * tf.math.rint(tf.random.uniform([1, self.channels, self.filters], 
                                                                  minval=-1, maxval=1, dtype=tf.dtypes.float32,
                                                                  seed=self.seed))'''


    def get_config(self):  # To support serialization
        return {"seed": self.seed}

from skimage.filters import sobel_h
from skimage.filters import sobel_v
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')

def getSobelAngle(f):

    s_h = sobel_h(f)
    s_v = sobel_v(f)
    return (np.degrees(np.arctan2(s_h,s_v)))%180

if __name__ == "__main__":

    gi = GeometricInit3x3(seed=5)
    x,y = gi.__call__([3,3,256,1])
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(x,y)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.set_box_aspect(1)
    plt.show()
    '''FILTER = [15] #list(range(t.shape[-1]))
    CHANNEL =  list(range(filters.shape[-2]))
    thetas = []

    print("F shape : ", filters.shape)
    mags = []
    for i, channel in enumerate(CHANNEL):
        for j, filter in enumerate(FILTER):
            
            f = filters[:,:,:, filter]
            f = np.array(f[:,:, channel])  
            print(f)
            theta = getSobelAngle(f)
            theta = theta[theta.shape[0]//2, theta.shape[1]//2]
            thetas.append(theta)
            mag = np.linalg.norm(f) 

            mags.append(mag)

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
    len(mags)'''