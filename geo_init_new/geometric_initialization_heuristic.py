import tensorflow as tf
from tensorflow.keras.initializers import Initializer
#from keras import backend
import tensorflow_probability as tfp
import math
#import numpy as np
from scipy.stats import wrapcauchy
import math as m


class GeometricInit3x3(Initializer):

    def __init__(self, seed = None, rho=0.99, beta=0.66):
        self.seed = seed
        self.channels = None
        self.filters = None
        self.n_avg = None
        self.k = None

        self.rho = rho
        self.beta = beta
        
        self.std_init = None

        self.antisym_dist = None
        self.var_rf2 = None

        self.var_ra2 = None
        self.var_ra = None

        self.var_rs2 = None
        self.var_rs = None
        self.e2_rs = None
        self.var_s = None


        #np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def _color(self, dist, cov):
        # Color Transfor (isotropic->anisotropic) 

        # Make distribution have unit variance
        dist = dist/tf.expand_dims(tf.math.sqrt(tfp.stats.variance(dist, -1)), -1)
        #print("unit var dist shape :", dist.shape)
        e_vals, e_vec = tf.linalg.eigh(cov)
        e_vals = tf.linalg.diag(e_vals)
        #print("e_vals shape:", e_vals.shape)
        #print("e_vec shape:", e_vec.shape)

        new_dist = e_vec @ e_vals**(1/2) @ dist
         
        #new_dist =  tf.reshape(new_dist, (-1, 2, self.channels, self.filters))

        #print(self.channels, self.filters)
        return new_dist

    def _objective(self, var_x):


        #input x : var(x)
        #Calculate the "colored" antisymetric distribution
        cov = tf.stack([var_x, self.rho*tf.math.sqrt(var_x*var_x),      
                        self.rho*tf.math.sqrt(var_x*var_x),  var_x ])
        cov = tf.cast(tf.reshape(cov, (2,2)), tf.dtypes.float32)
        dist = self._color(self.antisym_dist, cov)

        x = dist[:, 0, :]
        y = dist[:, 1, :]

        #self.var_rf2 = 18*self.std_init**4

        self.var_ra2 = tfp.stats.variance(x**2 + y**2, None)
        self.var_ra = tfp.stats.variance(tf.math.sqrt(x**2 + y**2), None)

        self.var_rs2 = (1-self.beta)*self.var_ra2/self.beta
        self.var_rs = (self.var_rs2/6.0)**0.5 * (3-8/m.pi)
        #self.var_s = tf.math.sqrt(self.var_rs2/66)

        print("Beta", self.var_ra2/(self.var_rs2+self.var_ra2))
        print("var_x", var_x)

        return 6*var_x + (3 + 8/m.pi)*self.var_rs - (2/self.channels)


    def __call__(self, shape, dtype=None, **kwargs):
        self.channels = int(shape[-2])
        self.filters = int(shape[-1])
        self.k = shape[0]        

        n = tf.cast(self.channels, dtype=tf.float32)
        p = tf.cast(self.rho, dtype=tf.float32)


        #Anti-symetric (isotropic distribution)
        t = tfp.distributions.Uniform(0, 2*np.pi).sample(sample_shape=(1,self.channels, self.filters), seed=self.seed)
        r = tfp.distributions.Chi(8).sample(sample_shape=(1,self.channels, self.filters), seed=self.seed)
        x = r*tf.math.cos(t)
        y = r*tf.math.sin(t) 


        self.antisym_dist  = tf.stack([x, y], axis=1)
        self.antisym_dist =  tf.reshape(self.antisym_dist, (-1, 2, self.channels*self.filters))


        # Color the  Anti-symetric distribution according to a covariance matrix
        var_x = tfp.math.find_root_chandrupatla(objective_fn=self._objective, low = [0], high=[2/(6*n)])[0]
        print("VAR_X = ", var_x)
        cov = tf.stack([var_x,          self.rho*var_x,      
                        self.rho*var_x,          var_x ])
        cov = tf.cast(tf.reshape(cov, (1, 2,2)), tf.dtypes.float32)

        self.antisym_dist  = tf.squeeze(tf.stack([x, y], axis=1), axis=0)

        self.antisym_dist  = tf.transpose(self.antisym_dist, perm=[2, 0, 1])
        self.antisym_dist = self._color(self.antisym_dist, cov)
        self.antisym_dist  = tf.expand_dims(tf.transpose(self.antisym_dist, perm=[2, 0, 1]), axis=0)

        x, y = self.antisym_dist[0,:,:,0], self.antisym_dist[0,:,:,1]

        ra = tf.math.sqrt(x**2 + y**2)
        theta = tf.expand_dims(tf.math.atan2(y, x), axis=0)


        # Generate Anti-symetric Filters according the  Anti-symetric distribution
        antisym_rotation = tfp.distributions.Uniform(-m.pi, m.pi).sample(sample_shape=(1,1,self.filters))

    
        theta += antisym_rotation
        
        #Simpified Anti-Sym Gabor

        a = -tf.math.sin(tf.math.cos(theta)- tf.math.sin(theta))
        b = tf.math.sin(tf.math.sin(theta))
        c = tf.math.sin(tf.math.cos(theta) + tf.math.sin(theta))
        d = -tf.math.sin(tf.math.cos(theta))


        '''a = -tf.math.sqrt(8.0)*tf.math.cos(theta - 9*math.pi/4)
        b = -2*tf.math.sin(theta)
        c = -tf.math.sqrt(8.0)*tf.math.sin(theta - 9*math.pi/4)
        d = -2*tf.math.cos(theta)'''
        
        asym_filters = tf.stack([tf.concat( [a,b,c], axis=0) , 
                        tf.concat( [d,tf.zeros([1, self.channels, self.filters]), -d], axis=0),
                        tf.concat( [-c, -b, -a], axis=0)])
                

        #Scale the filters to the correct magnitude from the  Anti-symetric distribution  
        norm = tf.sqrt(tf.reduce_sum(tf.square(asym_filters), axis=[0,1]))  
        asym_filters =  tf.math.multiply((asym_filters / norm) , ra)

        #print("Var ra2:", np.var(ra**2))
        #print(np.cov(x, y))
        # Symetric initialization

        #std_s = (self.var_rs2/64)**(1/4) #((self.var_ra2 * 2.5)/64)**(1/4) 
        #std_c = #(self.var_rs2/6)**(1/4) #((self.var_ra2 * 2.5)/64)**(1/4) 
        #std_a = #tf.math.sqrt((std_c**2) /4)
        #std_b = #std_a
        #print("VAR_C", std_c**2)

        var_rs = self.var_rs #1/((3+8/m.pi) * n)

        print("Var rs:", var_rs)
        print("Var ra2:", np.var(ra**2))
        #print("Var rs2 check :", 2*std_c**4 + 2*std_a**4 + 32*var_b**2)

        print("Var x:", var_x)
        var_i = tf.math.sqrt(self.var_rs2/66)

        print("Var i:", var_i)

        a = tf.random.normal([1, self.channels, self.filters], stddev = tf.math.sqrt(var_i), dtype=tf.dtypes.float32, seed = self.seed)
        b = tf.random.normal([1, self.channels, self.filters], stddev = tf.math.sqrt(var_i),  dtype=tf.dtypes.float32, seed = self.seed)
        c = tf.random.normal([1, self.channels, self.filters], stddev = tf.math.sqrt(var_i),  dtype=tf.dtypes.float32, seed = self.seed)

        sym_filters = tf.stack([tf.concat([a, b, a],  axis=0), 
                                tf.concat([b, c, b], axis=0),
                                tf.concat([a, b, a], axis=0)])
        

        print('sym_filters stds : ', np.std(c))

        return (asym_filters + sym_filters)


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

    gi = GeometricInit3x3(seed=7)
    filters = gi.__call__([3,3,128,128])
    FILTER = [1] #list(range(t.shape[-1]))
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

    plt.hist(anti_mags, bins=32)
    plt.xticks(np.arange(np.min(mags), np.max(mags), step=45), size='small', rotation=0)    
    plt.xlabel('magnitude ')
    plt.ylabel('Count')
    plt.show()
    #print(mags)
    print('v_r2: ',np.var(np.array(mags)**2))
    print('v_ra2: ',np.var(np.array(anti_mags)**2))
    print('v_rs2: ',np.var(np.array(sym_mags)**2))
    print('v_rs: ',np.var(np.array(sym_mags)))
    print('cov_ra_rs: ',np.cov(np.array(sym_mags)**2, np.array(anti_mags)**2))

    print('E2_rs: ',np.mean(np.array(sym_mags)))
    print('E2_ra: ',np.mean(np.array(anti_mags)))

    print('var_init: ',np.var(f_vals))
