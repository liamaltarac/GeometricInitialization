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

        self.var_rs2 = self.var_ra2 #self.var_rf2-
        self.var_rs = (self.var_rs2/6.0)**0.5 * (3-8/m.pi)
        self.var_s = tf.math.sqrt(self.var_rs2/66)
        self.e2_rs = 9*self.var_s - self.var_rs

        print(self.var_ra2, self.var_rs2)
        return 3*m.pi*(var_x +var_x +self.var_rs) + self.e2_rs*(3*m.pi-8) - m.pi/(self.channels*0.5)


    def __call__(self, shape, dtype=None, **kwargs):
        self.channels = int(shape[-2])
        self.filters = int(shape[-1])
        self.k = shape[0]        
        self.n_avg = (self.channels+self.filters)/2.0
        self.rho = 0.5
        self.std_init = 100 #tf.math.sqrt(1/(self.n_avg*self.k**2))  #He  #tf.math.sqrt(1/(self.n_avg*self.k**2))  #Glorot

        n = tf.cast(self.channels, dtype=tf.float32)
        p = tf.cast(self.rho, dtype=tf.float32)


        #Anti-symetric initialization
        t = tfp.distributions.Uniform(0, 2*np.pi).sample(sample_shape=(1,self.channels, self.filters), seed=self.seed)
        r = tfp.distributions.Chi(8).sample(sample_shape=(1,self.channels, self.filters), seed=self.seed)
        x = r*tf.math.cos(t)
        y = r*tf.math.sin(t) 

        self.antisym_dist  = tf.stack([x, y], axis=1)
        #print("anti sym orig shape:", self.antisym_dist.shape)

        self.antisym_dist =  tf.reshape(self.antisym_dist, (-1, 2, self.channels*self.filters))

        var_x = tfp.math.find_root_chandrupatla(objective_fn=self._objective, low = [0], high=[1/n])[0]

        #print("Var_x : ", var_x)

        theta = 2*np.pi*tf.math.ceil(tfp.distributions.Uniform(low=0, high=8).sample(sample_shape=(self.filters), seed=self.seed) )/8
        #theta = tf.linspace(np.pi/2, np.pi/2, self.filters, axis=0)
        #print(theta)

        #theta = tfp.distributions.Uniform(low=0, high=2*m.pi).sample(sample_shape=(self.filters), seed=self.seed)    
        R = tf.stack([tf.stack([tf.math.cos(-m.pi/4 + theta ), -tf.math.sin(-m.pi/4  + theta)], axis = -1),     
                     tf.stack([tf.math.sin(-m.pi/4  + theta),  tf.math.cos(-m.pi/4  + theta)], axis=-1)], axis= -1)
        R = tf.cast(R,  tf.dtypes.float32)
        
        
        
        cov = tf.stack([var_x,          self.rho*var_x,      
                        self.rho*var_x,          var_x ])
        cov = tf.cast(tf.reshape(cov, (1, 2,2)), tf.dtypes.float32)
        cov = tf.matmul(tf.matmul(R, cov), R,  transpose_b=True)
        #print("Cov shape :", cov.shape)

        self.antisym_dist  = tf.squeeze(tf.stack([x, y], axis=1), axis=0)

        #print("Dist shape :", self.antisym_dist.shape)

        self.antisym_dist  = tf.transpose(self.antisym_dist, perm=[2, 0, 1])
        self.antisym_dist = self._color(self.antisym_dist, cov)
        #print("New Dist shape :", self.antisym_dist.shape)

        self.antisym_dist  = tf.expand_dims(tf.transpose(self.antisym_dist, perm=[2, 0, 1]), axis=0)


        #print(np.rad2deg(-m.pi/4 + theta[0:5]))

        '''fig = plt.figure()
        ax = fig.add_subplot()

        x = self.antisym_dist[0, :, 0, 0]
        y = self.antisym_dist[0, :, 0, 1]
        plt.scatter(x,y)
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)

        ax.set_box_aspect(1)
        plt.show()    '''   

        x, y = self.antisym_dist[0,:,:,0], self.antisym_dist[0,:,:,1]

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

        
        # Symetric initialization

        #std_s = (self.var_rs2/64)**(1/4) #((self.var_ra2 * 2.5)/64)**(1/4) 
        #std_c = #(self.var_rs2/6)**(1/4) #((self.var_ra2 * 2.5)/64)**(1/4) 
        #std_a = #tf.math.sqrt((std_c**2) /4)
        #std_b = #std_a
        #print("VAR_C", std_c**2)
        print("Var rs2:", self.var_rs2)
        print("E(rs)2:", self.e2_rs)

        #print("Var rs2 check :", 2*std_c**4 + 2*std_a**4 + 32*var_b**2)

        print("Var ra2:", self.var_ra2)

        a = tf.random.normal([1, self.channels, self.filters], stddev = tf.math.sqrt(self.var_s), dtype=tf.dtypes.float32, seed = self.seed)
        b = tf.random.normal([1, self.channels, self.filters], stddev = tf.math.sqrt(self.var_s),  dtype=tf.dtypes.float32, seed = self.seed)
        c = tf.random.normal([1, self.channels, self.filters], stddev = tf.math.sqrt(self.var_s),  dtype=tf.dtypes.float32, seed = self.seed)

        sym_filters = tf.stack([tf.concat([a, b, a],  axis=0), 
                                tf.concat([b, c, b], axis=0),
                                tf.concat([a, b, a], axis=0)])

        #print('stds : ', s_i, s_i_up, s_i_low, std_a, std_s)

        return (asym_filters +   sym_filters)


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
    filters = gi.__call__([3,3,128,200])
    FILTER = [0] #list(range(t.shape[-1]))
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

    plt.hist(mags, bins=32)
    plt.xticks(np.arange(np.min(mags), np.max(mags), step=45), size='small', rotation=0)    
    plt.xlabel('magnitude ')
    plt.ylabel('Count')
    plt.show()
    len(mags)
    print('v_r2: ',np.var(np.array(mags)**2))
    print('v_ra2: ',np.var(np.array(anti_mags)**2))
    print('v_rs2: ',np.var(np.array(sym_mags)**2))
    print('E2_rs: ',np.mean(np.array(sym_mags))**2)

    print('var_init: ',np.var(np.array(f_vals)))
