import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Concatenate , Add, Dot, Activation, Lambda
from tensorflow.compat.v2.math import mod
import tensorflow_probability as tfp
import math as m

import sys 

class RotConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, padding = 'VALID', strides = (1, 1), activation=None, use_bias = True, seed=None):
        super(RotConv2D, self).__init__(trainable= True)
        self.seed = seed

        tf.random.set_seed(self.seed)
        
        self.filters = filters
        self.channels = None
        self.kernel_size = (3,3)
        self.k = 3       

        self.activation = activation
        self.padding = padding

        self.bias_initializer = tf.initializers.Zeros()
        self.strides = strides
        self.use_bias = use_bias

        self.antisym_rotation = None

        self.antisym_dist = None
        self.asym_filters = None
        self.sym_filters = None
        self.var_x = None
        self.x = None
        self.y = None
        
        self.w = None
        self.bias = None

        self.rho = None
        self.std_init = None

        self.var_rf2 = None

        self.var_ra2 = None
        self.var_ra = None

        self.var_rs2 = None
        self.var_rs = None

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "padding": self.padding,
            "strides": self.strides,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "seed": self.seed
        })
        return config


    def _color(self, x, y, cov):
        # Color Transfor (isotropic->anisotropic) 

        # Make distribution have unit variance
        x = x/tf.math.sqrt(tfp.stats.variance(x, 1))
        y = y/tf.math.sqrt(tfp.stats.variance(y, 1))
        #print("unit var dist shape :", dist.shape)
        e_vals, e_vec = tf.linalg.eigh(cov)
        e_vals = tf.linalg.diag(e_vals)
        H = e_vec @ e_vals**(1/2)
        #print("e_vals shape:", e_vals.shape)
        #print("e_vec shape:", e_vec.shape)

        new_x = H[:,0]  *  x
        new_y = H[:,1]  *  y
         
        #new_dist =  tf.reshape(new_dist, (-1, 2, self.channels, self.filters))

        #print(self.channels, self.filters)
        return new_x, new_y

    def _objective(self, var_x):


        #input x : var(x)
        #Calculate the "colored" antisymetric distribution
        cov = tf.stack([var_x, self.rho*tf.math.sqrt(var_x*var_x),      
                        self.rho*tf.math.sqrt(var_x*var_x),  var_x ])
        cov = tf.cast(tf.reshape(cov, (2,2)), tf.dtypes.float32)
        x, y = self._color(self.antisym_dist, cov)

        self.var_rf2 = 18*self.std_init**4

        self.var_ra2 = tfp.stats.variance(x**2 + y**2, None)
        self.var_ra = tfp.stats.variance(tf.math.sqrt(x**2 + y**2), None)

        self.var_rs2 = self.var_rf2-self.var_ra2
        self.var_rs = (self.var_rs2/6.0)**0.5 * (3-8/m.pi)
        #print(var_ra2, var_rs2)
        return 6*var_x + (3-8/m.pi)*self.var_rs - 1/self.channels
    

    def get_weights(self):
        
        return self.w, self.bias
    
    def build(self, shape):
        print(shape)
        self.channels = int(shape[-1])
        self.n_avg = (self.channels+self.filters)/2.0
        self.rho = 0.3

        self.std_init = tf.math.sqrt(2/(self.channels*self.k**2))  #He

        n = tf.cast(self.channels, dtype=tf.float32)
        p = tf.cast(self.rho, dtype=tf.float32)


        #Anti-symetric initialization
        self.t = tfp.distributions.Uniform(0, 2*np.pi).sample(sample_shape=(1,self.channels, self.filters), seed=self.seed)
        self.r = tfp.distributions.Chi(8).sample(sample_shape=(1,self.channels, self.filters), seed=self.seed)
        self.x = self.r*tf.math.cos(self.t)
        self.y = self.r*tf.math.sin(self.t) 
        self.antisym_dist  = tf.stack([self.x, self.y], axis=1)
        self.antisym_dist =  tf.reshape(self.antisym_dist, (-1, 2, self.channels*self.filters))
        self.var_x = tfp.math.find_root_chandrupatla(objective_fn=self._objective, low = [0], high=[1/n])[0]
        
        self.antisym_rotation = tf.Variable(initial_value = tf.linspace(np.pi/2, np.pi/2, self.filters, axis=0),    
                                            dtype='float32', trainable=False, name="antisym_rotation")

        R = tf.stack([tf.stack([tf.math.cos(-m.pi/4 + self.antisym_rotation ), -tf.math.sin(-m.pi/4  + self.antisym_rotation)], axis = -1),     
                     tf.stack([tf.math.sin(-m.pi/4  + self.antisym_rotation),  tf.math.cos(-m.pi/4  + self.antisym_rotation)], axis=-1)], axis= -1)
        R = tf.cast(R,  tf.dtypes.float32)
        
        self.cov = tf.stack([self.var_x,          self.rho*self.var_x,      
                        self.rho*self.var_x,          self.var_x ])
        self.cov = tf.cast(tf.reshape(self.cov, (1, 2,2)), tf.dtypes.float32)
        
        cov = tf.matmul(tf.matmul(R, self.cov), R,  transpose_b=True)

        self.antisym_dist  = tf.squeeze(tf.stack([self.x, self.y], axis=1), axis=0)


        self.antisym_dist  = tf.transpose(self.antisym_dist, perm=[2, 0, 1])
        self.antisym_dist = self._color(self.antisym_dist, cov)

        self.antisym_dist  = tf.expand_dims(tf.transpose(self.antisym_dist, perm=[2, 0, 1]), axis=0)


        x, y = self.antisym_dist[0,:,:,0], self.antisym_dist[0,:,:,1]

        ra = tf.math.sqrt(x**2 + y**2)
        theta = tf.expand_dims(tf.math.atan2(y, x), axis=0)

        a = -tf.math.sqrt(8.0)*tf.math.cos(theta - 9*m.pi/4)
        b = -2*tf.math.sin(theta)
        c = -tf.math.sqrt(8.0)*tf.math.sin(theta - 9*m.pi/4)
        d = -2*tf.math.cos(theta)
        
        self.asym_filters = tf.stack([tf.concat( [a,b,c], axis=0) , 
                        tf.concat( [d,tf.zeros([1, self.channels, self.filters]), -d], axis=0),
                        tf.concat( [-c, -b, -a], axis=0)])
                
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.asym_filters), axis=[0,1]))  
        self.asym_filters = tf.math.multiply((self.asym_filters / norm) , ra)


        #symetric initialization
        std_s = (self.var_rs2/64)**(1/4)
        a = tf.random.normal([1, self.channels, self.filters], stddev = std_s, dtype=tf.dtypes.float32, seed = self.seed)
        b = tf.random.normal([1, self.channels, self.filters], stddev = std_s,  dtype=tf.dtypes.float32, seed = self.seed)
        c = tf.random.normal([1, self.channels, self.filters], stddev = std_s,  dtype=tf.dtypes.float32, seed = self.seed)

        self.sym_filters = tf.stack([tf.concat([a, b, a],  axis=0), 
                                     tf.concat([b, c, b], axis=0),
                                     tf.concat([a, b, a], axis=0)])
        
        self.w = tf.Variable(initial_value=self.asym_filters + self.sym_filters,  trainable=False, name="weights")

        #Bias Initialization
        if self.use_bias:
            self.bias = tf.Variable(
                initial_value=self.bias_initializer(shape=(self.filters,), 
                                                    dtype='float32'),
                trainable=True, name="bias" )


    def call(self, inputs, training=None):

        R = tf.stack([tf.stack([tf.math.cos(-m.pi/4 + self.antisym_rotation ), -tf.math.sin(-m.pi/4  + self.antisym_rotation)], axis = -1),     
                     tf.stack([tf.math.sin(-m.pi/4  + self.antisym_rotation),  tf.math.cos(-m.pi/4  + self.antisym_rotation)], axis=-1)], axis= -1)
        R = tf.cast(R,  tf.dtypes.float32)
        cov = tf.matmul(tf.matmul(R, self.cov), R,  transpose_b=True)

        self.antisym_dist  = tf.squeeze(tf.stack([self.x, self.y], axis=1), axis=0)


        self.antisym_dist  = tf.transpose(self.antisym_dist, perm=[2, 0, 1])
        self.antisym_dist = self._color(self.antisym_dist, cov)

        self.antisym_dist  = tf.expand_dims(tf.transpose(self.antisym_dist, perm=[2, 0, 1]), axis=0)


        x, y = self.antisym_dist[0,:,:,0], self.antisym_dist[0,:,:,1]

        ra = tf.math.sqrt(x**2 + y**2)
        theta = tf.expand_dims(tf.math.atan2(y, x), axis=0)

        a = -tf.math.sqrt(8.0)*tf.math.cos(theta - 9*m.pi/4)
        b = -2*tf.math.sin(theta)
        c = -tf.math.sqrt(8.0)*tf.math.sin(theta - 9*m.pi/4)
        d = -2*tf.math.cos(theta)
        
        self.asym_filters = tf.stop_gradient(tf.stack([tf.concat( [a,b,c], axis=0) , 
                        tf.concat( [d,tf.zeros([1, self.channels, self.filters]), -d], axis=0),
                        tf.concat( [-c, -b, -a], axis=0)]))
                
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.asym_filters), axis=[0,1]))  
        self.asym_filters = tf.math.multiply((self.asym_filters / norm) , ra)

        self.w.assign(self.asym_filters + self.sym_filters)


        x =  tf.nn.conv2d(inputs, self.w , strides=self.strides, 
                          padding=self.padding)

        if self.use_bias:
            x = x+self.bias

        if self.activation :
            return  self.activation(x)  #self.activation(tf.math.add(x_a, x_s)) #, self.map
        else:
            return x
        