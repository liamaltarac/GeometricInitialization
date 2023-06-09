import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K  

import tensorflow_addons as tfa
import numpy as np
import sys
class Mutual_Information_Objective():
    def __init__(self, num_features, num_classes=10, conv_features=True):

        super(Mutual_Information_Objective, self).__init__()
        self.n_features = num_features
        self.n_classes = num_classes
        #self.prob = input_is_probability
        self.conv_features =  conv_features  #Is training using the output of convolution layer 

        self.centroid_table= tf.Variable(tf.zeros([ self.n_classes,  self.n_features]), trainable=False)
        #self.prob_mat= tf.Variable(tf.zeros([ self.n_classes,  self.n_classes]), trainable=False)


        self.num_features =  0
        tf.print("Starting")
    
    @tf.function()
    def maximize(self, y_true, y_pred):
        #tf.print(y_true)
        #tf.print(y_pred.shape)
        #norms = tf.zeros([0, 1])
        #centroids = tf.zeros([0, 100])

        def body(i, label, centroids):
            #Find all latent codes in batch belonging a specific class
            idx = tf.where(y_true == [label])[:,0]
            #tf.print(label)

            if tf.equal(tf.size(idx), 0) == False:
                #tf.print(idx)
                gathered_vectors = tf.gather(y_pred,  idx)

                #tf.io.write_file("Math_Experiments/dot.csv", tf.strings.as_string(prob_mat))
                centroid = tf.reduce_mean(gathered_vectors, axis=0)
                #tf.io.write_file("Math_Experiments/dot.csv", tf.strings.as_string(centroid))
                if self.conv_features:
                    #centroid = tf.keras.layers.GlobalAveragePooling2D()(tf.expand_dims(centroid, axis=0))
                    centroid = tf.reshape(centroid, [self.n_features])
                    #tf.io.write_file("Math_Experiments/dot.csv", tf.strings.as_string(centroid))

                    #centroid = K.clip(centroid, -1, 1, )

                    '''scale = centroid.shape[-1]//self.n_classes
                    centroid = tf.keras.layers.AveragePooling1D(pool_size=scale,
                                strides=scale, padding='valid')(tf.expand_dims(centroid, axis=-1))'''
                
                #centroid = tf.reshape(centroid, [self.n_classes])

                #if self.prob is False:
                #    centroid = tf.nn.softmax(centroid)
                

                centroid_sum = tf.math.reduce_sum(centroid)
                centroid_sum = K.clip(centroid_sum , K.epsilon(), centroid_sum )

                centroid = centroid / centroid_sum #( ensure that sum=1)

                #if prob is False:
                #    centroid = tf.nn.softmax(centroid)
                centroids = centroids.write(label, centroid)
            
            label += 1
            return i, label, centroids


        centroids = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=0, 
                                        dynamic_size=True, name="centroids",  clear_after_read=False) 
        

        centroids = centroids.unstack(self.centroid_table)

        i, _, centroids = tf.while_loop(lambda i, label, *_: tf.less(label, self.n_classes), body,[0,0, centroids] ,parallel_iterations=1)

        centroid_table =  centroids.stack()
        self.centroid_table.assign(centroid_table)
        
        prob_mat = K.clip(centroid_table / self.n_classes , K.epsilon(), 1, ) #Ensure that marginals sum to 1 and that the zeros are not actually 0 (prevents nan)
        #prob_mat = self.centroid_table / self.n_classes 
        prob_mat = tf.reshape(prob_mat, [self.n_classes, self.n_features])

        px = tf.expand_dims(tf.math.reduce_sum(prob_mat, axis=1), axis=0)
        px = tf.reshape(px, [1,  self.n_classes])
        py = tf.expand_dims(tf.math.reduce_sum(prob_mat, axis=0), axis=0)
        py = tf.reshape(py, [1,  self.n_features])

        #tf.print(py)
        px_py = tf.linalg.matmul(px , py, transpose_a=True)  #Px * Py
        #tf.io.write_file("Math_Experiments/dot.csv", tf.strings.as_string(prob_mat))


        MI = tf.math.reduce_sum(tf.math.multiply(prob_mat, tf.math.log(tf.math.divide(prob_mat, px_py))))
        #centroids.close()

        #self.centroids.close()
        #tf.print(angle)

        #tf.print(norms)

        return(-MI)