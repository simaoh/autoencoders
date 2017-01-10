import numpy as np
import tensorflow as tf
from layers import encode,decode, variable_summaries

#parameters
n_input=784

def autoencoder(x,dropout,noise_std):
    
    #Variables - filters  
    with tf.name_scope('weights'):
        #weights={'h1': tf.Variable(tf.truncated_normal([3,3,1,10],stddev=0.1)),
        #         'h2': tf.Variable(tf.truncated_normal([3,3,10,10],stddev=0.1)),
        #         'h3': tf.Variable(tf.truncated_normal([3,3,10,10],stddev=0.1))}
        weights={'h1': tf.Variable(tf.random_uniform([3,3,1,10],-1.0 /np.sqrt(10), 1.0 / np.sqrt(10))),
                 'h2': tf.Variable(tf.random_uniform([3,3,10,10],-1.0 / np.sqrt(10),1.0 / np.sqrt(10))),
                 'h3': tf.Variable(tf.random_uniform([3,3,10,10],-1.0 / np.sqrt(10),1.0 / np.sqrt(10)))}
        for a_layer in weights:
            with tf.name_scope('weights_'+a_layer):
                variable_summaries(weights[a_layer])
    #Variables - biases
    with tf.name_scope('biases'):        
        biases={'h1': tf.Variable(tf.zeros([10])),
                'h2': tf.Variable(tf.zeros([10])),
                'h3': tf.Variable(tf.zeros([10])),
                'h4': tf.Variable(tf.zeros([10])),
                'h5': tf.Variable(tf.zeros([10])),
                'h6': tf.Variable(tf.zeros([1]))}
        for a_layer in biases:
            with tf.name_scope('bias_'+a_layer):
                variable_summaries(biases[a_layer])
    
    
    #reshape input
    with tf.name_scope('input'):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        tf.summary.image('input', x, 10)
    
    #noise (training_time)
    with tf.name_scope('add_noise'):
        noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=noise_std, dtype=tf.float32) 
        x_noised=x+noise
        x=x_noised
    
    #encoding
    with tf.name_scope('encoding'):
        code=encode(x,weights, biases)
    
    #dropout (training_time)
    code=tf.nn.dropout(code,dropout)
    
    #decoding
    with tf.name_scope('decoding'):
        reconstruction=decode(code,weights, biases)
    
    #reshape
    tf.summary.image('output', reconstruction, 10)
    code=tf.reshape(code, [-1,tf.shape(code)[1]*tf.shape(code)[2]*tf.shape(code)[3]])
    reconstruction=tf.reshape(reconstruction, [-1,n_input])
    
    #logging
    tf.summary.histogram('code', code)
    tf.summary.histogram('reconstruction', reconstruction)
    return (code, reconstruction)