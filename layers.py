import numpy as np
import tensorflow as tf

#Tensorboard Variable stats
def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        #with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#generic weight variable
def weight_variable(w_shape,name=None):
    W=tf.Variable(tf.truncated_normal(w_shape, stddev=0.1), name=name)
    return W

## Layers

def conv2d(x, w, b, stride):
    conv_layer=tf.nn.conv2d(x, w, [1,stride,stride,1], padding="SAME")
    conv_layer=tf.add(conv_layer, b)
    
    out_channels=w.get_shape().as_list()[-1]
    mean, var = tf.nn.moments(conv_layer, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")
    
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv_layer, mean, var, beta, gamma, 0.005,
        scale_after_normalization=True)

    out = batch_norm
    return out
    

def maxpool2d(x,k):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1,k,k,1], padding="SAME")
   
    
def unpool(value, sh):
    """Unpooling adapted from https://github.com/tensorflow/tensorflow/issues/2169
    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """  
    dim = 2
    out = (tf.reshape(value, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
        out = tf.concat(i, [out, out])
    return out    


def encode(x,weights, biases):
    
    with tf.name_scope('layer_1'):
        layer_1=conv2d(x,weights['h1'],biases['h1'],stride=1)
        layer_1=tf.nn.relu(layer_1)
        layer_1=maxpool2d(layer_1,k=2)

    with tf.name_scope('layer_2'):
        layer_2=conv2d(layer_1,weights['h2'],biases['h2'],stride=1)
        layer_2=tf.nn.relu(layer_2)
        layer_2=maxpool2d(layer_2,k=2)
   
    with tf.name_scope('layer_3'):
        layer_3=conv2d(layer_2,weights['h3'],biases['h3'],stride=2)
        layer_3=tf.nn.relu(layer_3)
        
    return layer_3


def decode(x,weights, biases):
   
    with tf.name_scope('layer_4'):
        layer_4=tf.nn.relu(-tf.add(tf.nn.conv2d_transpose(x, weights['h3'],
                    tf.pack([tf.shape(x)[0], 7, 7, 10]),
                    strides=[1, 2, 2, 1], padding='SAME'), biases['h4']))
        
    with tf.name_scope('layer_5'):
        layer_5=unpool(layer_4, [-1,7,7,10])
        layer_5=tf.reshape(layer_5, [tf.shape(x)[0],14,14,10])
        layer_5=tf.nn.relu(-tf.add(tf.nn.conv2d_transpose(layer_5, weights['h2'],
                    tf.pack([tf.shape(layer_4)[0], 14, 14, 10]),
                    strides=[1, 1, 1, 1], padding='SAME'), biases['h5']))
        
    with tf.name_scope('layer_6'):
        layer_6=unpool(layer_5, [-1,14,14,10])
        layer_6=tf.reshape(layer_6, [tf.shape(x)[0],28,28,10])
        layer_6=tf.nn.relu(-tf.add(tf.nn.conv2d_transpose(layer_6, weights['h1'],
                    tf.pack([tf.shape(layer_5)[0], 28, 28, 1]),
                    strides=[1, 1,1, 1], padding='SAME'), biases['h6']))
        
    return layer_6