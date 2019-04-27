'''
Created on Jan 20, 2017

@author: jjordening
'''

import tensorflow as tf

def weightVariable(shape, name=''):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def biasVariable(shape, name=''):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name=name)