'''
Created on Jun 8, 2017

@author: jendrik
'''


from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import functools
from .Layer.Layer import ConvolutionLayer, DeconvolutionLayer, ConvolutionalBatchNormalization
from tensorflow.python.training import moving_averages
from Network.Layer.Layer import SplittedDeconvolutionLayer
#from tf.contrib.keras import MaxPooling2D

class Network(ABC):
    '''
    A class representation of a neural network
    '''


    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.avg = None
        self.groundTruths = []
    
    @abstractmethod
    def setWeights(self): pass
        
    def train(self, context, groundTruth, weights, sess):
        
        trainLoss, trainAcc, crossEntropy, imu, _ = sess.run([self.loss, self.accuracy, self.crossEntropy, self.imu, self.trainStep], 
                                                        feed_dict={self.inputs[0]: context,
                                                                self.labels: groundTruth, self.trainPh: True,
                                                                self.keepProbSpatial: .7, self.keepProb: .6, self.keepProbAE: .5, self.weights: weights})
        print(imu)
        return trainLoss, trainAcc, crossEntropy, imu
    
    def val(self, context, groundTruth, weights, sess):
        valLoss, valAcc, crossEntropy, imu = sess.run([self.loss, self.accuracy, self.crossEntropy, self.imu], feed_dict={self.inputs[0]: context, 
                                                                self.labels: groundTruth, self.trainPh: False,
                                                                self.keepProbSpatial: 1., self.keepProb: 1.,self.keepProbAE: 1., self.weights: weights})
        return valLoss, valAcc, crossEntropy, imu
    
    def eval(self,context, sess):
        return sess.run([self.outputs[0]], feed_dict={self.inputs[0]: context, self.trainPh: False,
                                                                 self.keepProbSpatial: 1., self.keepProb: 1.,self.keepProbAE: 1.})
    
    def evalWithAverage(self, context, sess):
        return sess.run([self.avg], feed_dict={self.inputs[0]: context, self.trainPh: False,
                                                                 self.keepProbSpatial: 1., self.keepProb: 1.,self.keepProbAE: 1.})

class SegNet(Network):
    
    def __init__(self, inputShape, numberOfClasses, learningRate, globalStep):
        super(SegNet, self).__init__()
        self.numberOfClasses = numberOfClasses
        self.inputs.append(tf.placeholder(tf.float32, shape=(None,inputShape[1],inputShape[2],inputShape[3]), name='inputImage'))
        
        self.groundTruths.append(tf.placeholder(tf.int32, shape=(None, int(inputShape[1])*int(inputShape[2])), name='gtImage'))
        self.weights = tf.placeholder(tf.float32, shape=(None, int(inputShape[1])*int(inputShape[2])), name='weights')
        self.trainPh = tf.placeholder(tf.bool, name='training_phase')
        self.keepProbSpatial = tf.placeholder(tf.float32, name='keepProbSpatial')
        self.keepProb = tf.placeholder(tf.float32, name='keepProb')
        self.keepProbAE = tf.placeholder(tf.float32, name='keepProbAE')
        self.layers = {}
        self.learningRate = learningRate
        self.globalStep = globalStep
        
        number = 0
        
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, 3, 32, stride=(1,1), padding='SAME', useBias=False, trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(self.inputs[0])
            bn = ConvolutionalBatchNormalization(number, 32,trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
            xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        
        print(xC.get_shape())
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 64, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
            
        xC4 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        number += 1
        
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 64, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        xC3 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        number += 1
        
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 128, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        xC2 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')

        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        xC1 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 1024, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 1024, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        """number += 1
        conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
        self.layers.update({conv.name: conv})
        xC = conv(xC)
        bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
        self.layers.update({bn.name: bn})
        xC = bn(xC)
        xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 1024, stride=(1,1), useBias=False, padding='SAME', trainable=False)
        self.layers.update({conv.name: conv})
        xC = conv(xC)
        bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
        self.layers.update({bn.name: bn})
        xC = bn(xC)
        xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)"""
        
        xC = tf.nn.dropout(xC, keep_prob=self.keepProbSpatial, noise_shape=[tf.shape(xC)[0], 1, 1,int( xC.get_shape()[-1])])
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 1024, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('localdeconvLayer%d'%number) as namespace:
            conv = SplittedDeconvolutionLayer(number, 1, xC.get_shape()[-1], 512, stride=(1,1), padding='SAME', useBias=False)
            #conv.addAutoencodeLoss(xC, tf.nn.elu, self.trainPh)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        with tf.variable_scope('upsample%d'%number):
            xC = tf.contrib.keras.layers.UpSampling2D(size=(2,2))(xC)
            if(xC.get_shape()[1] != xC1.get_shape()[1]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((int(xC1.get_shape()[1]) - int(xC.get_shape()[1]),0),(0,0)))(xC)
            if(xC.get_shape()[2] != xC1.get_shape()[2]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((0,0),(int(xC1.get_shape()[2]) - int(xC.get_shape()[2]),0)))(xC)
            xC = tf.concat([xC, xC1], axis=3)
            xC = tf.nn.dropout(xC, keep_prob=self.keepProbSpatial, noise_shape=[tf.shape(xC)[0], 1, 1,int( xC.get_shape()[-1])])
            
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('localdeconvLayer%d'%number) as namespace:
            conv = SplittedDeconvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), padding='SAME', useBias=False)
            #conv.addAutoencodeLoss(xC, tf.nn.elu, self.trainPh)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('localdeconvLayer%d'%number) as namespace:
            conv = SplittedDeconvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), padding='SAME', useBias=False)
            #conv.addAutoencodeLoss(xC, tf.nn.elu, self.trainPh, self.keepProbAE)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 256, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        with tf.variable_scope('upsample%d'%number):
            xC = tf.contrib.keras.layers.UpSampling2D(size=(2,2))(xC)
            if(xC.get_shape()[1] != xC2.get_shape()[1]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((int(xC2.get_shape()[1]) - int(xC.get_shape()[1]),0),(0,0)))(xC)
            if(xC.get_shape()[2] != xC2.get_shape()[2]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((0,0),(int(xC2.get_shape()[2]) - int(xC.get_shape()[2]),0)))(xC)
            xC = tf.concat([xC, xC2], axis=3)
            xC = tf.nn.dropout(xC, keep_prob=self.keepProbSpatial, noise_shape=[tf.shape(xC)[0], 1, 1,int( xC.get_shape()[-1])])
            print(xC.get_shape())
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
            
        number += 1
        with tf.variable_scope('localdeconvLayer%d'%number) as namespace:
            conv = SplittedDeconvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), padding='SAME', useBias=False)
            conv.addAutoencodeLoss(xC, tf.nn.elu, self.trainPh, self.keepProbAE)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        with tf.variable_scope('upsample%d'%number):
            xC = tf.contrib.keras.layers.UpSampling2D(size=(2,2))(xC)
            if(xC.get_shape()[1] != xC3.get_shape()[1]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((int(xC3.get_shape()[1]) - int(xC.get_shape()[1]),0),(0,0)))(xC)
            if(xC.get_shape()[2] != xC3.get_shape()[2]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((0,0),(int(xC3.get_shape()[2]) - int(xC.get_shape()[2]),0)))(xC)
            xC = tf.concat([xC, xC3], axis=3)
            xC = tf.nn.dropout(xC, keep_prob=self.keepProbSpatial, noise_shape=[tf.shape(xC)[0], 1, 1,int( xC.get_shape()[-1])])
            print(xC.get_shape())
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 256, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('localdeconvLayer%d'%number) as namespace:
            conv = SplittedDeconvolutionLayer(number, 1, xC.get_shape()[-1], 128, stride=(1,1), padding='SAME', useBias=False)
            conv.addAutoencodeLoss(xC, tf.nn.elu, self.trainPh, self.keepProbAE)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        with tf.variable_scope('upsample%d'%number):
            xC = tf.contrib.keras.layers.UpSampling2D(size=(2,2))(xC)
            print(xC.get_shape(), inputShape[1]//2, inputShape[2]//2)
            if(xC.get_shape()[1] != inputShape[1]//2):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((inputShape[1]//2 - int(xC.get_shape()[1]),0),(0,0)))(xC)
            if(xC.get_shape()[2] != inputShape[2]//2):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((0,0),(inputShape[2]//2 - int(xC.get_shape()[2]),0)))(xC)
            xC = tf.concat([xC, xC4], axis=3)
            xC = tf.nn.dropout(xC, keep_prob=self.keepProbSpatial, noise_shape=[tf.shape(xC)[0], 1, 1,int( xC.get_shape()[-1])])
            
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            conv.addAutoencodeLoss(xC, tf.nn.elu, self.trainPh, self.keepProbAE)
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], self.trainPh)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
            
        with tf.variable_scope('upsample%d'%number):
            xC = tf.nn.dropout(xC, keep_prob=self.keepProb)
            
            xC = tf.contrib.keras.layers.UpSampling2D(size=(2,2))(xC)
            print(xC.get_shape(), inputShape)
            if(xC.get_shape()[1] != inputShape[1]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((inputShape[1] - int(xC.get_shape()[1]),0),(0,0)))(xC)
            if(xC.get_shape()[2] != inputShape[2]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((0,0),(inputShape[2] - int(xC.get_shape()[2]),0)))(xC)
                
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], numberOfClasses, stride=(1,1), padding='SAME', useBias=False, )
            #conv.addAutoencodeLoss(xC, tf.nn.elu, self.trainPh)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
        
        logits = tf.reshape(xC, (-1, inputShape[1]*inputShape[2], numberOfClasses), name='outputImage')
        self.softmax = tf.nn.softmax(logits, dim=2)
        mean, var = tf.nn.moments(self.softmax, axes = [0])
        constZero = tf.constant(0.0, shape=[inputShape[1]*inputShape[2], self.numberOfClasses])
        avg = tf.Variable(constZero, trainable = False, name = "movingResults")

        self.updateAvg = moving_averages.assign_moving_average(
                        avg, mean, .7)           
        with tf.control_dependencies([self.updateAvg]):
            self.avg = tf.argmax(avg,axis=1, name='avgOutput')
        self.outputs.append(tf.argmax(self.softmax, axis=2, name='output'))
        
        self.labels = tf.placeholder(tf.int64, shape=(None, inputShape[1]* inputShape[2]))
        #weightsPh = tf.placeholder(tf.float32, shape=(weights.shape[1]))
        print(self.outputs[0].get_shape(), self.labels.get_shape())
        
        crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
        self.crossEntropy = tf.reduce_mean(crossEntropy)
        loss = tf.reduce_mean(tf.multiply(crossEntropy, self.weights))
        tf.add_to_collection('my_losses', tf.multiply(.1,loss))
        
        imu, self.imuOp = tf.metrics.mean_iou(self.labels, tf.argmax(logits, axis = 2), numberOfClasses, 
                                         self.weights, name = 'meanIMU')
        with tf.control_dependencies([self.imuOp]):
            self.imu = tf.subtract(tf.constant(1.), imu)
            tf.add_to_collection('my_losses', self.imu)
        self.loss = tf.reduce_sum(tf.stack(tf.get_collection('my_losses')))
        #self.loss = tf.scan(lambda a, x: tf.scalar_mul(x[0], x[1]), (weightsPh,loss))
         
        #self.trainStep = tf.train.MomentumOptimizer(learning_rate=self.learningRate, momentum=0.9).minimize(self.loss, self.globalStep)
        self.trainStep = tf.train.AdamOptimizer(5e-4).minimize(self.loss)
        #with tf.control_dependencies(self.movingVars):
        #    self.trainStep = trainStep
        correctPrediction = tf.equal(self.outputs, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        
        
    def setWeights(self, weightFile, sess):
        weightFile = open(weightFile, 'rb')  # ../darknet19_448.weights
        weights_header = np.ndarray(
                shape=(4,), dtype='int32', buffer=weightFile.read(16))
        print('Weights Header: ', weights_header)
        weightLoader = functools.partial(SegNet.load_weights, weightFile=weightFile)
        for i in range(16):
            conv = self.layers['conv%d'%i]
            weights = weightLoader(int(conv.dnshape[3]), int(conv.ksize), True, int(conv.dnshape[2]))
            conv.setWeights(weights)
            bn = self.layers['bn%d'%i]
            bn.setWeights(weights)
        print(tf.get_collection("assignOps"))
        sess.run(tf.get_collection("assignOps"))
        print('Unused Weights: ', len(weightFile.read()) / 4)
        
        
    @staticmethod
    def load_weights(filters, size, batchNormalisation, prevLayerFilter, weightFile):
        weights = {}
        weights_shape = (size, size, prevLayerFilter, filters)
        # Caffe weights have a different order:
        darknet_w_shape = (filters, weights_shape[2], size, size)
        weights_size = np.product(weights_shape)
        print(weights_shape)
        print(weights_size)
        
        conv_bias = np.ndarray(
                shape=(filters, ),
                    dtype='float32',
                    buffer=weightFile.read(filters * 4))
        weights.update({'bias' :conv_bias})
        
        if batchNormalisation:
            bn_weights = np.ndarray(
                shape=(3, filters),
                dtype='float32',
                buffer=weightFile.read(filters * 12))
            
            weights.update({'gamma' :bn_weights[0]})
            weights.update({'movingMean' :bn_weights[1]})
            weights.update({'movingVariance' :bn_weights[2]})
    
        conv_weights = np.ndarray(
            shape=darknet_w_shape,
            dtype='float32',
            buffer=weightFile.read(weights_size * 4))
    
        # DarkNet conv_weights are serialized Caffe-style:
        # (out_dim, in_dim, height, width)
        # We would like to set these to Tensorflow order:
        # (height, width, in_dim, out_dim)
        # TODO: Add check for Theano dim ordering.
        conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
        weights.update({'kernel' :conv_weights})
        return weights
    
class SegNetInference(Network):
    
    def __init__(self, inputShape, numberOfClasses, learningRate, globalStep, withAverage=True):
        super(SegNetInference, self).__init__()
        self.numberOfClasses = numberOfClasses
        self.inputs.append(tf.placeholder(tf.float32, shape=(None,inputShape[1],inputShape[2],inputShape[3]), name='inputImage'))
        
        self.groundTruths.append(tf.placeholder(tf.int32, shape=(None, int(inputShape[1])*int(inputShape[2])), name='gtImage'))
        self.weights = tf.placeholder(tf.float32, shape=(None, int(inputShape[1])*int(inputShape[2])), name='weights')
        self.trainPh = tf.placeholder(tf.bool, name='training_phase')
        self.keepProbSpatial = tf.placeholder(tf.float32, name='keepProbSpatial')
        self.keepProb = tf.placeholder(tf.float32, name='keepProb')
        self.keepProbAE = tf.placeholder(tf.float32, name='keepProbAE')
        self.layers = {}
        self.learningRate = learningRate
        self.globalStep = globalStep
        
        number = 0
        
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, 3, 32, stride=(1,1), padding='SAME', useBias=False, trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(self.inputs[0])
            bn = ConvolutionalBatchNormalization(number, 32,trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
            xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        
        print(xC.get_shape())
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 64, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
            
        xC4 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        number += 1
        
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 64, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        xC3 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        number += 1
        
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 128, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        xC2 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')

        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        xC1 = xC
        
        xC = tf.nn.max_pool(xC, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 1024, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        with tf.variable_scope('convLayer%d'%number) as namespace:
            conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 1024, stride=(1,1), useBias=False, padding='SAME', trainable=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        """number += 1
        conv = ConvolutionLayer(number, 1, xC.get_shape()[-1], 512, stride=(1,1), useBias=False, padding='SAME', trainable=False)
        self.layers.update({conv.name: conv})
        xC = conv(xC)
        bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
        self.layers.update({bn.name: bn})
        xC = bn(xC)
        xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)
        
        number += 1
        conv = ConvolutionLayer(number, 3, xC.get_shape()[-1], 1024, stride=(1,1), useBias=False, padding='SAME', trainable=False)
        self.layers.update({conv.name: conv})
        xC = conv(xC)
        bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1],trainable=False)
        self.layers.update({bn.name: bn})
        xC = bn(xC)
        xC = tf.maximum(.1 * xC, xC, name = 'relu%d'%number)"""
        
        xC = tf.nn.dropout(xC, keep_prob=self.keepProbSpatial, noise_shape=[tf.shape(xC)[0], 1, 1,int( xC.get_shape()[-1])])
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 1024, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('localdeconvLayer%d'%number) as namespace:
            conv = SplittedDeconvolutionLayer(number, 1, xC.get_shape()[-1], 512, stride=(1,1), padding='SAME', useBias=False)
            #conv.addAutoencodeLoss(xC, tf.nn.elu, trainable = False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        with tf.variable_scope('upsample%d'%number):
            xC = tf.contrib.keras.layers.UpSampling2D(size=(2,2))(xC)
            if(xC.get_shape()[1] != xC1.get_shape()[1]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((int(xC1.get_shape()[1]) - int(xC.get_shape()[1]),0),(0,0)))(xC)
            if(xC.get_shape()[2] != xC1.get_shape()[2]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((0,0),(int(xC1.get_shape()[2]) - int(xC.get_shape()[2]),0)))(xC)
            xC = tf.concat([xC, xC1], axis=3)
            xC = tf.nn.dropout(xC, keep_prob=self.keepProbSpatial, noise_shape=[tf.shape(xC)[0], 1, 1,int( xC.get_shape()[-1])])
            
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('localdeconvLayer%d'%number) as namespace:
            conv = SplittedDeconvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), padding='SAME', useBias=False)
            #conv.addAutoencodeLoss(xC, tf.nn.elu, trainable = False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('localdeconvLayer%d'%number) as namespace:
            conv = SplittedDeconvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), padding='SAME', useBias=False)
            #conv.addAutoencodeLoss(xC, tf.nn.elu, trainable = False, self.keepProbAE)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 256, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        with tf.variable_scope('upsample%d'%number):
            xC = tf.contrib.keras.layers.UpSampling2D(size=(2,2))(xC)
            if(xC.get_shape()[1] != xC2.get_shape()[1]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((int(xC2.get_shape()[1]) - int(xC.get_shape()[1]),0),(0,0)))(xC)
            if(xC.get_shape()[2] != xC2.get_shape()[2]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((0,0),(int(xC2.get_shape()[2]) - int(xC.get_shape()[2]),0)))(xC)
            xC = tf.concat([xC, xC2], axis=3)
            xC = tf.nn.dropout(xC, keep_prob=self.keepProbSpatial, noise_shape=[tf.shape(xC)[0], 1, 1,int( xC.get_shape()[-1])])
            print(xC.get_shape())
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 512, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
            
        number += 1
        with tf.variable_scope('localdeconvLayer%d'%number) as namespace:
            conv = SplittedDeconvolutionLayer(number, 1, xC.get_shape()[-1], 256, stride=(1,1), padding='SAME', useBias=False)
            conv.addAutoencodeLoss(xC, tf.nn.elu, self.trainPh, self.keepProbAE)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        with tf.variable_scope('upsample%d'%number):
            xC = tf.contrib.keras.layers.UpSampling2D(size=(2,2))(xC)
            if(xC.get_shape()[1] != xC3.get_shape()[1]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((int(xC3.get_shape()[1]) - int(xC.get_shape()[1]),0),(0,0)))(xC)
            if(xC.get_shape()[2] != xC3.get_shape()[2]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((0,0),(int(xC3.get_shape()[2]) - int(xC.get_shape()[2]),0)))(xC)
            xC = tf.concat([xC, xC3], axis=3)
            xC = tf.nn.dropout(xC, keep_prob=self.keepProbSpatial, noise_shape=[tf.shape(xC)[0], 1, 1,int( xC.get_shape()[-1])])
            print(xC.get_shape())
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 256, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('localdeconvLayer%d'%number) as namespace:
            conv = SplittedDeconvolutionLayer(number, 1, xC.get_shape()[-1], 128, stride=(1,1), padding='SAME', useBias=False)
            conv.addAutoencodeLoss(xC, tf.nn.elu, self.trainPh, self.keepProbAE)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
        with tf.variable_scope('upsample%d'%number):
            xC = tf.contrib.keras.layers.UpSampling2D(size=(2,2))(xC)
            print(xC.get_shape(), inputShape[1]//2, inputShape[2]//2)
            if(xC.get_shape()[1] != inputShape[1]//2):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((inputShape[1]//2 - int(xC.get_shape()[1]),0),(0,0)))(xC)
            if(xC.get_shape()[2] != inputShape[2]//2):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((0,0),(inputShape[2]//2 - int(xC.get_shape()[2]),0)))(xC)
            xC = tf.concat([xC, xC4], axis=3)
            xC = tf.nn.dropout(xC, keep_prob=self.keepProbSpatial, noise_shape=[tf.shape(xC)[0], 1, 1,int( xC.get_shape()[-1])])
            
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], 128, stride=(1,1), padding='SAME', useBias=False)
            self.layers.update({conv.name: conv})
            conv.addAutoencodeLoss(xC, tf.nn.elu, self.trainPh, self.keepProbAE)
            xC = conv(xC)
            bn = ConvolutionalBatchNormalization(number, xC.get_shape()[-1], trainable = False)
            self.layers.update({bn.name: bn})
            xC = bn(xC)
            xC = tf.nn.elu(xC)
            
        with tf.variable_scope('upsample%d'%number):
            xC = tf.nn.dropout(xC, keep_prob=self.keepProb)
            
            xC = tf.contrib.keras.layers.UpSampling2D(size=(2,2))(xC)
            print(xC.get_shape(), inputShape)
            if(xC.get_shape()[1] != inputShape[1]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((inputShape[1] - int(xC.get_shape()[1]),0),(0,0)))(xC)
            if(xC.get_shape()[2] != inputShape[2]):
                xC = tf.contrib.keras.layers.ZeroPadding2D(padding=((0,0),(inputShape[2] - int(xC.get_shape()[2]),0)))(xC)
                
        
        number += 1
        with tf.variable_scope('deconvLayer%d'%number) as namespace:
            conv = DeconvolutionLayer(number, 3, xC.get_shape()[-1], numberOfClasses, stride=(1,1), padding='SAME', useBias=False, )
            #conv.addAutoencodeLoss(xC, tf.nn.elu, trainable = False)
            self.layers.update({conv.name: conv})
            xC = conv(xC)
        
        logits = tf.reshape(xC, (-1, inputShape[1]*inputShape[2], numberOfClasses), name='outputImage')
        self.softmax = tf.nn.softmax(logits, dim=2)
        if(withAverage):
            mean, var = tf.nn.moments(self.softmax, axes = [0])
            constZero = tf.constant(0.0, shape=[inputShape[1]*inputShape[2], self.numberOfClasses])
            avg = tf.Variable(constZero, trainable = False, name = "movingResults")
            
            self.updateAvg = moving_averages.assign_moving_average(
                            avg, mean, .7)           
            with tf.control_dependencies([self.updateAvg]):
                self.avg = tf.argmax(avg,axis=1, name='avgOutput')
        self.outputs.append(tf.argmax(self.softmax, axis=2, name='output'))
        
        self.labels = tf.placeholder(tf.int64, shape=(None, inputShape[1]* inputShape[2]))
        #weightsPh = tf.placeholder(tf.float32, shape=(weights.shape[1]))
        print(self.outputs[0].get_shape(), self.labels.get_shape())
        
        crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
        self.crossEntropy = tf.reduce_mean(crossEntropy)
        loss = tf.reduce_mean(tf.multiply(crossEntropy, self.weights))
        tf.add_to_collection('my_losses', tf.multiply(.1,loss))
        
        imu, self.imuOp = tf.metrics.mean_iou(self.labels, tf.argmax(logits, axis = 2), numberOfClasses, 
                                         self.weights, name = 'meanIMU')
        with tf.control_dependencies([self.imuOp]):
            self.imu = tf.subtract(tf.constant(1.), imu)
            tf.add_to_collection('my_losses', self.imu)
        self.loss = tf.reduce_sum(tf.stack(tf.get_collection('my_losses')))
        #self.loss = tf.scan(lambda a, x: tf.scalar_mul(x[0], x[1]), (weightsPh,loss))
         
        #self.trainStep = tf.train.MomentumOptimizer(learning_rate=self.learningRate, momentum=0.9).minimize(self.loss, self.globalStep)
        self.trainStep = tf.train.AdamOptimizer(5e-4).minimize(self.loss)
        #with tf.control_dependencies(self.movingVars):
        #    self.trainStep = trainStep
        correctPrediction = tf.equal(self.outputs, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        
        
    def setWeights(self, weightFile, sess):
        weightFile = open(weightFile, 'rb')  # ../darknet19_448.weights
        weights_header = np.ndarray(
                shape=(4,), dtype='int32', buffer=weightFile.read(16))
        print('Weights Header: ', weights_header)
        weightLoader = functools.partial(SegNet.load_weights, weightFile=weightFile)
        for i in range(16):
            conv = self.layers['conv%d'%i]
            weights = weightLoader(int(conv.dnshape[3]), int(conv.ksize), True, int(conv.dnshape[2]))
            conv.setWeights(weights)
            bn = self.layers['bn%d'%i]
            bn.setWeights(weights)
        print(tf.get_collection("assignOps"))
        sess.run(tf.get_collection("assignOps"))
        print('Unused Weights: ', len(weightFile.read()) / 4)
        
        
    @staticmethod
    def load_weights(filters, size, batchNormalisation, prevLayerFilter, weightFile):
        weights = {}
        weights_shape = (size, size, prevLayerFilter, filters)
        # Caffe weights have a different order:
        darknet_w_shape = (filters, weights_shape[2], size, size)
        weights_size = np.product(weights_shape)
        print(weights_shape)
        print(weights_size)
        
        conv_bias = np.ndarray(
                shape=(filters, ),
                    dtype='float32',
                    buffer=weightFile.read(filters * 4))
        weights.update({'bias' :conv_bias})
        
        if batchNormalisation:
            bn_weights = np.ndarray(
                shape=(3, filters),
                dtype='float32',
                buffer=weightFile.read(filters * 12))
            
            weights.update({'gamma' :bn_weights[0]})
            weights.update({'movingMean' :bn_weights[1]})
            weights.update({'movingVariance' :bn_weights[2]})
    
        conv_weights = np.ndarray(
            shape=darknet_w_shape,
            dtype='float32',
            buffer=weightFile.read(weights_size * 4))
    
        # DarkNet conv_weights are serialized Caffe-style:
        # (out_dim, in_dim, height, width)
        # We would like to set these to Tensorflow order:
        # (height, width, in_dim, out_dim)
        # TODO: Add check for Theano dim ordering.
        conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
        weights.update({'kernel' :conv_weights})
        return weights
    
    
    
    
    
    
    
    
    
    
    
