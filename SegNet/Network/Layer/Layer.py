'''
Created on Jun 8, 2017

@author: jendrik
'''

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

WEIGHT_REG_CONST = 1e-5

class Layer(ABC):
    '''
    classdocs
    '''


    """
        An object representing a layer of a neural network.
    """

    def __init__(self, *args):
        self._signature = list(args)
        self.number = list(args)[0]
        self.name = list(args)[1]
        self.w = dict() # weights
        self.h = dict() # placeholders
        self.wshape = dict() # weight shape
        self.wsize = dict() # weight size
        self.present()
        for var in self.wshape:
            shp = self.wshape[var]
            size = np.prod(shp)
            self.wsize[var] = size
    
    @abstractmethod
    def setWeights(self, weights): pass

    @property
    def signature(self):
        return self._signature
    
    def addAutoencodeLoss(self, inputVar, act, trainPh, keepProbAE, noise_shape=None, halfPrecision = False):
        val = tf.nn.dropout(inputVar, keepProbAE, noise_shape)
        val = self(val)
        val = ConvolutionalBatchNormalization(self.number, self.n, trainPh, halfPrecision)(val)
        val = act(val)
        val = self.inverseLayer()(val)
        val = ConvolutionalBatchNormalization(self.number, self.dnshape[-1], trainPh, halfPrecision)(val)
        val = act(val)
        loss = tf.scalar_mul(tf.constant(0.2), tf.reduce_mean(tf.losses.mean_squared_error(tf.nn.tanh(inputVar), tf.nn.tanh(val))))
        tf.add_to_collection('my_losses', loss)

    # For comparing two layers
    def __eq__(self, other):
        return self.signature == other.signature
    def __ne__(self, other):
        return not self.__eq__(other)

    def varsig(self, var):
        if var not in self.wshape:
            return None
        sig = str(self.number)
        sig += '-' + self.type
        sig += '/' + var
        return sig

    def recollect(self, w): self.w = w
    def present(self): self.presenter = self
    
    #@abstractmethod
    #def preTrain(self): pass
    
    @abstractmethod
    def inverseLayer(self): pass
    
class ConvolutionLayer(Layer):
    
    def __init__(self, number, ksize, c, n, stride, 
              padding, useBias, trainable=True, halfPrecision=False):
        super(ConvolutionLayer, self).__init__(number, 'conv%d'%number)
        self.halfPrecision = halfPrecision
        precType = tf.float16 if halfPrecision else tf.float32
        self.stride = (1,stride[0],stride[1],1)
        self.ksize = ksize
        self.pad = padding
        self.useBias = useBias
        self.n = n
        self.dnshape = [int(ksize), int(ksize), int(c), int(n)]
        self.trainable = trainable
        print(self.dnshape)
        var = tf.truncated_normal(self.dnshape, stddev=0.1, dtype = precType)
        self.kernel = tf.Variable(var, name='kernel%s'%self.name, trainable=self.trainable)
        loss = WEIGHT_REG_CONST*tf.nn.l2_loss(self.kernel) 
        tf.add_to_collection('my_losses', loss)
        if(self.useBias):
            self.bias = tf.Variable(tf.constant(0.1, shape = [self.n], dtype = precType), name='bias%s'%self.name, trainable=self.trainable)
            loss = WEIGHT_REG_CONST*tf.nn.l2_loss(self.bias) 
            tf.add_to_collection('my_losses', loss)
        
    def __call__(self, inputVar):
        if(self.useBias):
            return tf.nn.conv2d(inputVar, self.kernel, padding=self.pad, strides=self.stride) + self.bias
        else:
            return tf.nn.conv2d(inputVar, self.kernel, padding=self.pad, strides=self.stride)
        
    def evalWithoutStride(self, inputVar):
        if(self.useBias):
            return tf.nn.conv2d(inputVar, self.kernel, padding=self.pad, strides=(1,1,1,1)) + self.bias
        else:
            return tf.nn.conv2d(inputVar, self.kernel, padding=self.pad, strides=(1,1,1,1))
    
    def setWeights(self, weights):
        tf.add_to_collection("assignOps", self.kernel.assign(self.kernel.assign(weights['kernel'])))
        if(self.useBias):
            tf.add_to_collection("assignOps", self.bias.assign(self.kernel.assign(weights['bias'])))
        
    def inverseLayer(self): 
        deconv = DeconvolutionLayer(self.number, self.ksize, self.dnshape[-1], self.dnshape[-2], self.stride, 
                                    self.pad, self.useBias, halfPrecision = self.halfPrecision)
        deconv.kernel = self.kernel
        return deconv 
    
    def addAutoencodeLoss(self, inputVar, act, trainPh, keepProbAE, noise_shape=None, halfPrecision = False):
        val = tf.nn.dropout(inputVar, keepProbAE, noise_shape)
        val = self.evalWithoutStride(val)
        val = ConvolutionalBatchNormalization(self.number, self.n, trainPh, halfPrecision)(val)
        val = act(val)
        val = self.inverseLayer().evalWithoutStride(val)
        val = ConvolutionalBatchNormalization(self.number, self.dnshape[-1], trainPh, halfPrecision)(val)
        val = act(val)
        loss = tf.scalar_mul(tf.constant(0.2), tf.reduce_mean(tf.losses.mean_squared_error(tf.nn.tanh(inputVar), tf.nn.tanh(val))))
        tf.add_to_collection('my_losses', loss)
        
        
class DeconvolutionLayer(Layer):
    
    def __init__(self, number, ksize, c, n, stride, 
              padding, useBias, trainable=True, halfPrecision=False):
        super(DeconvolutionLayer, self).__init__(number, 'deconv%d'%number)
        self.halfPrecision = halfPrecision
        precType = tf.float16 if halfPrecision else tf.float32
        self.stride = (1,stride[0],stride[1],1)
        self.ksize = ksize
        self.pad = padding
        self.useBias = useBias
        self.n = n
        self.dnshape = [ksize, ksize, int(n), int(c)]
        self.trainable = trainable
        self.kernel = tf.Variable(tf.truncated_normal(self.dnshape, stddev=0.1, dtype = precType), 
                                  name='kernel%s'%self.name, trainable=self.trainable)
        loss = WEIGHT_REG_CONST*tf.nn.l2_loss(self.kernel) 
        tf.add_to_collection('my_losses', loss)
        if(self.useBias):
            self.bias = Layer.variable(tf.constant(0.1, shape = [self.n], dtype = precType), 
                                       name='bias%s'%self.name, trainable=self.trainable)
            loss = WEIGHT_REG_CONST*tf.nn.l2_loss(self.bias) 
            tf.add_to_collection('my_losses', loss)
    
    def __call__(self, inputVar):
        outputShape = [tf.shape(inputVar)[0], tf.shape(inputVar)[1]*self.stride[1], 
                       tf.shape(inputVar)[2]*self.stride[2], self.dnshape[2]]
        op = None         
        if(self.useBias):
            op = tf.nn.conv2d_transpose(inputVar, self.kernel, padding=self.pad, strides=self.stride, output_shape=outputShape) + self.bias
        else:
            op = tf.nn.conv2d_transpose(inputVar, self.kernel, padding=self.pad, strides=self.stride, output_shape=outputShape)
        return tf.reshape(op, outputShape)
    
    def evalWithoutStride(self, inputVar):
        outputShape = [tf.shape(inputVar)[0], tf.shape(inputVar)[1], 
                        tf.shape(inputVar)[2], self.dnshape[2]]
        op = None         
        if(self.useBias):
            op = tf.nn.conv2d_transpose(inputVar, self.kernel, padding=self.pad, strides=(1,1,1,1), output_shape=outputShape) + self.bias
        else:
            op = tf.nn.conv2d_transpose(inputVar, self.kernel, padding=self.pad, strides=(1,1,1,1), output_shape=outputShape)
        return tf.reshape(op, outputShape)
        
    def setWeights(self, weights):
        tf.add_to_collection("assignOps", self.kernel.assign(self.kernel.assign(weights['kernel'])))
        if(self.useBias):
            tf.add_to_collection("assignOps", self.bias.assign(self.kernel.assign(weights['bias'])))
            
    def inverseLayer(self): 
        conv = ConvolutionLayer(self.number, self.ksize, self.dnshape[-1], self.dnshape[-2], 
                                self.stride, self.pad, self.useBias, halfPrecision = self.halfPrecision)
        conv.kernel = self.kernel
        return conv 
    
    def addAutoencodeLoss(self, inputVar, act, trainPh, keepProbAE, noise_shape=None, halfPrecision = False):
        val = tf.nn.dropout(inputVar, keepProbAE, noise_shape)
        val = self.evalWithoutStride(val)
        val = ConvolutionalBatchNormalization(self.number, self.n, trainPh, halfPrecision)(val)
        val = act(val)
        val = self.inverseLayer().evalWithoutStride(val)
        val = ConvolutionalBatchNormalization(self.number, self.dnshape[-1], trainPh, halfPrecision)(val)
        val = act(val)
        loss = tf.scalar_mul(tf.constant(0.2), tf.reduce_mean(tf.losses.mean_squared_error(tf.nn.tanh(inputVar), tf.nn.tanh(val))))
        tf.add_to_collection('my_losses', loss)
        
class ConvolutionalBatchNormalization(Layer):
    
    def __init__(self, number, size, trainPh=None, decay=.99, epsilon=1e-3,
                 trainable=True, scope='BN', halfPrecision = False):
        with tf.variable_scope(scope+'%d'%number) as bnScope:
            super(ConvolutionalBatchNormalization, self).__init__(number, 'bn%d'%number)
            self.halfPrecision = halfPrecision
            precType = tf.float16 if halfPrecision else tf.float32
            self.decay = decay
            self.scope = scope+'%d'%number
            self.train = trainPh
            self.epsilon = epsilon
            self.trainable = trainable
            self.opList = []
            self.movingMean = tf.Variable(tf.constant(0.0, shape=[size], dtype = precType),
                                trainable=False)
            self.movingVar = tf.Variable(tf.constant(1.0, shape=[size], dtype = precType),
                                    trainable=False)
            self.beta = tf.Variable(tf.constant(0.0, shape=[size], dtype = precType), trainable = self.trainable)
            self.gamma = tf.Variable(tf.constant(1.0, shape=[size], dtype = precType), trainable = self.trainable)
            if trainable:
                tf.add_to_collection('movingVars', self.movingMean)
                tf.add_to_collection('movingVars', self.movingVar)
                bnScope.reuse_variables()
            
        
    def __call__(self, inputVar):
        with tf.variable_scope(self.scope):
            if self.trainable:
                mean, var = tf.nn.moments(inputVar, axes = [0,1,2])
                self.realMean = mean
                self.realVar = var
                def meanVarWithUpdate():
                    update_moving_mean = moving_averages.assign_moving_average(
                        self.movingMean, mean, self.decay)
                    update_moving_variance = moving_averages.assign_moving_average(
                        self.movingVar, var, self.decay)            
                    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                        return tf.identity(mean), tf.identity(var)
                    
                    
                mean, var = tf.cond(self.train, meanVarWithUpdate, lambda: (self.movingMean, self.movingVar))
            
                self.infMean = mean
                return tf.nn.batch_normalization(
                        inputVar, mean, var, self.beta, self.gamma,
                        self.epsilon)
            else: 
                print()
                return tf.nn.batch_normalization(
                        inputVar, self.movingMean, self.movingVar, self.beta, self.gamma,
                        self.epsilon)
    
        
    def setWeights(self, weights):
        tf.add_to_collection("assignOps", self.movingMean.assign(weights['movingMean']))
        tf.add_to_collection("assignOps", self.movingVar.assign(weights['movingVariance']))
        tf.add_to_collection("assignOps", self.beta.assign(weights['bias']))
        tf.add_to_collection("assignOps", self.gamma.assign(weights['gamma']))
    
    
    def inverseLayer(self, number): 
        bn = ConvolutionalBatchNormalization(self.number, self.size, self.trainPh, 
                                             self.ema, self.epsilon, halfPrecision = self.halfPrecision)
        return bn 
    
class SplittedConvolutionLayer(Layer):
    
    def __init__(self, number, ksize, c, n, stride, 
              padding, useBias, horizontalSplits=3, verticalSplits=3, trainable=True, 
              halfPrecision=False):
        super(SplittedConvolutionLayer, self).__init__(number, 'conv%d'%number)
        self.halfPrecision = halfPrecision
        precType = tf.float16 if halfPrecision else tf.float32
        self.stride = (1,stride[0],stride[1],1)
        self.ksize = ksize
        self.pad = padding
        self.useBias = useBias
        self.n = n
        self.dnshape = [int(ksize), int(ksize), int(c), int(n)]
        self.trainable = trainable
        self.horizontalSplit = horizontalSplits
        self.verticalSplit = verticalSplits
        print(self.dnshape)
        self.convs = []
        for i in range(verticalSplits):
            horConvs = []
            for j in range(horizontalSplits):
                with tf.variable_scope('subConv%d%d'%(i,j)):
                    horConvs.append(ConvolutionLayer(number, ksize, c, n, stride, padding, 
                                                     useBias, trainable, halfPrecision))
            self.convs.append(horConvs)


    def splitVar(self, inputVar):
        varList = []
        
        height = int(inputVar.get_shape()[1])
        heightSplits = []
        heightDone = 0
        for i in range(self.verticalSplit-1):
            heightSplits.append(height//self.verticalSplit)
            heightDone+=height//self.verticalSplit
        heightSplits.append(height-heightDone)  
        
        print(heightSplits)
        
        widthSplits = []
        width = int(inputVar.get_shape()[2])
        lenDone = 0
        for i in range(self.horizontalSplit-1):
            widthSplits.append(width//self.horizontalSplit)
            lenDone += width//self.horizontalSplit
        widthSplits.append(width-lenDone)  
        print(widthSplits)
        vertList = tf.split(inputVar, num_or_size_splits=heightSplits, axis=1)
        for t in vertList:
            varList.append(tf.split(t, num_or_size_splits=widthSplits, axis=2))
        return varList
    
    def __call__(self, inputVar):
        varList = self.splitVar(inputVar)
        
        resList = []
        for vertVarList, vertConvList in zip(varList, self.convs):
            horRes = []
            for var, conv in zip(vertVarList, vertConvList):
                horRes.append(conv(var))
            resList.append(tf.concat(horRes, axis=2))
        
        return tf.concat(resList, axis=1)
    
    def setWeights(self, weights):
        for vertWeights, vertConvs in zip(weights, self.convs):
            for weight, conv in zip(vertWeights, vertConvs):
                conv.setWeights(weight)
                
    def inverseLayer(self): 
        raise NotImplementedError
    
    def addAutoencodeLoss(self, inputVar, act, trainPh, keepProbAE, noise_shape=None):
        varList = self.splitVar(inputVar)
        for vertVarList, vertConvList in zip(varList, self.convs):
            for var, conv in zip(vertVarList, vertConvList):
                conv.addAutoencodeLoss(var, act, trainPh, keepProbAE, noise_shape)
                

class SplittedDeconvolutionLayer(Layer):
    def __init__(self, number, ksize, c, n, stride, 
              padding, useBias, horizontalSplits=3, verticalSplits=3, 
              trainable=True , halfPrecision=False):
        super(SplittedDeconvolutionLayer, self).__init__(number, 'deconv%d'%number)
        self.halfPrecision = halfPrecision
        precType = tf.float16 if halfPrecision else tf.float32
        self.stride = (1,stride[0],stride[1],1)
        self.ksize = ksize
        self.pad = padding
        self.useBias = useBias
        self.n = n
        self.dnshape = [ksize, ksize, int(n), int(c)]
        self.trainable = trainable
        self.kernel = tf.Variable(tf.truncated_normal(self.dnshape, stddev=0.1), name='kernel%s'%self.name, trainable=self.trainable)
        
        self.horizontalSplit = horizontalSplits
        self.verticalSplit = verticalSplits
        self.convs = []
        for i in range(verticalSplits):
            horConvs = []
            for j in range(horizontalSplits):
                with tf.variable_scope('subDeonv%d%d'%(i,j)):
                    horConvs.append(DeconvolutionLayer(number, ksize, c, n, stride, padding, 
                                                     useBias, trainable, halfPrecision))
            self.convs.append(horConvs)

    def splitVar(self, inputVar):
        varList = []
        
        height = tf.shape(inputVar)[1]
        heightSplits = []
        heightDone = 0
        for i in range(self.verticalSplit-1):
            heightSplits.append(height//self.verticalSplit)
            heightDone+=height//self.verticalSplit
        heightSplits.append(height-heightDone)  
        
        widthSplits = []
        width = tf.shape(inputVar)[2]
        lenDone = 0
        for i in range(self.horizontalSplit-1):
            widthSplits.append(width//self.horizontalSplit)
            lenDone += width//self.horizontalSplit
        widthSplits.append(width-lenDone)
        vertList = tf.split(inputVar, num_or_size_splits=heightSplits, axis=1)
        for t in vertList:
            varList.append(tf.split(t, num_or_size_splits=widthSplits, axis=2))
        return varList
    
    def __call__(self, inputVar):
        varList = self.splitVar(inputVar)
        
        resList = []
        for vertVarList, vertConvList in zip(varList, self.convs):
            horRes = []
            for var, conv in zip(vertVarList, vertConvList):
                horRes.append(conv(var))
            resList.append(tf.concat(horRes, axis=2))
        
        return tf.concat(resList, axis=1)
    
    def setWeights(self, weights):
        for vertWeights, vertConvs in zip(weights, self.convs):
            for weight, conv in zip(vertWeights, vertConvs):
                conv.setWeights(weight)
                
    def inverseLayer(self): 
        raise NotImplementedError
    
    def addAutoencodeLoss(self, inputVar, act, trainPh, keepProbAE, noise_shape=None):
        varList = self.splitVar(inputVar)
        
        for vertVarList, vertConvList in zip(varList, self.convs):
            for var, conv in zip(vertVarList, vertConvList):
                conv.addAutoencodeLoss(var, act, trainPh, keepProbAE, noise_shape)