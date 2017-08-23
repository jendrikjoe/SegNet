'''
Created on Aug 22, 2017

@author: jendrik
'''
import numpy as np
from moviepy.editor import VideoFileClip
import functools
from Network.Network import SegNetInference
import tensorflow as tf
import os
from Data.Datasets import CityScapes, KittiStreet
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


def convertStreetToImage(target):
    image = np.zeros((target.shape[0], target.shape[1]))
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            image[i,j] = 255 if target[i,j] == 7 else 0
    return image.astype('uint8')

def convertTargetToImage(target, colourDictInv):
    image = np.zeros((target.shape[0], target.shape[1], 3))
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            #if (label == None):
            #    print(image[i,j])
            if(i == 719 and j == target.shape[1]//2): print(target[i,j],
                     np.argmax(target[i,j]), colourDictInv.get(np.argmax(target[i,j])))
            image[i,j, :] = colourDictInv.get(target[i,j])[:]
    return image.astype('uint8')

def pipeline(image, model, colorDictInv, sess):
    inputShape = image.shape
    image = np.reshape(image, (1,inputShape[0],inputShape[1],inputShape[2]))
    print(np.max(image))
    return convertTargetToImage(np.reshape(model.eval(image, sess), 
                        (inputShape[0], inputShape[1])), colorDictInv)

if __name__ == '__main__':
    dataset = CityScapes('../data/CityScapes', numberOfClasses=0, samplesPerBatch=1)
    colorDict = {}
    for i in range(len(dataset.id2label)):
        colorDict.update({i: dataset.id2label[i].color})
    
    globalStep = tf.Variable(0, name = 'globalStep')
    learningRate = tf.train.exponential_decay(
            1e-4,                # Base learning rate.
            globalStep,  # Current index into the dataset.
            1024,          # Decay step.
            (1.-0.0005),                # Decay rate.
            staircase=True)
    dataset = KittiStreet('../data/KittiStreet', numberOfClasses=0, samplesPerBatch=1)
    segNet = SegNetInference((1,512,1024,3), len(colorDict), learningRate, globalStep, withAverage=False)
    previous_variables = [
      var_name for var_name, _
      in tf.contrib.framework.list_variables('./ckpt/segNet.ckpt')]
    print(previous_variables)
    restore_map = {variable.op.name:variable for variable in tf.global_variables()
                   if variable.op.name in previous_variables}
    print(restore_map)
    tf.contrib.framework.init_from_checkpoint(
        './ckpt/segNet.ckpt', restore_map)
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True))
    tf.logging.set_verbosity(tf.logging.FATAL)
    sess.run(init)
    for img in dataset.testData['input']:
        image = mpimg.imread(img)
        inputShape = image.shape
        image = cv2.resize(image, (1024, 512))
        out = pipeline(image, segNet, colorDict, sess)
        print(out.shape)
        print(image.shape)
        #out = np.dstack((np.zeros_like(out),out, np.zeros_like(out)))
        out = cv2.addWeighted(out, 0.5, (255*image).astype('uint8'), 0.5,0)
        out = cv2.resize(out, (1242, 375))
        mpimg.imsave('../data/KittiStreet/testing/gt_image_2/'+img.split('/')[-1], out)
        
        
        