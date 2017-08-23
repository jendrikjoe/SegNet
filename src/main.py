'''
Created on Jun 8, 2017

@author: jendrik
'''
from Network.Network import SegNet
import numpy as np
from matplotlib import image as mpimg
import pickle
from keras.utils import to_categorical
import glob
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from Data.Datasets import CamVid, Kitti
import tensorflow as tf
import time
from datetime import datetime
import cv2
SMALLNET = False

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

def inverstigateDatasets(datasets, classDict):
    arr = np.zeros(datasets[0].numberOfClasses)
    for data in datasets:
        arr += data.getTrainDistribution()
    names = []
    for i in range(datasets[0].numberOfClasses):
        names.append(classDict.get(i))     
    plt.plot(range(datasets[0].numberOfClasses), arr)
    plt.xticks(range(datasets[0].numberOfClasses), names, size='small', rotation='vertical')
    plt.show()
    weights = np.array([5e6/val if val > 5e6 else 1 for val in arr])
    return weights

def loadCkpt(smallNet, sess):
    if smallNet: checkpointFile = os.path.join('./ckpt', 'smallSegNet.ckpt')
    else: checkpointFile = os.path.join('./ckpt', 'segNet.ckpt')
    saver.restore(sess, checkpointFile)
    
def trainOnDataSet(sess, dataset, epoch, summaryWriter, batchPerEpoch):
    trainQueuer = dataset.trainQueuer
    valQueuer = dataset.valQueuer
    globalStep = tf.Variable(epoch*batchPerEpoch, name='globalStep')
    image = dataset.generator(1, False).__next__()[0]['inputImg']
    inputShape = image.shape
    learningRate = tf.train.exponential_decay(
            1e-4,                # Base learning rate.
            globalStep,  # Current index into the dataset.
            1024,          # Decay step.
            (1.-0.0005),                # Decay rate.
            staircase=True)
    segNet = SegNet(inputShape, numberOfClasses, learningRate, epoch*batchPerEpoch)
    init = tf.global_variables_initializer()
    localInit = tf.local_variables_initializer()
    tf.logging.set_verbosity(tf.logging.FATAL)
    sess.run([init, localInit])
    saver = tf.train.Saver()
    if epoch == 0:
        summaryWriter.add_graph(
             sess.graph
        )

        testTrain = dataset.generator(1, True).__next__()
        print(np.min(testTrain[0]['inputImg']))
        print(np.max(testTrain[0]['inputImg']))
        print(np.min(image))
        print(np.max(image))
        
        
        segNet.setWeights('../darknet19_448.weights', sess)
        segNet.train(np.array(testTrain[0]['inputImg']), np.array(testTrain[1]['outputImg']), testTrain[2], sess)
        res = convertTargetToImage(np.reshape(segNet.eval(image,sess),
                                                            (inputShape[1], inputShape[2])), colorDictInv)
        
        imageTemp = (255.*image[0]).astype('uint8')
        print(image.shape)
        print(image.dtype)
        print(res.shape)
        print(res.dtype)
        mpimg.imsave('../images/initialImage.png', cv2.addWeighted(res, 0.5, imageTemp, 0.,0))
        tf.train.write_graph(sess.graph, '../ckpt', 'train.pb', as_text=False)
    else:
        checkpointFile = os.path.join('./ckpt', 'trainCkpt.ckpt')
        saver.restore(sess, checkpointFile)
    
    batchLoss = 0 
    batchAcc = 0
    batchCE = 0
    batchIMU = 0
    for j in range(batchPerEpoch):
        data = None
        while trainQueuer.is_running():
            if not trainQueuer.queue.empty():
                data = trainQueuer.queue.get()
                break
            else:
                time.sleep(.05)
        tmpLoss, tmpAcc, tmpCE, tmpImu= segNet.train(np.array(data[0]['inputImg']), np.array(data[1]['outputImg']), data[2], sess)
        print("Step: %d of %d, Train Loss: %g" % (j, batchPerEpoch, tmpLoss))
        batchLoss += tmpLoss
        batchAcc += tmpAcc
        batchCE += tmpCE
        batchIMU += tmpImu
    summary = tf.Summary()
    batchAcc /= batchPerEpoch
    batchLoss /= batchPerEpoch
    batchCE /= batchPerEpoch
    batchIMU /= batchPerEpoch
    summary.value.add(tag="TrainAccuracy", simple_value=batchAcc)
    summary.value.add(tag="TrainLoss", simple_value=batchLoss)
    summary.value.add(tag="TrainCrossEntropy", simple_value=batchCE)
    summary.value.add(tag="TrainIMU", simple_value=batchIMU)
    # Add it to the Tensorboard summary writer
    # Make sure to specify a step parameter to get nice graphs over time
    testAcc=0
    testLoss=0
    testCE=0
    testIMU=0
    for j in range(batchesPerVal):
        data = None
        while valQueuer.is_running():
            if not valQueuer.queue.empty():
                data = valQueuer.queue.get()
                break
            else:
                time.sleep(.01)
        tmpLoss, tmpAcc, tmpCE, tmpImu = segNet.val(np.array(data[0]['inputImg']), np.array(data[1]['outputImg']), data[2], sess)
        testLoss += tmpLoss
        testAcc += tmpAcc
        testCE += tmpCE
        testIMU += tmpImu
    testAcc/=batchesPerVal
    testLoss/=batchesPerVal
    testCE/=batchesPerVal
    testIMU /= batchPerEpoch
    summary.value.add(tag="ValidationAccuracy", simple_value=testAcc)
    summary.value.add(tag="ValidationLoss", simple_value=testLoss)
    summary.value.add(tag="ValidationCrossEntropy", simple_value=testCE)
    summary.value.add(tag="ValidationIMU", simple_value=testIMU)
    # Add it to the Tensorboard summary writer
    # Make sure to specify a step parameter to get nice graphs over time
    summaryWriter.add_summary(summary, epoch)
    res = convertTargetToImage(np.reshape(segNet.eval(image,sess),
                                                            (inputShape[1], inputShape[2])), colorDictInv)
    imageTemp = (255.*image[0]).astype('uint8')
    mpimg.imsave('../images/Epoch%d.png'%epoch, cv2.addWeighted(res, 0.5, imageTemp, 0.5, 0))
    checkpointFile = os.path.join('./ckpt', 'trainCkpt.ckpt')
    saver.save(sess, checkpointFile)
    return testCE, saver

if __name__ == '__main__':
    colorDict = {
        (0, 0, 0): 0,
        (0, 0, 64): 1,
        (0, 0, 192): 2,
        (0, 64, 64): 3,
        (0, 128, 64): 4,
        (0, 128, 192): 5,
        (64, 0, 64): 6,
        (64, 0, 128): 7,
        (64, 0, 192): 8,
        (64, 64, 0): 9,
        (64, 64, 128): 10,
        (64, 128, 64): 11,
        (64, 128, 192): 12,
        (64, 192, 0): 13,
        (64, 192, 128): 14,
        (128, 0, 0): 15,
        (128, 0, 192): 16,
        (128, 64, 64): 17,
        (128, 64, 128): 18,
        (128, 128, 0): 19,
        (128, 128, 64): 20,
        (128, 128, 128): 21,
        (128, 128, 192): 22, 
        (192, 0, 64): 23,
        (192, 0, 128): 24,
        (192, 0, 192): 25,
        (192, 64, 128): 26, 
        (192, 128, 64): 27,
        (192, 128, 128): 28,
        (192, 128, 192): 29,
        (192, 192, 0): 30,
        (192, 192, 128): 31
    }
    
    classDict = {
        0: 'Void',
        1: 'TrafficCone',
        2: 'Sidewalk',
        3: 'TrafficLight',
        4: 'Bridge',
        5: 'Bicyclist',
        6: 'Tunnel',
        7: 'Car',
        8: 'CartLuggagePram',
        9: 'Pedestrian',
        10: 'Fence',
        11: 'Animal',
        12: 'SUVPickupTruck',
        13: 'Wall',
        14: 'ParkingBlock',
        15: 'Building',
        16: 'LaneMkgsDriv',
        17: 'OtherMoving',
        18: 'Road',
        19: 'Tree',
        20: 'Misc_Text',
        21: 'Sky',
        22: 'RoadShoulder',
        23: 'LaneMkgsNonDriv',
        24: 'Archway',
        25: 'MotorcycleScooter',
        26: 'Train',
        27: 'Child',
        28: 'SignSymbol',
        29: 'Truck_Bus',
        30: 'VegetationMisc',
        31: 'Column_Pole',
    }
    
    colorDictInv = {v:k for k,v in colorDict.items()}
    numberOfClasses = len(colorDict)
    camVid = CamVid('../data/CamVid', numberOfClasses=numberOfClasses, samplesPerBatch=1)
    kitti = Kitti('../data/kitti_Philippe Xu', numberOfClasses=numberOfClasses, samplesPerBatch=1)
    datasets = [camVid]
    
    if 0:
        weights = inverstigateDatasets(datasets, classDict)
        with open('normWeights.pickle', 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('normWeights.pickle', 'rb') as handle:
            weights = pickle.load(handle)
    print(weights)
    for dat in datasets:
        dat.setWeights(weights)
    
    patience = 50
    samplesPerBatch = 1
    numEpochs = 500
    batchPerEpoch = 1024
    batchesPerVal = 128
    count = 0
    startEpoch = 55
    
    
    """shadowVars = {}
    for var in tf.get_collection('movingVars'):
        shadowVars.update({ema.average_name(var) : var})
    print(shadowVars)"""

    
    summaryWriter = tf.summary.FileWriter('./log/'+datetime.now().isoformat())
    crossEntropyComp = 10000.
    dataProbs = []
    for dataset in datasets:
        if(len(dataProbs) == 0): dataProbs.append(np.log(len(dataset.trainData)))
        else: dataProbs.append(np.log(len(dataset.trainData)) + sum(dataProbs))
    saveGraph = True
    config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True)
    jitLevel = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jitLevel
    
    for i in np.arange(startEpoch, numEpochs,1):
        with tf.Graph().as_default(), tf.Session(config=config) as sess:
            number = np.random.randint(0, max(dataProbs))
            if i < len(datasets):
                dataset = datasets[i]
            else:
                for j in range(len(dataProbs)):
                    if dataProbs[j] > number:
                        dataset = datasets[j]
                        break
                print(j)
            testCE, saver = trainOnDataSet(sess, dataset, i, summaryWriter, batchPerEpoch)
            saveGraph = False
            count += 1
            
            if testCE < (crossEntropyComp - .001):
                checkpointFile = os.path.join('./ckpt', 'segNet.ckpt')
                
                saver.save(sess, checkpointFile)
                count = 0
                crossEntropyComp = testCE
            if count >= patience:
                break
    print("Finished")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    