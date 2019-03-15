'''
Created on Jun 8, 2017

@author: jendrik
'''
from Network.Network import SegNet
import numpy as np
from matplotlib import image as mpimg
import pickle
from matplotlib import pyplot as plt
from Data.Datasets import Mapillary, CityScapes
import time
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import random
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
        print(data.numberOfClasses)
        arr += data.getTrainDistribution()
    #names = []
    #for i in range(datasets[0].numberOfClasses):
    #    names.append(classDict.get(i))     
    plt.plot(range(datasets[0].numberOfClasses), arr)
    #plt.xticks(range(datasets[0].numberOfClasses), names, size='small', rotation='vertical')
    plt.show()
    return arr
    

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
    mapillary = Mapillary('/media/jendrik/DataSwap/Datasets/mapillary-vistas/', numberOfClasses=12)
    cityscapes = CityScapes('/media/jendrik/DataSwap/Datasets/CityScapes', numberOfClasses=12)
    #camVid = CamVid('/media/jendrik/DataSwap1/Datasets/CamVid/', numberOfClasses=15)
    numberOfClasses = mapillary.numberOfClasses
    datasets = [mapillary, cityscapes]
    #weights = np.array([1.,1.,1.])
    if 0:
        weights = inverstigateDatasets(datasets, classDict)
        with open('normWeights.pickle', 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('normWeights.pickle', 'rb') as handle:
            weights = pickle.load(handle)
        plt.plot(range(mapillary.numberOfClasses), weights)
        plt.xticks(range(mapillary.numberOfClasses), 
                   [mapillary.id2label[i].name for i in range(mapillary.numberOfClasses)],
                   rotation=90)
        plt.subplots_adjust(bottom=0.3)
        
        #plt.show()
    print(weights)
    #weights = np.array([1 if val < np.e else 1/np.log(val) for val in weights])
    weights = np.array([1/np.log(val/1e7) if val > 1e7*np.e else 1 for val in weights])
    plt.plot(range(mapillary.numberOfClasses), weights)
    plt.xticks(range(mapillary.numberOfClasses),
               [mapillary.id2label[i].name for i in range(mapillary.numberOfClasses)],
               rotation=90)    
    plt.subplots_adjust(bottom=0.3)
    #plt.show()
    print(weights)
    #mapillary.setWeights(weights)
    
    patience = 15
    numEpochs = 220
    batchPerEpoch = 1024
    batchesPerVal = 1024
    count = 0
    startEpoch = 0
    
    
    """shadowVars = {}
    for var in tf.get_collection('movingVars'):
        shadowVars.update({ema.average_name(var) : var})
    print(shadowVars)"""

    
    crossEntropyComp = 10000.
    dataProbs = []
    saveGraph = True
    
    valQueuers = [dataset.valQueuer for dataset in datasets]
    trainQueuers = [dataset.trainQueuer for dataset in datasets]
    time.sleep(1.)
    print("Before getting data")
    data = None            
    while valQueuers[0].is_running():
        if not valQueuers[0].queue.empty():
            data = valQueuers[0].queue.get()[1]
            break
        else:
            time.sleep(.05)
    print(data)
    image = Variable(torch.from_numpy(data[0]['inputImg'])).float().permute(0,3,1,2)
    print("Got data")
    imageTemp = (255.*image[0].permute(1,2,0).data.numpy()).astype('uint8')
    print("Got temp image")
    
    #plt.imshow(image[0])
    #plt.show()
    inputShape = image.shape
    print("Start building Segnet!")
    writer = SummaryWriter()
    segNet = SegNet(numberOfClasses, Variable(torch.from_numpy(weights)).float().cuda(), dropProb = .2)
    writer.add_graph(segNet, segNet(image))
    print(segNet.conv1)
    segNet.cuda()
    print("Everything Setup!")
    switch = False
    
    for dataset in datasets:
        if(len(dataProbs) == 0): dataProbs.append(len(dataset.trainData))
        else: dataProbs.append(len(dataset.trainData) + sum(dataProbs))
    
    for i in np.arange(startEpoch, numEpochs, 1):
        quer = random.choice(valQueuers)
        print(type(quer))
        testTrain = quer.queue.get()[1]
        segNet.scheduler.step()
        if i == 0:
            segNet.setWeights('../darknet19_448.weights')
            print('Yolo untrainable')
            segNet.setYoloTrainable(False)
            img = Variable(torch.from_numpy(testTrain[0]['inputImg'])).float().permute(0,3,1,2)
            target = Variable(torch.from_numpy(testTrain[1]['outputImg'])).float()
            segNet.trainNet(img.cuda(), target.cuda())
            res = mapillary.convertTargetToImage(segNet.eval(image.cuda()).data.cpu().numpy()[0])
            print(res.dtype)
            print(res.shape)
            print(imageTemp.dtype)
            print(imageTemp.shape)
            mpimg.imsave('../images/initialImage.png', res)#cv2.addWeighted(res, 0.5, imageTemp, 0.5,0))
            
        batchCE = 0
        trainAcc = 0
        for j in range(batchPerEpoch):
            number = np.random.randint(0, max(dataProbs))
            for k in range(len(dataProbs)):
                if dataProbs[k] > number:
                    trainQueuer = trainQueuers[k]
                    break
            #print(k)
            data = None
            while trainQueuer.is_running():
                if not trainQueuer.queue.empty():
                    data = trainQueuer.queue.get()[1]
                    break
                else:
                    time.sleep(.05)
            tmpCE, tmpAcc = segNet.trainNet(
                Variable(torch.from_numpy(data[0]['inputImg'])).float().permute(0,3,1,2).cuda(), 
                Variable(torch.from_numpy(data[1]['outputImg'])).float().cuda())
            print("Step: %d of %d, Train Loss: %g Train Acc %g %s" % (j, batchPerEpoch, tmpCE, tmpAcc, data[2]))
            #print(data[0]['inputImg'].shape, data[1]['outputImg'].shape, np.unique(data[0]['inputImg']))
            if np.isnan(tmpCE) or tmpAcc == 0: quit()
            batchCE += tmpCE
            trainAcc += tmpAcc
        batchCE /= batchPerEpoch
        trainAcc /= batchPerEpoch
        writer.add_scalar('Train CrossEntropy', batchCE, i)
        writer.add_scalar('Train Accuracy', trainAcc, i)
        # Add it to the Tensorboard summary writer
        # Make sure to specify a step parameter to get nice graphs over time
        #input()
        testCE=0
        testAcc=0
        for j in range(batchesPerVal):
            number = np.random.randint(0, max(dataProbs))
            for k in range(len(dataProbs)):
                if dataProbs[k] > number:
                    valQueuer = valQueuers[k]
                    break
            data = None
            while valQueuer.is_running():
                if not valQueuer.queue.empty():
                    data = valQueuer.queue.get()[1]
                    break
                else:
                    time.sleep(.01)
            tmpCE, tmpAcc = segNet.val(
                Variable(torch.from_numpy(data[0]['inputImg']), requires_grad=False).float().permute(0,3,1,2).cuda(), 
                Variable(torch.from_numpy(data[1]['outputImg']), requires_grad=False).float().cuda())
            testCE += tmpCE
            testAcc += tmpAcc
        testCE /= batchesPerVal
        testAcc /= batchesPerVal
        writer.add_scalar('Val CrossEntropy', testCE, i)
        writer.add_scalar('Val Accuracy', testAcc, i)
        # Add it to the Tensorboard summary writer
        # Make sure to specify a step parameter to get nice graphs over time
        res = mapillary.convertTargetToImage(segNet.eval(image.cuda()).data.cpu().numpy()[0])
        mpimg.imsave('../images/Epoch%04d.png'%i, cv2.addWeighted(res, 0.5, imageTemp, 0.5, 0))
        count += 1
            
        if testCE < (crossEntropyComp - .001):
            torch.save(segNet.state_dict(), '../ckpt/bestModel.ckpt')
            count = 0
            crossEntropyComp = testCE
        if count >= patience:
            if not switch:
                switch = True 
                count = 0
                patience = 8
                print('Yolo trainable')
                segNet.setYoloTrainable(True)
            else:
                break
    writer.close()
    print("Finished")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    