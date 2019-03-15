'''
Created on May 31, 2017

@author: jendrik
'''
from abc import ABC, abstractmethod
import glob
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from sklearn.model_selection._split import train_test_split
import pickle
import os
from SegNet.Preprocess.Preprocess import mirrorImage, augmentImageAndLabel
import threading
from multiprocessing import Queue
from collections import namedtuple  
import multiprocessing
from multiprocessing import Pool
from matplotlib import pyplot as plt
import functools
import json
import torch
from torch.autograd import Variable
import time
import cv2
import traceback

class Dataset(ABC):
    '''
    classdocs
    '''


    def __init__(self, path, **kwargs):
        '''
        Constructor
        '''
        self.path = path
        self.train_data = None
        self.val_data = None
        
    @abstractmethod
    def generator(self, data, queue, train, **kwargs):
        pass
    
    @abstractmethod
    def get_train_distribution(self):
        pass

    @abstractmethod
    def launch_generators(self):
        pass
    
    @abstractmethod
    def setWeights(self):
        pass


class CityScapes(Dataset):
    
    def __init__(self, path, **kwargs):
        super(CityScapes, self).__init__(path=path)
        imageList = glob.glob(self.path + '/leftImg8bit/train/*/*.png')
        self.train_data = pd.DataFrame(imageList, columns = ['input'])
        self.train_data['output'] = self.train_data['input'].apply(lambda x: self.path + '/gtFine_trainvaltest/gtFine/train/'+
                                             self.getCityName(x)+'/'+self.getImageName(x)+'_gtFine_labelIds.png')
        imageList = glob.glob(self.path + '/leftImg8bit/val/*/*.png')
        self.val_data = pd.DataFrame(imageList, columns = ['input'])
        self.val_data['output'] = self.val_data['input'].apply(lambda x: self.path + '/gtFine_trainvaltest/gtFine/val/'+
                                             self.getCityName(x)+'/'+self.getImageName(x)+'_gtFine_labelIds.png')
        self.number_of_classes = kwargs['number_of_classes']
        self.weights = None
        Label = namedtuple( 'Label' , [

            'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                            # We use them to uniquely name a class
        
            'id'          , # An integer ID that is associated with this label.
                            # The IDs are used to represent the label in ground truth images
                            # An ID of -1 means that this label does not have an ID and thus
                            # is ignored when creating ground truth images (e.g. license plate).
                            # Do not modify these IDs, since exactly these IDs are expected by the
                            # evaluation server.
        
            'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                            # ground truth images with train IDs, using the tools provided in the
                            # 'preparation' folder. However, make sure to validate or submit results
                            # to our evaluation server using the regular IDs above!
                            # For trainIds, multiple labels might have the same ID. Then, these labels
                            # are mapped to the same class in the ground truth images. For the inverse
                            # mapping, we use the label that is defined first in the list below.
                            # For example, mapping all void-type classes to the same ID in training,
                            # might make sense for some approaches.
                            # Max value is 255!
        
            'category'    , # The name of the category that this label belongs to
        
            'categoryId'  , # The ID of this category. Used to create ground truth images
                            # on category level.
        
            'hasInstances', # Whether this label distinguishes between single instances or not
        
            'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                            # during evaluations or not
        
            'color'       , # The color of this label
        ] )
        
        labels = [
            #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
            Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
            Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
            Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
            Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
            Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
            Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
            Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
            Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
            Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
            Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
            Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
            Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
            Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
            Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
            Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
            Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
            Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
            Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
            Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
            Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
            Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
            Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
            Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
            Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
            Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
            Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
            Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
            Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
            Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
            #Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
        ]
        self.name2label = {label.name: label for label in labels}
        # id to label object
        self.id2label = {label.id: label for label in labels}
        self.dict = {
            0: 0,  # Void
            1: 0,  # EgoCar -> Void
            2: 0,  # Reectification Border -> Void
            3: 0,  # Out of Roi -> Void
            4: 0,  # Static -> Void
            5: 17,  # Dynamic -> Other Moving
            6: 22,  # Ground -> Road Shoulder
            7: 18,  # Road
            8: 2,  # Sidewalk
            9: 14,  # Parking
            10: 15,  # rail track -> Railway
            11: 13,  # Building -> Building/Wall
            12: 13,  # Wall -> Building/Wall
            13: 10,  # Fence
            14: 10,  # Guard Rail -> Fence
            15: 4,  # Bridge
            16: 6,  # Tunnel
            17: 31,  # Pole -> Column_Pole
            18: 31,  # PoleGroup -> Column_Pole
            19: 3,  # TrafficLight
            20: 28,  # Traffic Sign -> SignSymbol
            21: 19,  # Vegetation -> Tree
            22: 30,  # Terrain -> Misc Vegetation
            23: 21,  # Sky
            24: 9,  # Person -> Pedestrian
            25: 5,  # Rider -> Bicycle
            26: 7,  # Car
            27: 29,  # Truck -> Truck_Bus
            28: 29,  # Bus -> Truck_Bus
            29: 12,  # Caravan -> SUVPickupTruck
            30: 29,  # Trailer -> Truck_Bus
            31: 26,  # Train -> Train
            32: 25,  # Motorcycle -> Motorcycle
            33: 5,  # Bicycle -> Cyclist
            -1: 7,  # Licence Plate -> Car
        }
        self.lyftDict = {
            0: 0,  # Void -> Void
            1: 0,  # Ego Car -> Animal
            2: 0,  # Reectification Border -> Void
            3: 0,  # Out of Roi -> Void
            4: 2,  # Static -> Static
            5: 11,  # Dynamic -> car
            6: 3,  # Ground -> vegetation/dirt
            7: 1,  # Road
            8: 6,  # Sidewalk
            9: 1,  # Parking -> Road
            10: 1,  # rail track -> Road
            11: 2,  # Building -> static
            12: 7,  # Wall
            13: 7,  # Fence
            14: 7,  # Guard Rail -> Fence
            15: 2,  # Bridge
            16: 2,  # Tunnel
            17: 8,  # Pole -> Column_Pole
            18: 8,  # PoleGroup -> Column_Pole
            19: 9,  # TrafficLight
            20: 10,  # Traffic Sign -> SignSymbol
            21: 3,  # Vegetation -> Tree
            22: 3,  # Terrain -> Misc Vegetation
            23: 4,  # Sky
            24: 5,  # Person -> Pedestrian
            25: 5,  # Rider -> Person
            26: 11,  # Car
            27: 11,  # Truck -> Truck_Bus
            28: 11,  # Bus -> Truck_Bus
            29: 11,  # Caravan -> SUVPickupTruck
            30: 11,  # Trailer -> Truck_Bus
            31: 11,  # Train -> Train
            32: 11,  # Motorcycle -> Motorcycle
            33: 11,  # Bicycle -> Cyclist
            -1: 11,  # Licence Plate -> Car
        }
        labels_lyft = [
            #       name                   id    trainId   category    catId     hasInstances   ignoreInEval   color
            Label('void'            ,  0,    4, 'void'         , 4, False, False, (  0,  0,  0)),
            Label('road'            ,  1,    4, 'flat'         , 4, False, False, (128, 64,128)),
            Label('static'          ,  2,  255, 'construction' , 2, False, False, ( 70, 70, 70)),
            Label('vegetation/dirt' ,  3,  255, 'flat'         , 2, False, False, (107,142, 35)),
            Label('sky'             ,  4,  255, 'nature'       , 2, False, False, ( 70,130,180)),
            Label('person'          ,  5,  255, 'human'        , 2, False, False, (220, 20, 60)),
            Label('sidewalk'        ,  6,  255, 'flat'         , 2, False, False, (244, 35,232)),
            Label('Wall/Fence'      ,  7,    1, 'construction' , 1, False, False, (190,153,153)),
            Label('pole'            ,  8,    1, 'object'       , 1, False, False, (153,153,153)),
            Label('traffic-light'   ,  9,  255, 'object'       , 1, False, False, (250,170, 30)),
            Label('traffic-sign'    , 10,  255, 'object'       , 1, False, False, (220,220,220)),
            Label('car'             , 11,    2, 'object'       , 1, False, False, (  0,  0,142)),
            Label('Marking'         , 12,  255, 'flat'         , 1, False, False, (200,128,128)),
        ]
        self.thunderhill_dict = {
            0: 0,  # Void -> Void
            1: 0,  # Ego Car -> Void
            2: 0,  # Reectification Border -> Void
            3: 0,  # Out of Roi -> Void
            4: 2,  # Static
            5: 11,  # Dynamic -> car
            6: 3,  # Ground -> vegetation/dirt
            7: 1,  # Road
            8: 6,  # Sidewalk
            9: 1,  # Parking -> Road
            10: 1,  # rail track -> Road
            11: 2,  # Building -> static
            12: 7,  # Wall
            13: 7,  # Fence
            14: 7,  # Guard Rail -> Fence
            15: 2,  # Bridge -> static
            16: 2,  # Tunnel -> static
            17: 8,  # Pole
            18: 8,  # PoleGroup -> Pole
            19: 9,  # TrafficLight
            20: 10,  # Traffic Sign
            21: 3,  # Vegetation -> vegetation/dirt
            22: 3,  # Terrain -> vegetation/dirt
            23: 4,  # Sky
            24: 5,  # Person -> Pedestrian
            25: 13,  # Rider -> Cyclish
            26: 11,  # Car
            27: 12,  # Truck -> big vehicle
            28: 12,  # Bus -> big vehicle
            29: 12,  # Caravan -> big vehicle
            30: 12,  # Trailer -> big vehicle
            31: 12,  # Train -> big vehicle
            32: 13,  # Motorcycle -> Cyclish
            33: 13,  # Bicycle -> Cyclish
            -1: 11,  # Licence Plate -> Car
        }
        labels_thunderhill = [
            #       name                   id    trainId   category    catId     hasInstances   ignoreInEval   color
            Label('void', 0, 4, 'void', 4, False, False, (0, 0, 0)),
            Label('road', 1, 4, 'flat', 4, False, False, (128, 64, 128)),
            Label('static', 2, 255, 'construction', 2, False, False, (70, 70, 70)),
            Label('vegetation/dirt', 3, 255, 'flat', 2, False, False, (107, 142, 35)),
            Label('sky', 4, 255, 'nature', 2, False, False, (70, 130, 180)),
            Label('person', 5, 255, 'human', 2, False, False, (220, 20, 60)),
            Label('sidewalk', 6, 255, 'flat', 2, False, False, (244, 35, 232)),
            Label('Wall/Fence', 7, 1, 'construction', 1, False, False, (190, 153, 153)),
            Label('pole', 8, 1, 'object', 1, False, False, (153, 153, 153)),
            Label('traffic-light', 9, 255, 'object', 1, False, False, (250, 170, 30)),
            Label('traffic-sign', 10, 255, 'object', 1, False, False, (220, 220, 220)),
            Label('car', 11, 2, 'object', 1, False, False, (0, 0, 142)),
            Label('big vehicle', 12, 3, 'object', 1, False, False, (0, 0, 70)),
            Label('Cyclish', 13, 4, 'object', 1, False, False, (0, 0, 230)),
            Label('Marking', 14, 255, 'flat', 1, False, False, (200, 128, 128)),
        ]
        if self.number_of_classes == 12:
            self.name2label = {label.name: label for label in labels_lyft}
            # id to label object
            self.id2label = {label.id: label for label in labels_lyft}
        if self.number_of_classes == 0:
            self.trainQueuer = GeneratorEnqueuer(self.generatorAllClasses(True), False)
            self.trainQueuer.start(workers=1, max_queue_size=10)
            self.valQueuer = GeneratorEnqueuer(self.generatorAllClasses(False), False)
            self.valQueuer.start(workers=1, max_queue_size=10)
        if self.number_of_classes == 12:
            self.trainQueuer = GeneratorEnqueuer(self.generatorLyft(True), False)
            self.trainQueuer.start(workers=1, max_queue_size=10)
            self.valQueuer = GeneratorEnqueuer(self.generatorLyft(False), False)
            self.valQueuer.start(workers=1, max_queue_size=10)
        else:
            self.trainQueuer = GeneratorEnqueuer(self.generator(True), False)
            self.trainQueuer.start(workers=1, max_queue_size=10)
            self.valQueuer = GeneratorEnqueuer(self.generator(False), False)
            self.valQueuer.start(workers=1, max_queue_size=10)
        self.number_of_classes = len(labels) if self.number_of_classes == 0 else self.number_of_classes
    
    @staticmethod
    def getCityName(path):
        return path.split('/')[-2]
    
    def convertTargetToImage(self, target):
        image = np.zeros((target.shape[0], target.shape[1], 3))
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                #if (label == None):
                #    print(image[i,j])
                if(i == 719 and j == target.shape[1]//2): print(target[i,j],
                         np.argmax(target[i,j]), self.id2label.get(np.argmax(target[i,j])).color)
                image[i,j, :] = self.id2label.get(target[i,j]).color[:]
        return image.astype('uint8')
    
    @staticmethod
    def getImageName(path):
        image = path.split('/')[-1]
        return image[:image.rfind('_')]
        
    def generator(self, batchSize, train, **kwargs):
        batchSize = 1
        start=0
        inputShape = mpimg.imread(self.train_data.iloc[0]['input'])[::2,::2].shape
        data = self.train_data if train else self.val_data
        while(1):
            imageArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
            labelArr = np.zeros((batchSize, inputShape[0] * inputShape[1], self.number_of_classes))
            #weights = np.ones((batchSize, self.number_of_classes))
            weights = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            if train:
                indices = np.random.randint(0, len(data), batchSize)
            else:
                indices = np.zeros(batchSize).astype(int)
                for i in range(batchSize):
                    indices[i] = start%len(data)
                    start += 1
                start = start%len(data)
            for i, index in zip(range(len(indices)), indices):
                #if(not train): print(imgList[index])
                image = mpimg.imread(data.iloc[index]['input'])[::2, ::2]
                targetName = data.iloc[index]['output']
                #print(targetName)
                target = mpimg.imread(targetName)[::2, ::2]
                print(targetName)
                target = (target*255).astype('uint8')
                target = CityScapes.convertTarget(target, self.dict)
                
                #plt.imshow(target)
                #plt.show()
                flip = np.random.rand()
                if flip>.5:
                    image = mirrorImage(image)
                    target = mirrorImage(target)
                if train: image, target = augmentImageAndLabel(image, target)
                imageArr[i] = image
                target = np.reshape(target, (inputShape[0]*inputShape[1]))
                """if(not np.any(self.weights == None)):
                    weights[i] = self.weights
                    weights *= 4 # Kind of normalisation factor
                """
                if(np.any(self.weights == None)):
                    for j in range(len(target)): 
                        weights[i,j] = 1
                else:
                    for j in range(len(target)): 
                        weights[i,j] = self.weights[int(target[j])]
                    weights *= 8 # Kind of normalisation factor
                print(weights)
                labelArr[i] = target
            yield({'input_img': imageArr}, {'output_img': labelArr}, weights)
            yield({'input_img': imageArr}, {'output_img': labelArr}, weights)

    def generatorLyft(self, train, **kwargs):
        max_q_size = 16
        maxproc = 8
        data = self.train_data if train else self.val_data
        try:
            queue = multiprocessing.Queue(maxsize=max_q_size)

            # define producer (putting items into queue)
            def producer():
                np.random.seed(int(1e7*(time.time()-int(time.time()))))
                while(1):
                    #if train:
                    index = np.random.randint(0, len(data))
                    index2 = np.random.randint(0, len(data))
                    #else:
                    #    with lock:
                    #        index = start
                    #        start += 1
                    #        start = start%len(data)
                    image = mpimg.imread(data.iloc[index]['input'])[::2, ::2]
                    targetName = data.iloc[index]['output']
                    target = mpimg.imread(targetName)[::2, ::2]
                    image2 = mpimg.imread(data.iloc[index2]['input'])[::2, ::2]
                    targetName2 = data.iloc[index2]['output']
                    target2 = mpimg.imread(targetName2)[::2, ::2]
                    w, h = target.shape
                    inputShape = image.shape
                    target = (target*255).astype('uint8')
                    f = np.vectorize(lambda x: self.lyftDict[x])  # or use a different name if you want to keep the original f
                    target = f(target)
                    target2 = (target2*255).astype('uint8')
                    target2 = f(target2)
                    flip = np.random.rand()
                    weighting = np.random.rand()
                    image = cv2.addWeighted(image, weighting,
                                            image2, 1-weighting, 0)
                    target = target.reshape(-1)
                    target = np.eye(self.number_of_classes)[target]
                    target = target.reshape(w, h, self.number_of_classes)
                    target2 = target2.reshape(-1)
                    target2 = np.eye(self.number_of_classes)[target2]
                    target2 = target2.reshape(w, h, self.number_of_classes)
                    target = cv2.addWeighted(target, weighting,
                                             target2, 1-weighting, 0)
                    if flip > .5:
                        image = mirrorImage(image)
                        target = mirrorImage(target)
                    if train: image, target = augmentImageAndLabel(image, target)
                    image = image/255.
                    imageArr = image.reshape((1, inputShape[0], inputShape[1], inputShape[2]))
                    labelArr = target.reshape((1, inputShape[0], inputShape[1], self.number_of_classes))
                    #print(index, data.iloc[index]['input'], image.shape, np.unique(target), train)
                    queue.put(({'input_img': imageArr}, 
                      {'output_img': labelArr}, data.iloc[index]['input']))

            processes = []

            def start_process():
                for val in range(len(processes), maxproc):
                    print("Spawn Thread %d" %val)
                    thread = multiprocessing.Process(target=producer)
                    time.sleep(0.01)
                    thread.start()
                    processes.append(thread)

            # run as consumer (read items from queue, in current thread)
            while True:
                processes = [p for p in processes if p.is_alive()]

                if len(processes) < maxproc:
                    start_process()

                yield queue.get()

        except:
            print("Finishing")
            for th in processes:
                th.terminate()
            queue.close()
            raise    
            
    def generatorAllClasses(self, batchSize, train, **kwargs):
        start=0
        batchSize=1
        inputShape = mpimg.imread(self.train_data.iloc[0]['input'])[::2,::2].shape
        data = self.train_data if train else self.val_data
        while(1):
            imageArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
            labelArr = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            #weights = np.ones((batchSize, self.number_of_classes))
            weights = np.ones((batchSize, inputShape[0] * inputShape[1]))
            if train:
                indices = np.random.randint(0, len(data), batchSize)
            else:
                indices = np.zeros(batchSize).astype(int)
                for i in range(batchSize):
                    indices[i] = start%len(data)
                    start += 1
                start = start%len(data)
            for i, index in zip(range(len(indices)), indices):
                image = mpimg.imread(data.iloc[index]['input'])[::2, ::2]
                
                target_name = data.iloc[index]['output']
                target = mpimg.imread(target_name)[::2, ::2]
                target = (target*255).astype('uint8')
                flip = np.random.rand()
                if flip > .5:
                    image = mirrorImage(image)
                    target = mirrorImage(target)
                if train:
                    image, target = augmentImageAndLabel(image, target)
                imageArr[i] = image
                target = np.reshape(target, (inputShape[0]*inputShape[1]))
                """if(not np.any(self.weights == None)):
                    weights[i] = self.weights
                    weights *= 4 # Kind of normalisation factor
                """
                if(np.any(self.weights == None)):
                    for j in range(len(target)): 
                        weights[i, j] = 1
                else:
                    for j in range(len(target)): 
                        weights[i, j] = self.weights[int(target[j])]
                    weights *= 8  # Kind of normalisation factor
                labelArr[i] = target
            yield({'input_img': imageArr}, {'output_img': labelArr}, weights)
            
    
    @staticmethod
    def get_sub_train_distribution(df, number_of_classes, dictionary):
        arr = np.zeros(number_of_classes)
        for handle in df['output']:
            target = mpimg.imread(handle)[::2, ::2]
            target = (target*255).astype('uint8')
            target[target == -1] = 0
            if(number_of_classes < 30):
                target = CityScapes.convertTarget(target, dictionary)
            for val in target.ravel():
                arr[int(val)] += 1
        return arr
    
    def get_train_distribution(self):
        subDfs = []
        # Zähle die Prozessoren
        cores = multiprocessing.cpu_count() 
        # Teile den DataFrame in cores gleichgrosse Teile
        for i in range(cores):
            subDfs.append(self.train_data.iloc[int(i/cores*len(self.train_data)):int((i+1)/cores*len(self.train_data)), :])
        print("Build SubDfs")
        # Öffne einen Pool mit ensprechend vielen Prozessen 
        pool = Pool(processes=cores)
        # Wende die Funktion an
        func = functools.partial(CityScapes.get_sub_train_distribution, number_of_classes=self.number_of_classes, dictionary=self.lyftDict)
        arrs = pool.map(func, subDfs)
        pool.close()
        arr = np.zeros(self.number_of_classes)
        for i in range(len(arrs)):
            arr += arrs[i]
            
        return arr
    
    
    def setWeights(self, weights):
        self.weights = weights


class ApolloScape(Dataset):

    def __init__(self, path, **kwargs):
        super().__init__(path=path)
        # image_list = glob.glob(self.path + '/road0[1-3]_ins/ColorImage/Record*/Camera*/*.jpg')
        image_list = glob.glob(self.path + '/road01_ins/ColorImage/Record0*/Camera*/*.jpg')
        self.train_data = pd.DataFrame(image_list, columns=['input'])
        self.train_data['output'] = self.train_data['input'].apply(
            lambda x: x.replace('ColorImage', 'Label')[:-4] + '_bin.png')
        self.train_data['depth'] = self.train_data['input'].apply(
            lambda x: x.replace('_ins', '_ins_depth')[:-4] + '.png')
        # image_list = glob.glob(self.path + '/road04_ins/ColorImage/Record*/Camera*/*.jpg')
        image_list = glob.glob(self.path + '/road01_ins/ColorImage/Record1*/Camera*/*.jpg')
        self.val_data = pd.DataFrame(image_list, columns=['input'])
        self.val_data['output'] = self.val_data['input'].apply(
            lambda x: x.replace('ColorImage', 'Label')[:-4] + '_bin.png')
        self.val_data['depth'] = self.val_data['input'].apply(
            lambda x: x.replace('_ins', '_ins_depth')[:-4] + '.png')
        self.number_of_classes = kwargs['number_of_classes']
        self.weights = None
        Label = namedtuple('Label', [

            'name',  # The identifier of this label, e.g. 'car', 'person', ... .
            # We use them to uniquely name a class

            'id',  # An integer ID that is associated with this label.
            # The IDs are used to represent the label in ground truth images
            # An ID of -1 means that this label does not have an ID and thus
            # is ignored when creating ground truth images (e.g. license plate).
            # Do not modify these IDs, since exactly these IDs are expected by the
            # evaluation server.

            'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
            # ground truth images with train IDs, using the tools provided in the
            # 'preparation' folder. However, make sure to validate or submit results
            # to our evaluation server using the regular IDs above!
            # For trainIds, multiple labels might have the same ID. Then, these labels
            # are mapped to the same class in the ground truth images. For the inverse
            # mapping, we use the label that is defined first in the list below.
            # For example, mapping all void-type classes to the same ID in training,
            # might make sense for some approaches.
            # Max value is 255!

            'category',  # The name of the category that this label belongs to

            'categoryId',  # The ID of this category. Used to create ground truth images
            # on category level.

            'hasInstances',  # Whether this label distinguishes between single instances or not

            'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
            # during evaluations or not

            'color',  # The color of this label
        ])

        labels = [
            # name  id    trainId   category            catId     hasInstances   ignoreInEval   color
            Label('others', 0, 255, 'others', 0, False, True, (0, 0, 0)),
            Label('rover', 1, 255, 'others', 0, False, True, (0, 0, 0)),
            Label('sky', 17, 0, 'sky', 0, False, True, (0, 0, 0)),
            Label('car', 33, 1, 'movable object', 0, False, True, (0, 0, 0)),
            Label('motorbicycle', 34, 2, 'movable object', 0, False, True, (111, 74, 0)),
            Label('bicycle', 35, 3, 'movable object', 1, False, False, (128, 64, 128)),
            Label('person', 36, 4, 'movable object', 1, False, False, (128, 64, 128)),
            Label('rider', 37, 5, 'movable object', 1, False, False, (128, 64, 128)),
            Label('truck', 38, 6, 'movable object', 1, False, False, (128, 64, 128)),
            Label('bus', 39, 7, 'movable object', 1, False, False, (128, 64, 128)),
            Label('tricycle', 40, 8, 'movable object', 1, False, False, (128, 64, 128)),
            Label('road', 49, 9, 'flat', 1, False, False, (128, 64, 128)),
            Label('sidewalk', 50, 10, 'flat', 1, False, False, (128, 64, 128)),
            Label('traffic_cone', 65, 11, 'road obstacles', 1, False, False, (128, 64, 128)),
            Label('road_pile', 66, 12, 'road obstacles', 1, False, False, (128, 64, 128)),
            Label('fence', 67, 13, 'road obstacles', 1, False, False, (128, 64, 128)),
            Label('traffic_light', 81, 14, 'roadside objects', 1, False, False, (128, 64, 128)),
            Label('pole', 82, 15, 'roadside objects', 1, False, False, (128, 64, 128)),
            Label('traffic_sign', 83, 16, 'roadside objects', 1, False, False, (128, 64, 128)),
            Label('wall', 84, 17, 'roadside objects', 1, False, False, (128, 64, 128)),
            Label('dustbin', 85, 18, 'roadside objects', 1, False, False, (128, 64, 128)),
            Label('billboard', 86, 19, 'roadside objects', 1, False, False, (128, 64, 128)),
            Label('building', 97, 20, 'building', 1, False, False, (128, 64, 128)),
            Label('bridge', 98, 255, 'building', 1, False, False, (128, 64, 128)),
            Label('tunnel', 99, 255, 'building', 1, False, False, (128, 64, 128)),
            Label('overpass', 100, 255, 'building', 1, False, False, (128, 64, 128)),
            Label('vegetation', 113, 21, 'natural', 1, False, False, (128, 64, 128)),
            Label('car_group', 161, 1, 'movable object', 0, False, True, (0, 0, 0)),
            Label('motorbicycle_group', 162, 2, 'movable object', 0, False, True, (111, 74, 0)),
            Label('bicycle_group', 163, 3, 'movable object', 1, False, False, (128, 64, 128)),
            Label('person_group', 164, 4, 'movable object', 1, False, False, (128, 64, 128)),
            Label('rider_group', 165, 5, 'movable object', 1, False, False, (128, 64, 128)),
            Label('truck_group', 166, 6, 'movable object', 1, False, False, (128, 64, 128)),
            Label('bus_group', 167, 7, 'movable object', 1, False, False, (128, 64, 128)),
            Label('tricycle_group', 168, 8, 'movable object', 1, False, False, (128, 64, 128)),
            Label('unlabeled', 255, 255, 'unlabled', 1, False, False, (128, 64, 128)),
        ]
        self.name2label = {label.name: label for label in labels}
        # id to label object
        self.id2label = {label.id: label for label in labels}
        self.dict = {
            0: 0,  # Void
            1: 0,  # Rover -> Void
            17: 21,  # Sky
            33: 7,  # Car
            34: 25,  # Motorcycle
            35: 5,  # Bicycle
            36: 9,  # Person -> Pedestrian
            37: 5,  # Rider -> Bicycle
            38: 29,  # Truck -> Truck_bus
            39: 29,  # Bus -> Truck_bus
            40: 17,  # Tricycle -> Other Moving
            49: 18,  # Road
            50: 2,  # Sidewalk
            65: 31,  # traffic_cone -> column_pole
            66: 30,  # road_pile -> Misc Vegetation
            67: 10,  # Fence
            81: 3,  # Traffic Light
            82: 31,  # Pole -> column_pole
            83: 28,  # Traffic sign -> SignSymbol
            84: 13,  # Wall -> Building/Wall
            85: 13,  # dustbin -> Building/Wall
            86: 13,  # billboard -> Building/Wall
            97: 13,  # building -> Building/Wall
            98: 4,  # bridge
            99: 6,  # tunnel
            100: 4,  # overpass -> bridge
            113: 30,  # vegetation -> Misc Vegetation
            161: 7,  # Car_group -> Car
            162: 25,  # Motorcycle_group -> Motorcycle
            163: 5,  # Bicycle_group -> Bicycle
            164: 9,  # Person_group -> Pedestrian
            165: 5,  # Rider_group -> Bicycle
            166: 29,  # Truck_group -> Truck_bus
            167: 29,  # Bus_group -> Truck_bus
            168: 17,  # Tricycle_group -> Other Moving
            255: 0,  # unlabeled (unluckily this is mainly sky...) -> Unlabeled
        }
        self.thunderhill_dict = {
            0: 0,  # Void
            1: 0,  # Rover -> Void
            17: 4,  # Sky
            33: 11,  # Car
            34: 13,  # Motorcycle -> Cyclish
            35: 13,  # Bicycle -> Cyclish
            36: 5,  # Person
            37: 13,  # Rider -> Cyclish
            38: 12,  # Truck -> big vehicle
            39: 12,  # Bus -> big vehicle
            40: 13,  # Tricycle -> Cyclish
            49: 1,  # Road
            50: 6,  # Sidewalk
            65: 8,  # traffic_cone -> pole
            66: 3,  # road_pile -> vegetation/dirt
            67: 7,  # Fence -> Wall/Fence
            81: 9,  # Traffic Light
            82: 8,  # Pole -> pole
            83: 10,  # Traffic sign -> traffic-sign
            84: 7,  # Wall -> Wall/Fence
            85: 2,  # dustbin -> static
            86: 2,  # billboard -> static
            97: 2,  # building -> static
            98: 2,  # bridge -> static
            99: 2,  # tunnel -> static
            100: 2,  # overpass -> static
            113: 3,  # vegetation -> vegetation/dirt
            161: 11,  # Car_group -> Car
            162: 13,  # Motorcycle_group -> Cyclish
            163: 13,  # Bicycle_group -> Cyclish
            164: 5,  # Person_group -> Person
            165: 13,  # Rider_group -> Cyclish
            166: 22,  # Truck_group -> big vehicle
            167: 22,  # Bus_group -> big vehicle
            168: 13,  # Tricycle_group -> Cyclish
            255: 4,  # unlabeled (unluckily this is mainly sky...) -> Sky
        }
        labels_thunderhill = [
            #       name                   id    trainId   category    catId     hasInstances   ignoreInEval   color
            Label(  'void'            ,  0 ,    4 , 'void'         , 4 , False , False , (  0,  0,  0) ),
            Label(  'road'            ,  1 ,    4 , 'flat'         , 4 , False , False , (128, 64,128) ),
            Label(  'static'          ,  2 ,  255 , 'construction' , 2 , False , False , ( 70, 70, 70) ),
            Label(  'vegetation/dirt' ,  3 ,  255 , 'flat'         , 2 , False , False , (107,142, 35) ),
            Label(  'sky'             ,  4 ,  255 , 'nature'       , 2 , False , False , ( 70,130,180) ),
            Label(  'person'          ,  5 ,  255 , 'human'        , 2 , False , False , (220, 20, 60) ),
            Label(  'sidewalk'        ,  6 ,  255 , 'flat'         , 2 , False , False , (244, 35,232) ),
            Label(  'Wall/Fence'      ,  7 ,    1 , 'construction' , 1 , False , False , (190,153,153) ),
            Label(  'pole'            ,  8 ,    1 , 'object'       , 1 , False , False , (153,153,153) ),
            Label(  'traffic-light'   ,  9 ,  255 , 'object'       , 1 , False , False , (250,170, 30) ),
            Label(  'traffic-sign'    , 10 ,  255 , 'object'       , 1 , False , False , (220,220,220) ),
            Label(  'car'             , 11 ,    2 , 'object'       , 1 , False , False , (  0,  0,142) ),
            Label(  'big vehicle'     , 12 ,    3 , 'object'       , 1 , False , False , (  0,  0, 70) ),
            Label(  'Cyclish'         , 13 ,    4 , 'object'       , 1 , False , False , (  0,  0,230) ),
            Label(  'Marking'         , 14 ,  255 , 'flat'         , 1 , False , False , (200,128,128) ),
        ]

        self.name2label = {label.name: label for label in labels}
        # id to label object
        self.id2label = {label.id: label for label in labels}
        if self.number_of_classes == 15:
            self.name2label = {label.name: label for label in labels_thunderhill}
            # id to label object
            self.id2label = {label.id: label for label in labels_thunderhill}
        if self.number_of_classes == 0:
            self.trainQueuer = self.generatorAllClasses(True)
            time.sleep(1)
            self.valQueuer = self.generatorAllClasses(False)

        elif self.number_of_classes == 15:
            self.gen_function = self.generator_thunderhill
            self.colorDict = {label.id: label.color for label in labels_thunderhill}
        elif self.number_of_classes == 3:
            self.gen_function = self.generator_thunderhill_res
        else:
            self.gen_function = self.generator
        self.number_of_classes = len(labels) if self.number_of_classes == 0 else self.number_of_classes
        self.train_queue = None
        self.val_queue = None
        self.train_processes = []
        self.val_processes = []
        self.running = False
        self.number_of_classes = len(labels) if self.number_of_classes == 0 else self.number_of_classes

    def convert_target_to_image(self, target):
        image = np.zeros((target.shape[0], target.shape[1], 3))
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if i == 719 and j == target.shape[1] // 2:
                    print(target[i, j],
                          np.argmax(target[i, j]), self.id2label.get(np.argmax(target[i, j])).color)
                image[i, j, :] = self.id2label.get(target[i, j]).color[:]
        return image.astype('uint8')

    def generator(self, data, queue, train, **kwargs):
        index = np.random.randint(0, len(data))
        image = mpimg.imread(data.iloc[index]['input'])
        if image.shape[0] < 600 or image.shape[1] < 600:
            index = np.random.randint(0, len(data))
            image = mpimg.imread(data.iloc[index]['input'])
        target_name = data.iloc[index]['output']
        target = mpimg.imread(target_name)
        while image.shape[0] > 1300 or image.shape[1] > 1300:
            image = image[::2, ::2]
            target = target[::2, ::2]
        input_shape = image.shape
        image_arr = np.zeros((1, input_shape[0], input_shape[1], input_shape[2]))
        label_arr = np.zeros((1, input_shape[0], input_shape[1]))
        target = (target * 255).astype('uint8')
        flip = np.random.rand()
        if flip > .5:
            image = mirrorImage(image)
            target = mirrorImage(target)
        if train:
            image, target = augmentImageAndLabel(image, target)
        image_arr[0] = image
        for i in range(len(target)):
            for j in range(target.shape[1]):
                target[i, j] = self.dict[target[i, j]]
        label_arr[0] = target
        yield ({'input_img': Variable(torch.from_numpy(image_arr)).float().permute(0, 3, 1, 2)},
               {'output_img': Variable(torch.from_numpy(label_arr)).long()})

    def generatorAllClasses(self, train, **kwargs):
        max_q_size = 16
        maxproc = 8
        data = self.train_data if train else self.val_data
        try:
            queue = multiprocessing.Queue(maxsize=max_q_size)

            # define producer (putting items into queue)
            def producer():
                inputShape = (1024, 768, 3)
                np.random.seed(int(1e7 * (time.time() - int(time.time()))))
                while (1):
                    # if train:
                    index = np.random.randint(0, len(data))
                    # else:
                    #    with lock:
                    #        index = start
                    #        start += 1
                    #        start = start%len(data)
                    image = mpimg.imread(data.iloc[index]['input'])
                    targetName = data.iloc[index]['output']
                    target = mpimg.imread(targetName)
                    while (image.shape[0] > 1300 or image.shape[1] > 1300):
                        image = image[::2, ::2]
                        target = target[::2, ::2]
                    if (image.shape[0] < 600 or image.shape[1] < 600): continue
                    inputShape = image.shape
                    target = (target * 255).astype('uint8')
                    flip = np.random.rand()
                    if flip > .5:
                        image = mirrorImage(image)
                        target = mirrorImage(target)
                    if train: image, target = augmentImageAndLabel(image, target)
                    image = image / 255.
                    imageArr = image.reshape((1, inputShape[0], inputShape[1], inputShape[2]))
                    labelArr = target.reshape((1, inputShape[0], inputShape[1])).astype('uint8')
                    # print(index, data.iloc[index]['input'], image.shape, np.unique(target), train)
                    queue.put(({'input_img': imageArr},
                               {'output_img': labelArr}, data.iloc[index]['input']))

            processes = []

            def start_process():
                for val in range(len(processes), maxproc):
                    print("Spawn Thread %d" % val)
                    thread = multiprocessing.Process(target=producer)
                    time.sleep(0.01)
                    thread.start()
                    processes.append(thread)

            # run as consumer (read items from queue, in current thread)
            while True:
                processes = [p for p in self.processes if p.is_alive()]

                if len(processes) < maxproc:
                    start_process()

                yield queue.get()

        except:
            print("Finishing")
            for th in self.processes:
                th.terminate()
            queue.close()
            raise

    def __next__(self, train):
        if train:
            return self.train_queue.get()
        else:
            return self.val_queue.get()

    def launch_generators(self, max_q_size, max_proc):
        if self.train_queue is not None:
            return
        self.running = True
        self.train_queue = multiprocessing.Queue(maxsize=max_q_size)
        self.val_queue = multiprocessing.Queue(maxsize=max_q_size)
        # try:

        for train in [True, False]:
            processes = self.train_processes if train else self.val_processes
            data = self.train_data if train else self.val_data
            queue = self.train_queue if train else self.val_queue

            def start_process():
                np.random.seed(int(1e7 * (time.time() - int(time.time()))))
                for val in range(len(processes), max_proc):
                    print("Spawn Thread %d" % val)
                    producer = functools.partial(
                        self.gen_function, data=data, queue=queue, train=train)
                    thread = multiprocessing.Process(target=producer, daemon=True)
                    time.sleep(0.01)
                    thread.start()
                    processes.append(thread)

            processes = [p for p in processes if p.is_alive()]

            if len(processes) < max_proc:
                start_process()

        # except:
        #     print("Finishing")
        #     for th in processes:
        #         th.terminate()
        #     queue.close()
        #     raise

    def stop_generators(self):
        self.running = False
        for train in [True, False]:
            processes = self.train_processes if train else self.val_processes
            for process in processes:
                process.terminate()

    def generator_thunderhill(self, data, queue, train, **kwargs):
        while 1:
            index = np.random.randint(0, len(data))
            image = mpimg.imread(data.iloc[index]['input'])
            target_name = data.iloc[index]['output']
            try:
                target = mpimg.imread(target_name)
            except FileNotFoundError:
                pass
            while not os.path.exists(target_name) or np.all(target == target[0, 0]):
                index = np.random.randint(0, len(data))
                image = mpimg.imread(data.iloc[index]['input'])
                target_name = data.iloc[index]['output']
                try:
                    target = mpimg.imread(target_name)
                except FileNotFoundError:
                    pass
            image = image[::2, ::2]
            target = target[::2, ::2]
            input_shape = image.shape
            target = (target * 255).astype('uint8')
            f = np.vectorize(lambda x: self.thunderhill_dict.get(x, 0))
            # or use a different name if you want to keep the original f
            target = f(target)
            depth_name = data.iloc[index]['output']
            depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH) / 200.
            depth = depth[::2, ::2]
            flip = np.random.rand()
            if flip > .5:
                image = mirrorImage(image)
                target = mirrorImage(target)
                depth = mirrorImage(depth)
            if train:
                image, target, depth = augmentImageAndLabel(image, target, depth)
            image = image / 255.
            image_arr = image.reshape((1, input_shape[0], input_shape[1], input_shape[2]))
            label_arr = target.reshape((1, input_shape[0], input_shape[1]))
            label_arr = label_arr.astype('uint8')
            depth_arr = depth.reshape((1, input_shape[0], input_shape[1]))
            # print(index, data.iloc[index]['input'], image.shape, np.unique(target), train)
            queue.put(({'input_img': image_arr},
                       {'output_img': label_arr, 'depth_img': depth_arr},
                       data.iloc[index]['input']))

    @staticmethod
    def get_sub_train_distribution(df, number_of_classes, dictionary):
        arr = np.zeros(number_of_classes)
        f = np.vectorize(lambda x: dictionary.get(x, 0))
        for handle in df['output']:
            if not os.path.exists(handle):
                continue
            target = mpimg.imread(handle)[::4, ::4]
            if np.all(target == target[0, 0]):
                continue
            target = (target*255).astype('uint8')
            if number_of_classes < 30:
                target = f(target)
            print(np.unique(target))
            for val in target.ravel():
                arr[int(val)] += 1
        return arr

    def get_train_distribution(self):
        sub_dfs = []
        # Zähle die Prozessoren
        cores = multiprocessing.cpu_count()
        # Teile den DataFrame in cores gleichgrosse Teile
        for i in range(cores):
            sub_dfs.append(self.train_data.iloc[int(i/cores*len(self.train_data)):int((i+1)/cores*len(self.train_data)), :])

        print("Build SubDfs")
        # Öffne einen Pool mit ensprechend vielen Prozessen
        pool = Pool(processes=cores)
        # Wende die Funktion an
        func = functools.partial(ApolloScape.get_sub_train_distribution,
                                 number_of_classes=self.number_of_classes,
                                 dictionary=self.thunderhill_dict)
        arrs = pool.map(func, sub_dfs)
        pool.close()
        arr = np.zeros(self.number_of_classes)
        for i in range(len(arrs)):
            print(arrs[i])
            arr += arrs[i]
        return arr

    def setWeights(self, weights):
        self.weights = weights

        
class Mapillary(Dataset):
    
    def __init__(self, path, **kwargs):
        super(Mapillary, self).__init__(path=path)
        image_list = glob.glob(self.path + 'mapillary-vistas-dataset_public_v1.0/training/images/*')
        print(self.path + 'mapillary-vistas-dataset_public_v1.0/training/images/*')
        self.train_data = pd.DataFrame(image_list, columns=['input'])
        self.train_data['output'] = self.train_data['input'].apply(
            lambda x: self.path + 'mapillary-vistas-dataset_public_v1.0/training/instances/'+
            x.split('/')[-1][:-4]+'.png')
        image_list = glob.glob(self.path + 'mapillary-vistas-dataset_public_v1.0/validation/images/*')
        self.val_data = pd.DataFrame(image_list, columns=['input'])
        self.val_data['output'] = self.val_data['input'].apply(
            lambda x: self.path + 'mapillary-vistas-dataset_public_v1.0/validation/instances/' +
            x.split('/')[-1][:-4]+'.png')
        self.number_of_classes = kwargs['number_of_classes']
        self.weights = None
        Label = namedtuple( 'Label' , [
            'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                            # We use them to uniquely name a class
        
            'id'          , # An integer ID that is associated with this label.
                            # The IDs are used to represent the label in ground truth images
                            # An ID of -1 means that this label does not have an ID and thus
                            # is ignored when creating ground truth images (e.g. license plate).
                            # Do not modify these IDs, since exactly these IDs are expected by the
                            # evaluation server.
        
            'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                            # ground truth images with train IDs, using the tools provided in the
                            # 'preparation' folder. However, make sure to validate or submit results
                            # to our evaluation server using the regular IDs above!
                            # For trainIds, multiple labels might have the same ID. Then, these labels
                            # are mapped to the same class in the ground truth images. For the inverse
                            # mapping, we use the label that is defined first in the list below.
                            # For example, mapping all void-type classes to the same ID in training,
                            # might make sense for some approaches.
                            # Max value is 255!
        
            'category'    , # The name of the category that this label belongs to
        
            'categoryId'  , # The ID of this category. Used to create ground truth images
                            # on category level.
        
            'hasInstances', # Whether this label distinguishes between single instances or not
        
            'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                            # during evaluations or not
        
            'color'       , # The color of this label
        ])
        
        labels = [
            #       name                                     id    trainId   category    catId     hasInstances   ignoreInEval   color
            Label(  'animal--bird'                          ,  0 ,    4 , 'animal'       , 4 , True  , True  , (165,  42,  42) ),
            Label(  'animal--ground-animal'                 ,  1 ,    4 , 'animal'       , 4 , True  , True  , (  0, 192,   0) ),
            Label(  'construction--barrier--curb'           ,  2 ,  255 , 'construction' , 2 , False , True  , (196, 196, 196) ),
            Label(  'construction--barrier--fence'          ,  3 ,  255 , 'construction' , 2 , False , True  , (190, 153, 153) ),
            Label(  'construction--barrier--guard-rail'     ,  4 ,  255 , 'construction' , 2 , False , True  , (180, 165,180) ),
            Label(  'construction--barrier--other-barrier'  ,  5 ,  255 , 'construction' , 2 , False , True  , (102,102,156) ),
            Label(  'construction--barrier--wall'           ,  6 ,  255 , 'construction' , 2 , False , True  , (102,102,156) ),
            Label(  'construction--flat--bike-lane'         ,  7 ,    1 , 'flat'         , 1 , False , False , (128, 64,255) ),
            Label(  'construction--flat--crosswalk-plain'   ,  8 ,    1 , 'flat'         , 1 , True  , False , (140,140,200) ),
            Label(  'construction--flat--curb-cut'          ,  9 ,  255 , 'flat'         , 1 , False , True  , (170,170,170) ),
            Label(  'construction--flat--parking'           , 10 ,  255 , 'flat'         , 1 , False , True  , (250,170,160) ),
            Label(  'construction--flat--pedestrian-area'   , 11 ,    2 , 'flat'         , 1 , False , False , (96,96,96) ),
            Label(  'construction--flat--rail-track'        , 12 ,    3 , 'flat'         , 1 , False , False , (230,150,140) ),
            Label(  'construction--flat--road'              , 13 ,    4 , 'flat'         , 1 , False , False , (128, 64,128) ),
            Label(  'construction--flat--service-lane'      , 14 ,  255 , 'flat'         , 1 , False , True  , (110,110,110) ),
            Label(  'construction--flat--sidewalk'          , 15 ,  255 , 'flat'         , 1 , False , True  , (244, 35,232) ),
            Label(  'construction--structure--bridge'       , 16 ,  255 , 'construction' , 2 , False , True  , (150,100,100) ),
            Label(  'construction--structure--building'     , 17 ,    5 , 'construction' , 2 , False , False , ( 70, 70, 70) ),
            Label(  'construction--structure--tunnel'       , 18 ,  255 , 'construction' , 2 , False , True  , (150,120, 90) ),
            Label(  'human--person'                         , 19 ,    6 , 'human'        , 6 , True  , False , (220, 20, 60) ),
            Label(  'human--rider--bicyclist'               , 20 ,    7 , 'human'        , 6 , True  , False , (255,  0,  0) ),
            Label(  'human--rider--motorcyclist'            , 21 ,    8 , 'human'        , 6 , True  , False , (255,  0,  0) ),
            Label(  'human--rider--other-rider'             , 22 ,    9 , 'human'        , 6 , True  , False , (255,  0,  0) ),
            Label(  'marking--crosswalk-zebra'              , 23 ,   10 , 'marking'      , 8 , True  , False , (200,128,128) ),
            Label(  'marking--general'                      , 24 ,   11 , 'marking'      , 8 , False , False , (255,255,255) ),
            Label(  'nature--mountain'                      , 25 ,   12 , 'nature'       , 4 , False , False , ( 64,170, 64) ),
            Label(  'nature--sand'                          , 26 ,   13 , 'nature'       , 4 , False , False , (128, 64,128) ),
            Label(  'nature--sky'                           , 27 ,   12 , 'nature'       , 4 , False , False , ( 70,130,180) ),
            Label(  'nature--snow'                          , 28 ,   13 , 'nature'       , 4 , False , False , (255,255,255) ),
            Label(  'nature--terrain'                       , 29 ,   13 , 'nature'       , 4 , False , False , (152,251,152) ),
            Label(  'nature--vegetation'                    , 30 ,   12 , 'nature'       , 4 , False , False , (107,142,35) ),
            Label(  'nature--water'                         , 31 ,   13 , 'nature'       , 4 , False , False , (  0,170, 30) ),
            Label(  'object--banner'                        , 32 ,   13 , 'object'       , 3 , True  , False , (255,255,128) ),
            Label(  'object--bench'                         , 33 ,   12 , 'object'       , 3 , True  , False , (250,  0, 30) ),
            Label(  'object--bike-rack'                     , 34 ,   13 , 'object'       , 3 , True  , False , (  0,  0,  0) ),
            Label(  'object--billboard'                     , 35 ,   13 , 'object'       , 3 , True  , False , (220,220,220) ),
            Label(  'object--catch-basin'                   , 36 ,   12 , 'object'       , 3 , True  , False , (170,170,170) ),
            Label(  'object--cctv-camera'                   , 37 ,   13 , 'object'       , 3 , True  , False , (222, 40, 40) ),
            Label(  'object--fire-hydrant'                  , 38 ,   13 , 'object'       , 3 , True  , False , (100,170, 30) ),
            Label(  'object--junction-box'                  , 39 ,   12 , 'object'       , 3 , True  , False , (40, 40, 40) ),
            Label(  'object--mailbox'                       , 40 ,   13 , 'object'       , 3 , True  , False , ( 33, 33, 33) ),
            Label(  'object--manhole'                       , 41 ,   13 , 'object'       , 3 , True  , False , (170,170,170) ),
            Label(  'object--phone-booth'                   , 42 ,   12 , 'object'       , 3 , True  , False , (  0,  0,142) ),
            Label(  'object--pothole'                       , 43 ,   13 , 'object'       , 3 , True  , False , (170,170,170) ),
            Label(  'object--street-light'                  , 44 ,   13 , 'object'       , 3 , True  , False , (210,170,100) ),
            Label(  'object--support--pole'                 , 45 ,   12 , 'object'       , 3 , True  , False , (153,153,153) ),
            Label(  'object--traffic-sign-frame'            , 46 ,   13 , 'object'       , 3 , True  , False , (128,128,128) ),
            Label(  'object--utility-pole'                  , 47 ,   13 , 'object'       , 3 , True  , False , (  0,  0,142) ),
            Label(  'object--traffic-light'                 , 48 ,   12 , 'object'       , 3 , True  , False , (250,170, 30) ),
            Label(  'object--traffic-sign--back'            , 49 ,   13 , 'object'       , 3 , True  , False , (192,192,192) ),
            Label(  'object--traffic-sign--front'           , 50 ,   13 , 'object'       , 3 , True  , False , (220,220,220) ),
            Label(  'object--trash-can'                     , 51 ,   13 , 'object'       , 3 , True  , False , (180,165,180) ),
            Label(  'object--vehicle--bicycle'              , 52 ,   12 , 'vehicle'      , 7 , True  , False , (119, 11, 32) ),
            Label(  'object--vehicle--boat'                 , 53 ,   13 , 'vehicle'      , 7 , True  , False , (  0,  0,142) ),
            Label(  'object--vehicle--bus'                  , 54 ,   13 , 'vehicle'      , 7 , True  , False , (  0, 60,100) ),
            Label(  'object--vehicle--car'                  , 55 ,   12 , 'vehicle'      , 7 , True  , False , (  0,  0,142) ),
            Label(  'object--vehicle--caravan'              , 56 ,   13 , 'vehicle'      , 7 , True  , False , (  0,  0, 90) ),
            Label(  'object--vehicle--motorcycle'           , 57 ,   13 , 'vehicle'      , 7 , True  , False , (  0,  0,230) ),
            Label(  'object--vehicle--on-rails'             , 58 ,   12 , 'vehicle'      , 7 , True  , False , (  0, 80,100) ),
            Label(  'object--vehicle--other-vehicle'        , 59 ,   13 , 'vehicle'      , 7 , True  , False , (128, 64,128) ),
            Label(  'object--vehicle--trailer'              , 60 ,   13 , 'vehicle'      , 7 , True  , False , (  0,  0,110) ),
            Label(  'object--vehicle--truck'                , 61 ,   12 , 'vehicle'      , 7 , True  , False , (  0,  0, 70) ),
            Label(  'object--vehicle--wheeled-slow'         , 62 ,   13 , 'vehicle'      , 7 , True  , False , (  0,  0,192) ),
            Label(  'void--car-mount'                       , 63 ,   13 , 'void'         , 7 , False , False , ( 32, 32, 32) ),
            Label(  'void--ego-vehicle'                     , 64 ,   12 , 'void'         , 7 , False , False , (  0,  0,  0) ),
            Label(  'void--unlabeled'                       , 65 ,   13 , 'void'         , 7 , True  , False , (  0,  0,  0) ),
        ]
        self.dict = {
            0: 11, # Bird -> Animal
            1: 11, # Ground Animal -> Animal
            2: 12, # Curb -> Wall
            3: 13, # Fence
            4: 14, # Guard Rail
            5: 17, # Other Barrier -> Wall
            6: 22, # Wall
            7: 8, # Bike Lane -> Sidewalk
            8: 7, # Crosswalk -> Road
            9: 7, # curb-cut -> Road
            10: 9, # Parking
            11: 8, # Pedestrian Area -> Sidewalk
            12: 10, # Rail Track
            13: 7, # Road
            14: 7, # Service Lane -> Road
            15: 8, # Sidewalk
            16: 15, # Bride
            17: 11, # Building
            18: 16, # Tunnel
            19: 24, # Person
            20: 24, # Bycyclist -> Person
            21: 24, # Motorcyclist -> Person
            22: 24, # Other Rider -> Person
            23: 34, # Marking Crosswalk Zebra -> New Id
            24: 34, # Marking General -> New Id
            25: 22, # Mountain -> Terrain
            26: 22, # Sand -> Terrain
            27: 23, # Sky
            28: 29, # Snow -> Terrain
            29: 22, # Terrain
            30: 21, # Vegetation
            31: 22, # Water -> Terrain
            32: 11, # Banner -> Building
            33: 4, # Bench -> static
            34: 5, # Bike Rack -> dynamic
            35: 4, # BillBoard -> Static
            36: 4, # catch-basin -> static
            37: 4, # cctv-camera -> static
            38: 4, # Fire Hydrant -> static
            39: 4, # junction box -> static
            40: 4, # mailbox -> static
            41: 4, # manhole -> static
            42: 4, # phone booth -> static
            43: 7, # pothole -> road
            44: 17, # street-light -> pole
            45: 17, # pole
            46: 17, # sign frame -> pole
            47: 17, # pole 
            48: 19, # traffic light
            49: 20, # traffic sign back -> traffic sign
            50: 4, # traffic sign front -> static
            51: 4, # trash can -> static
            52: 33, # Bicycle
            53: 5, # Boat -> Dynamic
            54: 28, # Bus
            55: 26, # Car
            56: 29, # Caravan
            57: 32, # Motorcycle
            58: 31, # on rails -> train
            59: 17, # other vehicle -> Car
            60: 30, # trailer
            61: 27, # truck
            62: 27, # wheeled slow -> truck
            63: 1, # car-mount -> ego Car
            64: 1, # EgoCar
            65: 0, # Void
        }
        self.thunderhill_dict = {
            0: 0, # Bird -> Animal
            1: 0, # Ground Animal -> Animal
            2: 7, # Curb -> Wall
            3: 7, # Fence
            4: 7, # Guard Rail
            5: 7, # Other Barrier -> Wall
            6: 7, # Wall
            7: 1, # Bike Lane -> Road
            8: 1, # Crosswalk -> Road
            9: 1, # curb-cut -> Road
            10: 1, # Parking -> Road
            11: 1, # Pedestrian Area -> Road
            12: 1, # Rail Track
            13: 1, # Road
            14: 1, # Service Lane -> Road
            15: 6, # Sidewalk
            16: 2, # Bride -> static
            17: 2, # Building -> static
            18: 2, # Tunnel -> static
            19: 5, # Person
            20: 5, # Bycyclist -> Person
            21: 5, # Motorcyclist -> Person
            22: 5, # Other Rider -> Person
            23: 14, # Marking Crosswalk Zebra -> Marking
            24: 14, # Marking General -> Markings
            25: 3, # Mountain -> Terrain
            26: 3, # Sand -> Terrain
            27: 4, # Sky
            28: 3, # Snow -> Terrain
            29: 3, # Terrain
            30: 3, # Vegetation -> Terrain
            31: 3, # Water -> Terrain
            32: 2, # Banner -> static
            33: 2, # Bench -> static
            34: 2, # Bike Rack -> static
            35: 2, # BillBoard -> Static
            36: 2, # catch-basin -> static
            37: 2, # cctv-camera -> static
            38: 2, # Fire Hydrant -> static
            39: 2, # junction box -> static
            40: 2, # mailbox -> static
            41: 1, # manhole -> road
            42: 2, # phone booth -> static
            43: 1, # pothole -> road
            44: 8, # street-light -> pole
            45: 8, # pole
            46: 8, # sign frame -> pole
            47: 8, # pole 
            48: 9, # traffic light
            49: 10, # traffic sign back -> traffic sign
            50: 2, # traffic sign front -> static
            51: 2, # trash can -> static
            52: 13, # Bicycle -> Cyclish
            53: 0, # Boat -> Void
            54: 12, # Bus -> Big Vehicle
            55: 11, # Car
            56: 12, # Caravan -> Big Vehicle
            57: 13, # Motorcycle -> Cyclish
            58: 12, # on rails -> Big Vehicle
            59: 11, # other vehicle -> Car
            60: 11, # trailer -> Big Vehicle
            61: 11, # truck -> Big Vehicle
            62: 11, # wheeled slow -> Big Vehicle
            63: 0, # car-mount -> Void
            64: 0, # EgoCar -> Void
            65: 0, # Void
        }
        self.thunderhill_dict_res = {
            0: 0, # Bird -> Animal
            1: 0, # Ground Animal -> Animal
            2: 0, # Curb -> Wall
            3: 0, # Fence
            4: 0, # Guard Rail
            5: 0, # Other Barrier -> Wall
            6: 0, # Wall
            7: 1, # Bike Lane -> Sidewalk
            8: 1, # Crosswalk -> Road
            9: 1, # curb-cut -> Road
            10: 1, # Parking -> Road
            11: 1, # Pedestrian Area -> Road
            12: 1, # Rail Track
            13: 1, # Road
            14: 1, # Service Lane -> Road
            15: 0, # Sidewalk
            16: 0, # Bride -> static
            17: 0, # Building -> static
            18: 0, # Tunnel -> static
            19: 0, # Person
            20: 0, # Bycyclist -> Person
            21: 0, # Motorcyclist -> Person
            22: 0, # Other Rider -> Person
            23: 2, # Marking Crosswalk Zebra -> Marking
            24: 2, # Marking General -> Markings
            25: 0, # Mountain -> Terrain
            26: 0, # Sand -> Terrain
            27: 0, # Sky
            28: 0, # Snow -> Terrain
            29: 0, # Terrain
            30: 0, # Vegetation -> Terrain
            31: 0, # Water -> Terrain
            32: 0, # Banner -> static
            33: 0, # Bench -> static
            34: 0, # Bike Rack -> static
            35: 0, # BillBoard -> Static
            36: 0, # catch-basin -> static
            37: 0, # cctv-camera -> static
            38: 0, # Fire Hydrant -> static
            39: 0, # junction box -> static
            40: 0, # mailbox -> static
            41: 1, # manhole -> road
            42: 0, # phone booth -> static
            43: 1, # pothole -> road
            44: 0, # street-light -> pole
            45: 0, # pole
            46: 0, # sign frame -> pole
            47: 0, # pole 
            48: 0, # traffic light
            49: 0, # traffic sign back -> traffic sign
            50: 0, # traffic sign front -> static
            51: 0, # trash can -> static
            52: 0, # Bicycle -> Cyclish
            53: 0, # Boat -> Void
            54: 0, # Bus -> Big Vehicle
            55: 0, # Car
            56: 0, # Caravan -> Big Vehicle
            57: 0, # Motorcycle -> Cyclish
            58: 0, # on rails -> Big Vehicle
            59: 0, # other vehicle -> Car
            60: 0, # trailer -> Big Vehicle
            61: 0, # truck -> Big Vehicle
            62: 0, # wheeled slow -> Big Vehicle
            63: 0, # car-mount -> Void
            64: 0, # EgoCar -> Void
            65: 0, # Void
        }
        labels_thunderhill = [
            #       name                   id    trainId   category    catId     hasInstances   ignoreInEval   color
            Label(  'void'            ,  0 ,    4 , 'void'         , 4 , False , False , (  0,  0,  0) ),
            Label(  'road'            ,  1 ,    4 , 'flat'         , 4 , False , False , (128, 64,128) ),
            Label(  'static'          ,  2 ,  255 , 'construction' , 2 , False , False , ( 70, 70, 70) ),
            Label(  'vegetation/dirt' ,  3 ,  255 , 'flat'         , 2 , False , False , (107,142, 35) ),
            Label(  'sky'             ,  4 ,  255 , 'nature'       , 2 , False , False , ( 70,130,180) ),
            Label(  'person'          ,  5 ,  255 , 'human'        , 2 , False , False , (220, 20, 60) ),
            Label(  'sidewalk'        ,  6 ,  255 , 'flat'         , 2 , False , False , (244, 35,232) ),
            Label(  'Wall/Fence'      ,  7 ,    1 , 'construction' , 1 , False , False , (190,153,153) ),
            Label(  'pole'            ,  8 ,    1 , 'object'       , 1 , False , False , (153,153,153) ),
            Label(  'traffic-light'   ,  9 ,  255 , 'object'       , 1 , False , False , (250,170, 30) ),
            Label(  'traffic-sign'    , 10 ,  255 , 'object'       , 1 , False , False , (220,220,220) ),
            Label(  'car'             , 11 ,    2 , 'object'       , 1 , False , False , (  0,  0,142) ),
            Label(  'big vehicle'     , 12 ,    3 , 'object'       , 1 , False , False , (  0,  0, 70) ),
            Label(  'Cyclish'         , 13 ,    4 , 'object'       , 1 , False , False , (  0,  0,230) ),
            Label(  'Marking'         , 14 ,  255 , 'flat'         , 1 , False , False , (200,128,128) ),
        ]
        self.lyft_dict = {
            0: 0, # Bird -> Animal
            1: 0, # Ground Animal -> Animal
            2: 7, # Curb -> Wall
            3: 7, # Fence
            4: 7, # Guard Rail
            5: 7, # Other Barrier -> Wall
            6: 7, # Wall
            7: 1, # Bike Lane -> Road
            8: 1, # Crosswalk -> Road
            9: 1, # curb-cut -> Road
            10: 1, # Parking -> Road
            11: 1, # Pedestrian Area -> Road
            12: 1, # Rail Track
            13: 1, # Road
            14: 1, # Service Lane -> Road
            15: 6, # Sidewalk
            16: 2, # Bride -> static
            17: 2, # Building -> static
            18: 2, # Tunnel -> static
            19: 5, # Person
            20: 5, # Bycyclist -> Person
            21: 5, # Motorcyclist -> Person
            22: 5, # Other Rider -> Person
            23: 1, # Marking Crosswalk Zebra -> Marking
            24: 1, # Marking General -> Markings
            25: 3, # Mountain -> Terrain
            26: 3, # Sand -> Terrain
            27: 4, # Sky
            28: 3, # Snow -> Terrain
            29: 3, # Terrain
            30: 3, # Vegetation -> Terrain
            31: 3, # Water -> Terrain
            32: 2, # Banner -> static
            33: 2, # Bench -> static
            34: 2, # Bike Rack -> static
            35: 2, # BillBoard -> Static
            36: 2, # catch-basin -> static
            37: 2, # cctv-camera -> static
            38: 2, # Fire Hydrant -> static
            39: 2, # junction box -> static
            40: 2, # mailbox -> static
            41: 1, # manhole -> road
            42: 2, # phone booth -> static
            43: 1, # pothole -> road
            44: 8, # street-light -> pole
            45: 8, # pole
            46: 8, # sign frame -> pole
            47: 8, # pole 
            48: 9, # traffic light
            49: 10, # traffic sign back -> traffic sign
            50: 2, # traffic sign front -> static
            51: 2, # trash can -> static
            52: 11, # Bicycle -> Cyclish
            53: 0, # Boat -> Void
            54: 11, # Bus -> Big Vehicle
            55: 11, # Car
            56: 11, # Caravan -> Big Vehicle
            57: 11, # Motorcycle -> Cyclish
            58: 11, # on rails -> Big Vehicle
            59: 11, # other vehicle -> Car
            60: 11, # trailer -> Big Vehicle
            61: 11, # truck -> Big Vehicle
            62: 11, # wheeled slow -> Big Vehicle
            63: 0, # car-mount -> Void
            64: 0, # EgoCar -> Void
            65: 0, # Void
        }
        labels_lyft = [
            #       name                   id    trainId   category    catId     hasInstances   ignoreInEval   color
            Label(  'void'            ,  0 ,    4 , 'void'         , 4 , False , False , (  0,  0,  0) ),
            Label(  'road'            ,  1 ,    4 , 'flat'         , 4 , False , False , (128, 64,128) ),
            Label(  'static'          ,  2 ,  255 , 'construction' , 2 , False , False , ( 70, 70, 70) ),
            Label(  'vegetation/dirt' ,  3 ,  255 , 'flat'         , 2 , False , False , (107,142, 35) ),
            Label(  'sky'             ,  4 ,  255 , 'nature'       , 2 , False , False , ( 70,130,180) ),
            Label(  'person'          ,  5 ,  255 , 'human'        , 2 , False , False , (220, 20, 60) ),
            Label(  'sidewalk'        ,  6 ,  255 , 'flat'         , 2 , False , False , (244, 35,232) ),
            Label(  'Wall/Fence'      ,  7 ,    1 , 'construction' , 1 , False , False , (190,153,153) ),
            Label(  'pole'            ,  8 ,    1 , 'object'       , 1 , False , False , (153,153,153) ),
            Label(  'traffic-light'   ,  9 ,  255 , 'object'       , 1 , False , False , (250,170, 30) ),
            Label(  'traffic-sign'    , 10 ,  255 , 'object'       , 1 , False , False , (220,220,220) ),
            Label(  'car'             , 11 ,    2 , 'object'       , 1 , False , False , (  0,  0,142) ),
        ]
        self.name2label = {label.name: label for label in labels}
        # id to label object
        self.id2label = {label.id: label for label in labels}
        if self.number_of_classes == 15:
            self.name2label = {label.name: label for label in labels_thunderhill}
            # id to label object
            self.id2label = {label.id: label for label in labels_thunderhill}
        if self.number_of_classes == 12:
            self.name2label = {label.name: label for label in labels_lyft}
            # id to label object
            self.id2label = {label.id: label for label in labels_lyft}
        if self.number_of_classes == 0:
            self.trainQueuer = self.generatorAllClasses(True)
            time.sleep(1)
            self.valQueuer = self.generatorAllClasses(False)
            
        elif self.number_of_classes == 15:
            self.gen_function = self.generator_thunderhill
            self.colorDict = {label.id: label.color for label in labels_thunderhill}
        elif self.number_of_classes == 12:
            self.gen_function = self.generator_lyft
            self.colorDict = {label.id: label.color for label in labels_lyft}

        elif self.number_of_classes == 3:
            self.gen_function = self.generator_thunderhill_res
        else:
            self.gen_function = self.generator
        self.number_of_classes = len(labels) if self.number_of_classes == 0 else self.number_of_classes
        self.train_queue = None
        self.val_queue = None
        self.train_processes = []
        self.val_processes = []
        self.running = False
    
    def convert_target_to_image(self, target):
        image = np.zeros((target.shape[0], target.shape[1], 3))
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if i == 719 and j == target.shape[1] // 2:
                    print(target[i, j],
                          np.argmax(target[i, j]), self.id2label.get(np.argmax(target[i, j])).color)
                image[i, j, :] = self.id2label.get(target[i, j]).color[:]
        return image.astype('uint8')
        
    def generator(self, data, queue, train, **kwargs):
            index = np.random.randint(0, len(data))
            image = mpimg.imread(data.iloc[index]['input'])
            if image.shape[0] < 600 or image.shape[1] < 600:
                index = np.random.randint(0, len(data))
                image = mpimg.imread(data.iloc[index]['input'])
            target_name = data.iloc[index]['output']
            target = mpimg.imread(target_name)
            while image.shape[0] > 1300 or image.shape[1] > 1300:
                image = image[::2, ::2]
                target = target[::2, ::2]
            input_shape = image.shape
            image_arr = np.zeros((1, input_shape[0], input_shape[1], input_shape[2]))
            label_arr = np.zeros((1, input_shape[0], input_shape[1]))
            target = (target*255).astype('uint8')
            flip = np.random.rand()
            if flip > .5:
                image = mirrorImage(image)
                target = mirrorImage(target)
            if train:
                image, target = augmentImageAndLabel(image, target)
            image_arr[0] = image
            for i in range(len(target)):
                for j in range(target.shape[1]):
                    target[i, j] = self.dict[target[i, j]]
            label_arr[0] = target
            yield({'input_img': Variable(torch.from_numpy(image_arr)).float().permute(0, 3, 1, 2)},
                  {'output_img': Variable(torch.from_numpy(label_arr)).long()})
            
    def generatorAllClasses(self, train, **kwargs):
        max_q_size = 16
        maxproc = 8
        data = self.train_data if train else self.val_data
        try:
            queue = multiprocessing.Queue(maxsize=max_q_size)

            # define producer (putting items into queue)
            def producer():
                inputShape = (1024, 768, 3)
                np.random.seed(int(1e7*(time.time()-int(time.time()))))
                while(1):
                    #if train:
                    index = np.random.randint(0, len(data))
                    #else:
                    #    with lock:
                    #        index = start
                    #        start += 1
                    #        start = start%len(data)
                    image = mpimg.imread(data.iloc[index]['input'])
                    targetName = data.iloc[index]['output']
                    target = mpimg.imread(targetName)
                    while(image.shape[0] > 1300 or image.shape[1] > 1300): 
                        image = image[::2,::2]
                        target = target[::2,::2]
                    if (image.shape[0] < 600 or image.shape[1]< 600): continue
                    inputShape = image.shape
                    target = (target*255).astype('uint8')
                    flip = np.random.rand()
                    if flip>.5:
                        image = mirrorImage(image)
                        target = mirrorImage(target)
                    if train: image, target = augmentImageAndLabel(image, target)
                    image = image/255.
                    imageArr = image.reshape((1, inputShape[0], inputShape[1], inputShape[2]))
                    labelArr = target.reshape((1, inputShape[0], inputShape[1])).astype('uint8')
                    #print(index, data.iloc[index]['input'], image.shape, np.unique(target), train)
                    queue.put(({'input_img': imageArr}, 
                               {'output_img': labelArr}, data.iloc[index]['input']))

            processes = []

            def start_process():
                for val in range(len(processes), maxproc):
                    print("Spawn Thread %d" %val)
                    thread = multiprocessing.Process(target=producer)
                    time.sleep(0.01)
                    thread.start()
                    processes.append(thread)

            # run as consumer (read items from queue, in current thread)
            while True:
                processes = [p for p in self.processes if p.is_alive()]

                if len(processes) < maxproc:
                    start_process()

                yield queue.get()

        except:
            print("Finishing")
            for th in self.processes:
                th.terminate()
            queue.close()
            raise

    def __next__(self, train):
        if train:
            return self.train_queue.get()
        else:
            return self.val_queue.get()

    def launch_generators(self, max_q_size, max_proc):
        if self.train_queue is not None:
            return
        self.running = True
        self.train_queue = multiprocessing.Queue(maxsize=max_q_size)
        self.val_queue = multiprocessing.Queue(maxsize=max_q_size)
        # try:

        for train in [True, False]:
            processes = self.train_processes if train else self.val_processes
            data = self.train_data if train else self.val_data
            queue = self.train_queue if train else self.val_queue

            def start_process():
                np.random.seed(int(1e7*(time.time()-int(time.time()))))
                for val in range(len(processes), max_proc):
                    print("Spawn Thread %d" % val)
                    producer = functools.partial(
                        self.gen_function, data=data, queue=queue, train=train)
                    thread = multiprocessing.Process(target=producer, daemon=True)
                    time.sleep(0.01)
                    thread.start()
                    processes.append(thread)

            processes = [p for p in processes if p.is_alive()]

            if len(processes) < max_proc:
                start_process()

        # except:
        #     print("Finishing")
        #     for th in processes:
        #         th.terminate()
        #     queue.close()
        #     raise

    def stop_generators(self):
        self.running = False
        for train in [True, False]:
            processes = self.train_processes if train else self.val_processes
            for process in processes:
                process.terminate()
        
    def generator_thunderhill(self, data, queue, train, **kwargs):
        while 1:
            index = np.random.randint(0, len(data))
            image = mpimg.imread(data.iloc[index]['input'])
            while image.shape[0] < 600 or image.shape[1] < 600:
                index = np.random.randint(0, len(data))
                image = mpimg.imread(data.iloc[index]['input'])
            target_name = data.iloc[index]['output']
            target = mpimg.imread(target_name)
            while image.shape[0] > 1300 or image.shape[1] > 1300:
                image = image[::2, ::2]
                target = target[::2, ::2]
            input_shape = image.shape
            target = (target*255).astype('uint8')
            f = np.vectorize(lambda x: self.thunderhill_dict[x])
            # or use a different name if you want to keep the original f
            target = f(target)
            flip = np.random.rand()
            if flip > .5:
                image = mirrorImage(image)
                target = mirrorImage(target)
            if train:
                image, target = augmentImageAndLabel(image, target)
            image = image/255.
            image_arr = image.reshape((1, input_shape[0], input_shape[1], input_shape[2]))
            label_arr = target.reshape((1, input_shape[0], input_shape[1])).astype('uint8')
            # print(index, data.iloc[index]['input'], image.shape, np.unique(target), train)
            queue.put(({'input_img': image_arr},
                       {'output_img': label_arr}, data.iloc[index]['input']))


    def generator_lyft(self, data, queue, train, **kwargs):
        while 1:
            index = np.random.randint(0, len(data))
            image = mpimg.imread(data.iloc[index]['input'])
            while image.shape[0] < 600 or image.shape[1] < 600:
                index = np.random.randint(0, len(data))
                image = mpimg.imread(data.iloc[index]['input'])
            target_name = data.iloc[index]['output']
            target = mpimg.imread(target_name)
            while image.shape[0] > 1300 or image.shape[1] > 1300:
                image = image[::2, ::2]
                target = target[::2, ::2]
            if image.shape[0] < 600 or image.shape[1] < 600: continue
            input_shape = image.shape
            target = (target*255).astype('uint8')
            w, h = target.shape
            f = np.vectorize(lambda x: self.lyft_dict[x])  # or use a different name if you want to keep the original f
            target = f(target)
            target = target.reshape(-1)
            target = np.eye(self.number_of_classes)[target]
            target = target.reshape(w, h, self.number_of_classes)
            flip = np.random.rand()
            if flip > .5:
                image = mirrorImage(image)
                target = mirrorImage(target)
            if train:
                image, target = augmentImageAndLabel(image, target)
            image = image/255.
            image_arr = image.reshape((1, input_shape[0], input_shape[1], input_shape[2]))
            label_arr = target.reshape((1, input_shape[0], input_shape[1], self.number_of_classes))
            #print(index, data.iloc[index]['input'], image.shape, np.unique(target), train)
            queue.put(({'input_img': image_arr},
                       {'output_img': label_arr}, data.iloc[index]['input']))

        
            
    def generatorThunderhillRes(self, train, **kwargs):
        max_q_size = 16
        maxproc = 8
        data = self.train_data if train else self.val_data
        try:
            queue = multiprocessing.Queue(maxsize=max_q_size)

            # define producer (putting items into queue)
            def producer():
                np.random.seed(int(1e7*(time.time()-int(time.time()))))
                while(1):
                    #if train:
                    index = np.random.randint(0, len(data))
                    #else:
                    #    with lock:
                    #        index = start
                    #        start += 1
                    #        start = start%len(data)
                    image = mpimg.imread(data.iloc[index]['input'])
                    targetName = data.iloc[index]['output']
                    target = mpimg.imread(targetName)
                    while(image.shape[0] > 1300 or image.shape[1] > 1300): 
                        image = image[::2,::2]
                        target = target[::2,::2]
                    
                    if (image.shape[0] < 600 or image.shape[1]< 600): continue
                    inputShape = image.shape
                    
                    target = (target*255).astype('uint8')
                    f = np.vectorize(lambda x: self.thunderhillDictRes[x])  # or use a different name if you want to keep the original f
                    target = f(target) 
                    flip = np.random.rand()
                    if flip>.5:
                        image = mirrorImage(image)
                        target = mirrorImage(target)
                    if train: image, target = augmentImageAndLabel(image, target)
                    image = image/255.
                    imageArr = image.reshape((1, inputShape[0], inputShape[1], inputShape[2]))
                    labelArr = target.reshape((1, inputShape[0], inputShape[1])).astype('uint8')
                    #print(index, data.iloc[index]['input'], image.shape, np.unique(target), train)
                    queue.put(({'input_img': imageArr}, 
                      {'output_img': labelArr}, data.iloc[index]['input']))

            processes = []

            def start_process():
                for val in range(len(processes), maxproc):
                    print("Spawn Thread %d" %val)
                    thread = multiprocessing.Process(target=producer)
                    time.sleep(0.01)
                    thread.start()
                    processes.append(thread)

            # run as consumer (read items from queue, in current thread)
            while True:
                processes = [p for p in processes if p.is_alive()]

                if len(processes) < maxproc:
                    start_process()

                yield queue.get()

        except:
            print("Finishing")
            for th in processes:
                th.terminate()
            queue.close()
            raise

    @staticmethod
    def get_sub_train_distribution(df, number_of_classes, dictionary):
        arr = np.zeros(number_of_classes)
        f = np.vectorize(lambda x: dictionary[x])
        for handle in df['output']:
            target = mpimg.imread(handle)[::4, ::4]
            target = (target*255).astype('uint8')
            if number_of_classes < 30:
                target = f(target)
            print(np.unique(target))
            for val in target.ravel():
                arr[int(val)] += 1
        return arr
    
    def get_train_distribution(self):
        sub_dfs = []
        # Zähle die Prozessoren
        cores = multiprocessing.cpu_count() 
        # Teile den DataFrame in cores gleichgrosse Teile
        for i in range(cores):
            sub_dfs.append(self.train_data.iloc[int(i/cores*len(self.train_data)):int((i+1)/cores*len(self.train_data)), :])

        print("Build SubDfs")
        # Öffne einen Pool mit ensprechend vielen Prozessen 
        pool = Pool(processes=cores)
        # Wende die Funktion an
        func = functools.partial(Mapillary.get_sub_train_distribution,
                                 number_of_classes=self.number_of_classes,
                                 dictionary=self.thunderhill_dict)
        arrs = pool.map(func, sub_dfs)
        pool.close()
        arr = np.zeros(self.number_of_classes)
        for i in range(len(arrs)):
            print(arrs[i])
            arr += arrs[i]
            
        return arr
    
    def setWeights(self, weights):
        self.weights = weights
    
class CamVid(Dataset):
    
    def __init__(self, path, **kwargs):
        super(CamVid, self).__init__(path=path)
        imageList = glob.glob(self.path + '/701_StillsRaw_full/*.png')
        data = pd.DataFrame(imageList, columns = ['input'])
        data['output'] = data['input'].apply(lambda x: self.path + '/LabeledApproved_full' + x[x.rfind('/'):-4] + '_L.png')
        data['outputPickled'] = data['input'].apply(lambda x: self.path + '/LabeledApproved_full' + x[x.rfind('/'):-4] + '_L.pickle')
        self.train_data, self.val_data = train_test_split(data, test_size = .1, random_state = 42)
        self.number_of_classes = kwargs['number_of_classes']
        self.weights = None
        self.treeDict = {
        0: {
            0: {
                0: 0, # Void
                64: 1, #TrafficCone
                192: 2 #Sidewalk
                },
            64: {64: 3}, #TrafficLight
            128: {
                64: 4, #Bridge 
                192: 5 #Bicyclist
                }
            },
        64: {
            0: {
                64: 6, #Tunnel
                128: 7, #Car
                192: 8 #CartLuggagePram
                },
            64: {
                 0: 9, # Pedestrian
                 128: 10 #Fence
                },
            128: {
                64: 11, #Animal
                192: 12 #SUVPickupTruck
                },
            192: {
                0: 13, # Building/Wall
                128: 14 #ParkingBlock
                }
            },
        128: {
            0: {
                0:13, # Building/Wall
                192: 16 #LaneMkgsDriv
                },
            64: {
                64: 17, #OtherMoving
                128: 18 #Road
                },
            128: {
                0: 19, #Tree
                64: 20, #Misc_Text
                128: 21, #Sky
                192: 22 #RoadShoulder
                }
            },
        192: {
            0: {
                64: 23, #LaneMkgsNonDriv
                128: 24, #Archway
                192: 25 #MotorcycleScooter
                },
            64: {
                128: 26, #Train
                },
            128: {
                64: 27, #Child
                128: 28, #SignSymbol
                192: 29, #Truck_Bus
                },
            192: {
                0: 30, #VegetationMisc
                128: 31 #Column_Pole
                }
            }
        }
        
        self.treeDictThunderhill = {
        0: {
            0: {
                0: 0, # Void
                64: 8, #TrafficCone -> Pole
                192: 6 #Sidewalk
                },
            64: {64: 9}, #TrafficLight
            128: {
                64: 2, #Bridge ->Static
                192: 13 #Bicyclist -> Cyclish
                }
            },
        64: {
            0: {
                64: 2, #Tunnel
                128: 11, #Car
                192: 11 #CartLuggagePram
                },
            64: {
                 0: 5, #Pedestrian
                 128: 7 #Fence
                },
            128: {
                64: 0, #Animal
                192: 12 #SUVPickupTruck -> Big Vehicle
                },
            192: {
                0: 7, # Building/Wall
                128: 1 #ParkingBlock
                }
            },
        128: {
            0: {
                0:2, # Building/Wall -> Static
                192: 14 #LaneMkgsDriv
                },
            64: {
                64: 13, #OtherMoving
                128: 1 #Road
                },
            128: {
                0: 3, #Tree -> Terrain
                64: 0, #Misc_Text -> Void
                128: 4, #Sky
                192: 1 #RoadShoulder
                }
            },
        192: {
            0: {
                64: 14, #LaneMkgsNonDriv
                128: 2, #Archway -> Static
                192: 13 #MotorcycleScooter -> Cyclish
                },
            64: {
                128: 12, #Train-> Big Vehicle
                },
            128: {
                64: 5, #Pedestrian
                128: 10, #SignSymbol -> Traffic Sign
                192: 12, #Truck_Bus -> Big Vehicle
                },
            192: {
                0: 3, #VegetationMisc -> Terrain
                128: 9 #Column_Pole -> Pole
                }
            }
        }
        if self.number_of_classes == 0:
            self.trainQueuer = GeneratorEnqueuer(self.generator(1, True), False)
            self.trainQueuer.start(workers=1, max_queue_size=10)
            self.valQueuer = GeneratorEnqueuer(self.generator(1, False), False)
            self.valQueuer.start(workers=1, max_queue_size=10)
        elif self.number_of_classes == 16:
            self.trainQueuer = GeneratorEnqueuer(self.generatorThunderhill(True), False)
            self.trainQueuer.start(workers=1, max_queue_size=10)
            self.valQueuer = GeneratorEnqueuer(self.generatorThunderhill(False), False)
            self.valQueuer.start(workers=1, max_queue_size=10)
        
    def generator(self, batch_size, train, **kwargs):
        start=0
        inputShape = mpimg.imread(self.train_data.iloc[0]['input']).shape
        data = self.train_data if train else self.val_data
        while(1):
            imageArr = np.zeros((batch_size, inputShape[0], inputShape[1], inputShape[2]))
            labelArr = np.zeros((batch_size, inputShape[0] * inputShape[1]))
            #weights = np.ones((batchSize, self.number_of_classes))
            weights = np.zeros((batch_size, inputShape[0] * inputShape[1]))
            if train:
                indices = np.random.randint(0, len(data), batch_size)
            else:
                indices = np.zeros(batch_size).astype(int)
                for i in range(batch_size):
                    indices[i] = start % len(data)
                    start += 1
                start = start%len(data)
            for i, index in zip(range(len(indices)), indices):
                #if(not train): print(imgList[index])
                image = mpimg.imread(data.iloc[index]['input'])
                pickleName = data.iloc[index]['outputPickled']
                targetName = data.iloc[index]['output']
                #print(targetName)
                try:
                    with open(pickleName, 'rb') as handle:
                        target = pickle.load(handle)
                        
                except:
                    target = mpimg.imread(targetName)
                    print(targetName)
                    target = (target*255).astype('uint8')
                    
                    #plt.imshow(target)
                    #plt.show()
                    target = self.convertTarget(target, self.treeDict)
                    with open(pickleName, 'wb') as handle:
                        pickle.dump(target, handle, protocol=pickle.HIGHEST_PROTOCOL)
                flip = np.random.rand()
                if flip>.5:
                    image = mirrorImage(image)
                    target = mirrorImage(target)
                if train: image, target = augmentImageAndLabel(image, target)
                imageArr[i] = image
                target = np.reshape(target, (inputShape[0]*inputShape[1]))
                """if(not np.any(self.weights == None)):
                    weights[i] = self.weights
                    weights *= 4 # Kind of normalisation factor
                """
                if(np.any(self.weights == None)):
                    for j in range(len(target)): 
                        weights[i,j] = 1
                else:
                    for j in range(len(target)): 
                        weights[i,j] = self.weights[int(target[j])]
                    weights *= 8 # Kind of normalisation factor
                labelArr[i] = target
            yield({'input_img': imageArr}, {'output_img': labelArr}, weights)
            
    def generatorThunderhill(self, train, **kwargs):
        max_q_size = 16
        maxproc = 8
        data = self.train_data if train else self.val_data
        inputShape = mpimg.imread(self.train_data.iloc[0]['input']).shape
        try:
            queue = multiprocessing.Queue(maxsize=max_q_size)

            # define producer (putting items into queue)
            def producer():
                np.random.seed(int(1e7*(time.time()-int(time.time()))))
                while(1):
                    index = np.random.randint(0, len(data))
                    image = mpimg.imread(data.iloc[index]['input'])
                    image = (image*255).astype('uint8')
                    pickleName = data.iloc[index]['outputPickled']
                    targetName = data.iloc[index]['output']
                    #print(targetName)
                    try:
                        with open(pickleName, 'rb') as handle:
                            target = pickle.load(handle)
                                
                    except:
                        print('Start converting traget')
                        target = mpimg.imread(targetName)
                        print(targetName)
                        target = (target*255).astype('uint8')
                            
                        #plt.imshow(target)
                        #plt.show()
                        target = self.convertTarget(target, self.treeDictThunderhill)
                        with open(pickleName, 'wb') as handle:
                            pickle.dump(target, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        print('Converted Target')
                    flip = np.random.rand()
                    if flip>.5:
                        image = mirrorImage(image)
                        target = mirrorImage(target)
                    if train: image, target = augmentImageAndLabel(image, target)
                    image = image/255.
                    imageArr = image.reshape((1, inputShape[0], inputShape[1], inputShape[2]))
                    labelArr = target.reshape((1, inputShape[0], inputShape[1])).astype('uint8')
                    yield({'input_img': imageArr}, {'output_img': labelArr}, data.iloc[index]['input'])

            processes = []

            def start_process():
                for val in range(len(processes), maxproc):
                    print("Spawn Thread %d" %val)
                    thread = multiprocessing.Process(target=producer)
                    time.sleep(0.01)
                    thread.start()
                    processes.append(thread)

            # run as consumer (read items from queue, in current thread)
            while True:
                processes = [p for p in processes if p.is_alive()]

                if len(processes) < maxproc:
                    start_process()

                yield queue.get()

        except:
            print("Finishing")
            for th in processes:
                th.terminate()
            queue.close()
            raise
            
            
    def convertTarget(self, image, convDict):
        target = np.zeros((image.shape[0], image.shape[1], 1))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                label = convDict[image[i,j,0]][image[i,j,1]][image[i,j,2]]
                #if (label == None):
                #    print(image[i,j])
                target[i,j,0] = label
        return target
    
    
    def getTrainDistribution(self):
        arr = np.zeros(self.number_of_classes)
        for handle in self.train_data['outputPickled']:
            with open(handle, 'rb') as handle:
                target = pickle.load(handle)
            for val in target.ravel():
                arr[int(val)] += 1
        return arr
    
    def setWeights(self, weights):
        self.weights = weights
    
class Kitti(Dataset):
    
    def __init__(self, path, samplesPerBatch, **kwargs):
        super(Kitti, self).__init__(path=path)
        imageList = glob.glob(self.path + '/lefttest/*.png')
        data = pd.DataFrame(imageList, columns = ['input'])
        data['output'] = data['input'].apply(lambda x: self.path + '/gttest' + x[x.rfind('/'):])
        self.train_data, self.val_data = train_test_split(data, test_size = .1, random_state = 42)
        self.number_of_classes = kwargs['number_of_classes']
        self.weights = None
        
        self.dict = {
            0: 0, # Void
            1: 21, # Sky
            2: 18, #Road
            3: 2, #Sidewalk
            4: 22, # Bicycle Lane -> Road Shoulder
            5: 16, #Lane Markings -> Lane Markings Drive
            6: 15, # Railway
            7: 30, # Grass -> Misc Vegetation
            8: 19, # Tree
            9: 30, # Misc Vegetation
            10: 13, # Building/Wall
            11: 4, # Bridge
            12: 31, #Pole -> Column_Pole
            13: 28, #Panel ->SignSymbol
            14: 3, #TrafficLight
            15: 10, #Fence
            16: 13, #Other Infrastructure -> Building/Wall
            17: 7, #Car
            18: 29, #Truck -> Truck_Bus
            19: 29, #Bus -> Truck_Bus
            20: 26, #Train_Tramway -> Train
            21: 9, #Adult -> Pedestrian
            22: 27, #Child
            23: 5, # Cyclist -> Bicyclist
            24: 5, # Bicycle -> Bicyclist
            25: 25, # Motorcyclist
            26: 25, # Motorcycle -> Motorcyclist
            27: 11, #Animal
            28: 17 # Other Moving
            }
        self.trainQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, True), False)
        self.trainQueuer.start(workers=1, max_queue_size=10)
        self.valQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, False), False)
        self.valQueuer.start(workers=1, max_queue_size=10)
        
    def generator(self, batchSize, train, **kwargs):
        start=0
        inputShape = mpimg.imread(self.train_data.iloc[0]['input']).shape
        data = self.train_data if train else self.val_data
        while(1):
            imageArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
            labelArr = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            #weights = np.zeros((batchSize, self.number_of_classes))
            weights = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            if train:
                indices = np.random.randint(0, len(data), batchSize)
            else:
                indices = np.zeros(batchSize).astype(int)
                for i in range(batchSize):
                    indices[i] = start%len(data)
                    start += 1
                start = start%len(data)
            for i, index in zip(range(len(indices)), indices):
                #if(not train): print(imgList[index])
                image = mpimg.imread(data.iloc[index]['input'])
                #print(data.iloc[index]['input'], image.shape)
                #if(image.shape[0] == 1231 and image.s 1242):
                #    image = np.lib.pad(image, ((9,9),(3,2),(0,0)), mode = 'constant', constant_values=(0))
                if(image.shape[1] == 1224):
                    image = np.lib.pad(image, ((3,2),(9,9),(0,0)), mode = 'constant', constant_values=(0))
                elif(image.shape[1] == 1226):
                    image = np.lib.pad(image, ((3,2),(8,8),(0,0)), mode = 'constant', constant_values=(0))
                elif(image.shape[1] == 1238):
                    image = np.lib.pad(image, ((1,0),(2,2),(0,0)), mode = 'constant', constant_values=(0))
                targetName = data.iloc[index]['output']
                #print(targetName)
                target = mpimg.imread(targetName)
                if(target.shape[1] == 1224):
                    target = np.lib.pad(target, ((3,2),(9,9)), mode = 'constant', constant_values=(0))
                elif(target.shape[1] == 1226):
                    target = np.lib.pad(target, ((3,2),(8,8)), mode = 'constant', constant_values=(0))
                elif(target.shape[1] == 1238):
                    target = np.lib.pad(target, ((1,0),(2,2)), mode = 'constant', constant_values=(0))
                target = (target*255).astype('uint8')
                    
                    #plt.imshow(target)
                    #plt.show()
                target = self.convertTarget(target)
                flip = np.random.rand()
                if flip>.5:
                    image = mirrorImage(image)
                    target = mirrorImage(target)
                if train: image, target = augmentImageAndLabel(image, target)
                imageArr[i] = image
                target = np.reshape(target, (inputShape[0]*inputShape[1]))
                """if(not np.any(self.weights == None)):
                    weights[i] = self.weights
                    weights *= 4 # Kind of normalisation factor
                """
                if(np.any(self.weights == None)):
                    for j in range(len(target)): 
                        weights[i,j] = 1
                else:
                    for j in range(len(target)): 
                        weights[i,j] = self.weights[int(target[j])]
                    weights *= 8 # Kind of normalisation factor
                labelArr[i] = target
            yield({'input_img': imageArr}, {'output_img': labelArr}, weights)
            
            
    def convertTarget(self, image):
        #print(image.shape)
        image = np.reshape(image, (int(image.shape[0]), int(image.shape[1]) , 1))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j,0] = self.dict[image[i,j,0]]
        return image
    
    def getTrainDistribution(self):
        arr = np.zeros(self.number_of_classes)
        for handle in self.train_data['output']:
            target = mpimg.imread(handle)
            target = (target*255).astype('uint8')
            target = self.convertTarget(target)
            for val in target.ravel():
                arr[int(val)] += 1
        return arr
    
    def setWeights(self, weights):
        self.weights = weights
        
    
        
class KittiStreet(Dataset):
    
    def __init__(self, path, samplesPerBatch, **kwargs):
        super(KittiStreet, self).__init__(path=path)
        imageList = glob.glob(self.path + '/training/image_2/*.png')
        data = pd.DataFrame(imageList, columns = ['input'])
        data['output'] = data['input'].apply(lambda x: self.path + '/training/gt_image_2/um_lane' + x[x.rfind('_'):])
        self.train_data, self.val_data = train_test_split(data, test_size = .1, random_state = 42)
        self.number_of_classes = 2
        self.weights = None
        self.trainQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, True), False)
        self.trainQueuer.start(workers=1, max_queue_size=10)
        self.valQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, False), False)
        self.valQueuer.start(workers=1, max_queue_size=10)
        imageList = glob.glob(self.path + '/testing/image_2/*.png')
        self.testData = pd.DataFrame(imageList, columns = ['input'])
         
        
    def generator(self, batchSize, train, **kwargs):
        start=0
        inputShape = mpimg.imread(self.train_data.iloc[0]['input']).shape
        data = self.train_data if train else self.val_data
        while(1):
            imageArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
            labelArr = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            #weights = np.zeros((batchSize, self.number_of_classes))
            weights = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            if train:
                indices = np.random.randint(0, len(data), batchSize)
            else:
                indices = np.zeros(batchSize).astype(int)
                for i in range(batchSize):
                    indices[i] = start%len(data)
                    start += 1
                start = start%len(data)
            for i, index in zip(range(len(indices)), indices):
                #if(not train): print(imgList[index])
                print(data.iloc[index]['input'])
                image = mpimg.imread(data.iloc[index]['input'])
                #print(data.iloc[index]['input'], image.shape)
                #if(image.shape[0] == 1231 and image.s 1242):
                #    image = np.lib.pad(image, ((9,9),(3,2),(0,0)), mode = 'constant', constant_values=(0))
                if(image.shape[1] == 1224):
                    image = np.lib.pad(image, ((3,2),(9,9),(0,0)), mode = 'constant', constant_values=(0))
                elif(image.shape[1] == 1226):
                    image = np.lib.pad(image, ((3,2),(8,8),(0,0)), mode = 'constant', constant_values=(0))
                elif(image.shape[1] == 1238):
                    image = np.lib.pad(image, ((1,0),(2,2),(0,0)), mode = 'constant', constant_values=(0))
                targetName = data.iloc[index]['output']
                #print(targetName)
                target = mpimg.imread(targetName)
                if(target.shape[1] == 1224):
                    target = np.lib.pad(target, ((3,2),(9,9)), mode = 'constant', constant_values=(0))
                elif(target.shape[1] == 1226):
                    target = np.lib.pad(target, ((3,2),(8,8)), mode = 'constant', constant_values=(0))
                elif(target.shape[1] == 1238):
                    target = np.lib.pad(target, ((1,0),(2,2)), mode = 'constant', constant_values=(0))
                target = (target*255).astype('uint8')
                    
                    #plt.imshow(target)
                    #plt.show()
                target = self.convertTarget(target)
                flip = np.random.rand()
                if flip>.5:
                    image = mirrorImage(image)
                    target = mirrorImage(target)
                if train: image, target = augmentImageAndLabel(image, target)
                imageArr[i] = image
                target = np.reshape(target, (inputShape[0]*inputShape[1]))
                """if(not np.any(self.weights == None)):
                    weights[i] = self.weights
                    weights *= 4 # Kind of normalisation factor
                """
                if(np.any(self.weights == None)):
                    for j in range(len(target)): 
                        weights[i,j] = 1
                else:
                    for j in range(len(target)): 
                        weights[i,j] = self.weights[int(target[j])]
                    weights *= 8 # Kind of normalisation factor
                labelArr[i] = target
            yield({'input_img': imageArr}, {'output_img': labelArr}, weights)
            
            
    def convertTarget(self, image):
        #print(image.shape)
        target = np.zeros((int(image.shape[0]),int(image.shape[1])))
        image = np.reshape(image, (int(image.shape[0]), int(image.shape[1]) , 3))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                target[i,j] = 1 if np.any(image[i,j,0]) else 0
        return target
    
    def getTrainDistribution(self):
        arr = np.zeros(self.number_of_classes)
        for handle in self.train_data['output']:
            target = mpimg.imread(handle)
            target = (target*255).astype('uint8')
            target = self.convertTarget(target)
            for val in target.ravel():
                arr[int(val)] += 1
        return arr
    
    def setWeights(self, weights):
        self.weights = weights
        
    
        
class NYUDataset(Dataset):
    
    def __init__(self, path, samplesPerBatch, **kwargs):
        super(NYUDataset, self).__init__(path=path)
        imageList = glob.glob(self.path + '/Data/training/*/*_colors.png')
        assert(len(imageList))
        self.train_data = pd.DataFrame(imageList, columns = ['input'])
        self.train_data['output'] = self.train_data['input'].apply(lambda x: x[:x.rfind('_')]+'_ground_truth.png')
        imageList = glob.glob(self.path + '/Data/testing/*/*_colors.png')
        self.val_data = pd.DataFrame(imageList, columns = ['input'])
        self.val_data['output'] = self.val_data['input'].apply(lambda x: x[:x.rfind('_')]+'_ground_truth.png')
        self.train_data.append(self.val_data[::2])
        self.val_data = self.val_data[1::2]
        self.number_of_classes = kwargs['number_of_classes']
        self.weights = None
        Label = namedtuple( 'Label' , [

            'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                            # We use them to uniquely name a class
        
            'id'          , # An integer ID that is associated with this label.
                            # The IDs are used to represent the label in ground truth images
                            # An ID of -1 means that this label does not have an ID and thus
                            # is ignored when creating ground truth images (e.g. license plate).
                            # Do not modify these IDs, since exactly these IDs are expected by the
                            # evaluation server.
        
            'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                            # ground truth images with train IDs, using the tools provided in the
                            # 'preparation' folder. However, make sure to validate or submit results
                            # to our evaluation server using the regular IDs above!
                            # For trainIds, multiple labels might have the same ID. Then, these labels
                            # are mapped to the same class in the ground truth images. For the inverse
                            # mapping, we use the label that is defined first in the list below.
                            # For example, mapping all void-type classes to the same ID in training,
                            # might make sense for some approaches.
                            # Max value is 255!
        
            'category'    , # The name of the category that this label belongs to
        
            'categoryId'  , # The ID of this category. Used to create ground truth images
                            # on category level.
        
            'hasInstances', # Whether this label distinguishes between single instances or not
        
            'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                            # during evaluations or not
        
            'color'       , # The color of this label
        ] )
        
        labels = [
            #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
            Label(  'unlabeled'   ,  0 ,      255 , 'void'            , 0       , False        , False , (  0,  0,  0) ),
            Label(  'outdoor'     ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0, 43, 54) ),
            Label(  'floor'       ,  2 ,      255 , 'void'            , 0       , False        , True         , (111,  74,  0) ),
            Label(  'wall'        ,  3 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
            Label(  'ceiling'     ,  4 ,      255 , 'void'            , 0       , False        , True         , (128, 64,128) ),
            Label(  'window'      ,  5 ,      255 , 'void'            , 0       , False        , True         , (244, 35,232) ),
            Label(  'structural'  ,  6 ,      255 , 'void'            , 0       , False        , True         , (250,170,160) ),
            Label(  'food'        ,  7 ,        0 , 'flat'            , 1       , False        , False        , (230,150,140) ),
            Label(  'indoor'      ,  8 ,        1 , 'flat'            , 1       , False        , False        , ( 70, 70, 70) ),
            Label(  'prop'        ,  9 ,      255 , 'flat'            , 1       , False        , True         , (102,102,156) ),
            Label(  'furniture'   , 10 ,      255 , 'flat'            , 1       , False        , True         , (190,153,153) ),
            Label(  'appliance'   , 11 ,        2 , 'construction'    , 2       , False        , False        , (180,165,180) ),
            Label(  'sports'      , 12 ,        3 , 'construction'    , 2       , False        , False        , (150,100,100) ),
            Label(  'accesory'    , 13 ,        4 , 'construction'    , 2       , False        , False        , (150,120, 90) ),
            Label(  'animal'      , 14 ,      255 , 'construction'    , 2       , False        , True         , (153,153,153) ),
            Label(  'vehicle'     , 15 ,      255 , 'construction'    , 2       , False        , True         , (250,170, 30) ),
            Label(  'person'      , 16 ,      255 , 'construction'    , 2       , False        , True         , (220,220,  0) ),
            Label(  'electronic'  , 17 ,        5 , 'object'          , 3       , False        , False        , (107,142, 35) ),
            Label(  'kitchen'     , 18 ,      255 , 'object'          , 3       , False        , True         , (152,251,152) ),
            Label(  'water'       , 19 ,        6 , 'object'          , 3       , False        , False        , ( 70,130,180) ),
            Label(  'ground'      , 20 ,        7 , 'object'          , 3       , False        , False        , (220, 20, 60) ),
            Label(  'solid'       , 21 ,        8 , 'nature'          , 4       , False        , False        , (255,  0,  0) ),
            Label(  'sky'         , 22 ,        9 , 'nature'          , 4       , False        , False        , (  0,  0,142) ),
            Label(  'plant'       , 23 ,       10 , 'sky'             , 5       , False        , False        , (  0,  0, 70) ),
            Label(  'builiding'   , 24 ,       11 , 'human'           , 6       , True         , False        , (  0, 60,100) ),
            Label(  'textile'     , 25 ,       12 , 'human'           , 6       , True         , False        , (  0,  0, 90) ),
            Label(  'rawmaterial' , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,110) ),
        ]
        self.name2label      = { label.name    : label for label in labels           }
        # id to label object
        self.id2label = { label.id : label for label in labels }
        if self.number_of_classes == 0:
            self.trainQueuer = GeneratorEnqueuer(self.generatorAllClasses(samplesPerBatch, True), False)
            self.trainQueuer.start(workers=1, max_queue_size=10)
            self.valQueuer = GeneratorEnqueuer(self.generatorAllClasses(samplesPerBatch, False), False)
            self.valQueuer.start(workers=1, max_queue_size=10)
        else:
            self.trainQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, True), False)
            self.trainQueuer.start(workers=1, max_queue_size=10)
            self.valQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, False), False)
            self.valQueuer.start(workers=1, max_queue_size=10)
        self.number_of_classes = len(labels) if self.number_of_classes == 0 else self.number_of_classes
    
    @staticmethod
    def getCityName(path):
        return path.split('/')[-2]
    
    def convertTargetToImage(self, target):
        image = np.zeros((target.shape[0], target.shape[1], 3))
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                #if (label == None):
                #    print(image[i,j])
                if(i == 719 and j == target.shape[1]//2): print(target[i,j],
                         np.argmax(target[i,j]), self.id2label.get(np.argmax(target[i,j])).color)
                image[i,j, :] = self.id2label.get(target[i,j]).color[:]
        return image.astype('uint8')
    
    @staticmethod
    def getImageName(path):
        image = path.split('/')[-1]
        return image[:image.rfind('_')]
        
    def generator(self, batchSize, train, **kwargs):
        start=0
        inputShape = mpimg.imread(self.train_data.iloc[0]['input']).shape
        data = self.train_data if train else self.val_data
        while(1):
            imageArr = np.zeros((batchSize, inputShape[2], inputShape[0], inputShape[1]))
            labelArr = np.zeros((batchSize, inputShape[0], inputShape[1]))
            if train:
                indices = np.random.randint(0, len(data), batchSize)
            else:
                indices = np.zeros(batchSize).astype(int)
                for i in range(batchSize):
                    indices[i] = start%len(data)
                    start += 1
                start = start%len(data)
            for i, index in zip(range(len(indices)), indices):
                #if(not train): print(imgList[index])
                image = mpimg.imread(data.iloc[index]['input'])
                targetName = data.iloc[index]['output']
                #print(targetName)
                target = mpimg.imread(targetName)
                print(targetName)
                target = (target*255).astype('uint8')
                
                imageArr[i] = np.transpose(image, [2, 0, 1])
                labelArr[i] = target
            
            yield({'input_img': Variable(torch.from_numpy(imageArr)).float().cuda()}, 
                  {'output_img': Variable(torch.from_numpy(labelArr)).long().cuda()})
            
    def generatorAllClasses(self, batchSize, train, **kwargs):
        start=0
        print(self.train_data.iloc[0]['input'])
        inputShape = mpimg.imread(self.train_data.iloc[0]['input']).shape
        data = self.train_data if train else self.val_data
        while(1):
            imageArr = np.zeros((batchSize, inputShape[2], inputShape[0], inputShape[1]))
            labelArr = np.zeros((batchSize, inputShape[0], inputShape[1]))
            if train:
                indices = np.random.randint(0, len(data), batchSize)
            else:
                indices = np.zeros(batchSize).astype(int)
                for i in range(batchSize):
                    indices[i] = start%len(data)
                    start += 1
                start = start%len(data)
            for i, index in zip(range(len(indices)), indices):
                #if(not train): print(imgList[index])
                image = mpimg.imread(data.iloc[index]['input'])
                
                targetName = data.iloc[index]['output']
                #print(targetName)
                target = mpimg.imread(targetName)
                target = (target*255).astype('uint8')
                #target[target==-1] = 0
                #plt.imshow(target)
                #plt.show()
                imageArr[i] = np.transpose(image, [2, 0, 1])
                labelArr[i] = target
            
            yield({'input_img': Variable(torch.from_numpy(imageArr)).float().cuda()}, 
                  {'output_img': Variable(torch.from_numpy(labelArr)).long().cuda()})
            
    @staticmethod
    def convertTarget(image, dictonary):
        #print(image.shape)
        image = np.reshape(image, (int(image.shape[0]), int(image.shape[1]) , 1))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j,0] = dictonary[image[i,j,0]]
        return image
    
    @staticmethod
    def getSubTrainDistribution(df, number_of_classes):
        arr = np.zeros(number_of_classes)
        for handle in df['output']:
            target = mpimg.imread(handle)
            target = (target*255).astype('uint8')
            target[target == -1] = 0
            for val in target.ravel():
                arr[int(val)] += 1
        return arr
    
    def getTrainDistribution(self):
        subDfs = []
        # Zähle die Prozessoren
        cores = multiprocessing.cpu_count() 
        # Teile den DataFrame in cores gleichgrosse Teile
        for i in range(cores):
            subDfs.append(self.train_data.iloc[int(i/cores*len(self.train_data)):int((i+1)/cores*len(self.train_data)), :])
        print("Build SubDfs")
        # Öffne einen Pool mit ensprechend vielen Prozessen 
        pool = Pool(processes=cores)
        # Wende die Funktion an
        func = functools.partial(NYUDataset.getSubTrainDistribution, number_of_classes=self.number_of_classes)
        arrs = pool.map(func, subDfs)
        pool.close()
        arr = np.zeros(self.number_of_classes)
        for i in range(len(arrs)):
            arr += arrs[i]
            
        return arr
    
    
    def setWeights(self, weights):
        self.weights = weights
        