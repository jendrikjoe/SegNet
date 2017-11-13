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
from Preprocess.Preprocess import mirrorImage, augmentImageAndLabel
from keras.utils.np_utils import to_categorical
from keras.layers import Input
import threading
from multiprocessing import Queue
from keras.engine.training import GeneratorEnqueuer
from collections import namedtuple  
import multiprocessing
from multiprocessing import Pool
from matplotlib import pyplot as plt
import functools
class Dataset(ABC):
    '''
    classdocs
    '''


    def __init__(self, path, **kwargs):
        '''
        Constructor
        '''
        self.path = path
        self.trainData = None
        self.valData = None
        
    @abstractmethod
    def generator(self, batchSize, train, **kwargs):
        pass
    
    def buildInputLayer(self):
        inputShape = mpimg.imread(self.trainData.iloc[0]['input']).shape
        return Input(shape=(inputShape[0], inputShape[1], inputShape[2]), name='inputImg')
    
    @abstractmethod
    def getTrainDistribution(self):
        pass
    
    @abstractmethod
    def setWeights(self):
        pass
    
class CityScapes(Dataset):
    
    def __init__(self, path, samplesPerBatch, **kwargs):
        super(CityScapes, self).__init__(path=path)
        imageList = glob.glob(self.path + '/leftImg8bit/train/*/*.png')
        imageList = [path.replace('\\', '/') for path in imageList]
        self.trainData = pd.DataFrame(imageList, columns = ['input'])
        self.trainData['output'] = self.trainData['input'].apply(lambda x: self.path + '/gtFine_trainvaltest/gtFine/train/'+
                                             self.getCityName(x)+'/'+self.getImageName(x)+'_gtFine_labelIds.png')
        imageList = glob.glob(self.path + '/leftImg8bit/val/*/*.png')
        self.valData = pd.DataFrame(imageList, columns = ['input'])
        self.valData['output'] = self.valData['input'].apply(lambda x: self.path + '/gtFine_trainvaltest/gtFine/val/'+
                                             self.getCityName(x)+'/'+self.getImageName(x)+'_gtFine_labelIds.png')
        self.numberOfClasses = kwargs['numberOfClasses']
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
        self.name2label      = { label.name    : label for label in labels           }
        # id to label object
        self.id2label = { label.id : label for label in labels }
        self.dict = {
            0: 0, # Void
            1: 7, # EgoCar -> Car
            2: 0, # Reectification Border -> Void
            3: 0, # Out of Roi -> Void
            4: 0, # Static -> Void
            5: 17, # Dynamic -> Other Moving
            6: 22, # Ground -> Road Shoulder
            7: 18, # Road
            8: 2, # Sidewalk
            9: 14, # Parking
            10: 15, #rail track -> Railway
            11: 13, #Building -> Building/Wall
            12: 13, #Wall -> Building/Wall
            13: 10, #Fence
            14: 10, #Guard Rail -> Fence
            15: 4, # Bridge
            16: 6, # Tunnel
            17: 31, #Pole -> Column_Pole
            18: 31, #PoleGroup -> Column_Pole
            19: 3, #TrafficLight
            20: 28, #Traffic Sign -> SignSymbol
            21: 19, #Vegetation -> Tree
            22: 30, #Terrain -> Misc Vegetation
            23: 21, # Sky
            24: 9, #Person -> Pedestrian
            25: 5, # Rider -> Bicycle
            26: 7, #Car
            27: 29, #Truck -> Truck_Bus
            28: 29, #Bus -> Truck_Bus
            29: 12,# Caravan -> SUVPickupTruck
            30: 29,# Trailer -> Truck_Bus
            31: 26, #Train -> Train
            32: 25, # Motorcycle -> Motorcycle
            33: 5, # Bicycle -> Cyclist
            -1: 7, # Licence Plate -> Car
        }
        if self.numberOfClasses == 0:
            self.trainQueuer = GeneratorEnqueuer(self.generatorAllClasses(samplesPerBatch, True), False)
            self.trainQueuer.start(workers=1, max_q_size=10)
            self.valQueuer = GeneratorEnqueuer(self.generatorAllClasses(samplesPerBatch, False), False)
            self.valQueuer.start(workers=1, max_q_size=10)
        else:
            self.trainQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, True), False)
            self.trainQueuer.start(workers=1, max_q_size=10)
            self.valQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, False), False)
            self.valQueuer.start(workers=1, max_q_size=10)
        self.numberOfClasses = len(labels) if self.numberOfClasses == 0 else self.numberOfClasses
    
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
        inputShape = mpimg.imread(self.trainData.iloc[0]['input'])[::2,::2].shape
        data = self.trainData if train else self.valData
        while(1):
            imageArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
            labelArr = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            #weights = np.ones((batchSize, self.numberOfClasses))
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
                image = mpimg.imread(data.iloc[index]['input'])[::2,::2]
                targetName = data.iloc[index]['output']
                #print(targetName)
                target = mpimg.imread(targetName)[::2,::2]
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
            yield({'inputImg': imageArr}, {'outputImg': labelArr}, weights)
            yield({'inputImg': imageArr}, {'outputImg': labelArr}, weights)
            
    def generatorAllClasses(self, batchSize, train, **kwargs):
        start=0
        inputShape = mpimg.imread(self.trainData.iloc[0]['input'])[::2,::2].shape
        data = self.trainData if train else self.valData
        while(1):
            imageArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
            labelArr = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            #weights = np.ones((batchSize, self.numberOfClasses))
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
                #if(not train): print(imgList[index])
                image = mpimg.imread(data.iloc[index]['input'])[::2,::2]
                
                targetName = data.iloc[index]['output']
                #print(targetName)
                target = mpimg.imread(targetName)[::2,::2]
                target = (target*255).astype('uint8')
                #target[target==-1] = 0
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
                labelArr[i] = target
            yield({'inputImg': imageArr}, {'outputImg': labelArr}, weights)
            
    @staticmethod
    def convertTarget(image, dictonary):
        #print(image.shape)
        image = np.reshape(image, (int(image.shape[0]), int(image.shape[1]) , 1))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j,0] = dictonary[image[i,j,0]]
        return image
    
    @staticmethod
    def getSubTrainDistribution(df, numberOfClasses, dictionary):
        arr = np.zeros(numberOfClasses)
        for handle in df['output']:
            target = mpimg.imread(handle)[::2,::2]
            target = (target*255).astype('uint8')
            target[target == -1] = 0
            if(numberOfClasses < 30):
                target = CityScapes.convertTarget(target, dictionary)
            for val in target.ravel():
                arr[int(val)] += 1
        return arr
    
    def getTrainDistribution(self):
        subDfs = []
        # Zähle die Prozessoren
        cores = multiprocessing.cpu_count() 
        # Teile den DataFrame in cores gleichgrosse Teile
        for i in range(cores):
            subDfs.append(self.trainData.iloc[int(i/cores*len(self.trainData)):int((i+1)/cores*len(self.trainData)), :])
        print("Build SubDfs")
        # Öffne einen Pool mit ensprechend vielen Prozessen 
        pool = Pool(processes=cores)
        # Wende die Funktion an
        func = functools.partial(CityScapes.getSubTrainDistribution, numberOfClasses=self.numberOfClasses, dictionary=self.dict)
        arrs = pool.map(func, subDfs)
        pool.close()
        arr = np.zeros(self.numberOfClasses)
        for i in range(len(arrs)):
            arr += arrs[i]
            
        return arr
    
    
    def setWeights(self, weights):
        self.weights = weights
    
class CamVid(Dataset):
    
    def __init__(self, path, samplesPerBatch, **kwargs):
        super(CamVid, self).__init__(path=path)
        imageList = glob.glob(self.path + '/701_StillsRaw_full/*.png')
        imageList = [path.replace('\\', '/') for path in imageList]
        data = pd.DataFrame(imageList, columns = ['input'])
        data['output'] = data['input'].apply(lambda x: self.path + '/LabeledApproved_full' + x[x.rfind('/'):-4] + '_L.png')
        data['outputPickled'] = data['input'].apply(lambda x: self.path + '/LabeledApproved_full' + x[x.rfind('/'):-4] + '_L.pickle')
        self.trainData, self.valData = train_test_split(data, test_size = .1, random_state = 42)
        self.numberOfClasses = kwargs['numberOfClasses']
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
        self.trainQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, True), False)
        self.trainQueuer.start(workers=1, max_q_size=10)
        self.valQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, False), False)
        self.valQueuer.start(workers=1, max_q_size=10)
        
    def generator(self, batchSize, train, **kwargs):
        start=0
        inputShape = mpimg.imread(self.trainData.iloc[0]['input']).shape
        data = self.trainData if train else self.valData
        while(1):
            imageArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
            labelArr = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            #weights = np.ones((batchSize, self.numberOfClasses))
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
                    target = self.convertTarget(target)
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
            yield({'inputImg': imageArr}, {'outputImg': labelArr}, weights)
            
            
    def convertTarget(self, image):
        target = np.zeros((image.shape[0], image.shape[1], 1))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                label = self.treeDict[image[i,j,0]][image[i,j,1]][image[i,j,2]]
                #if (label == None):
                #    print(image[i,j])
                target[i,j,0] = label
        return target
    
    
    def getTrainDistribution(self):
        arr = np.zeros(self.numberOfClasses)
        for handle in self.trainData['outputPickled']:
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
        imageList = [path.replace('\\', '/') for path in imageList]
        data = pd.DataFrame(imageList, columns = ['input'])
        data['output'] = data['input'].apply(lambda x: self.path + '/gttest' + x[x.rfind('/'):])
        self.trainData, self.valData = train_test_split(data, test_size = .1, random_state = 42)
        self.numberOfClasses = kwargs['numberOfClasses']
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
        self.trainQueuer.start(workers=1, max_q_size=10)
        self.valQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, False), False)
        self.valQueuer.start(workers=1, max_q_size=10)
        
    def generator(self, batchSize, train, **kwargs):
        start=0
        inputShape = mpimg.imread(self.trainData.iloc[0]['input']).shape
        data = self.trainData if train else self.valData
        while(1):
            imageArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
            labelArr = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            #weights = np.zeros((batchSize, self.numberOfClasses))
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
            yield({'inputImg': imageArr}, {'outputImg': labelArr}, weights)
            
            
    def convertTarget(self, image):
        #print(image.shape)
        image = np.reshape(image, (int(image.shape[0]), int(image.shape[1]) , 1))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j,0] = self.dict[image[i,j,0]]
        return image
    
    def getTrainDistribution(self):
        arr = np.zeros(self.numberOfClasses)
        for handle in self.trainData['output']:
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
        imageList = [path.replace('\\', '/') for path in imageList]
        data = pd.DataFrame(imageList, columns = ['input'])
        data['output'] = data['input'].apply(lambda x: self.path + '/training/gt_image_2/um_lane' + x[x.rfind('_'):])
        self.trainData, self.valData = train_test_split(data, test_size = .1, random_state = 42)
        self.numberOfClasses = 2
        self.weights = None
        self.trainQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, True), False)
        self.trainQueuer.start(workers=1, max_q_size=10)
        self.valQueuer = GeneratorEnqueuer(self.generator(samplesPerBatch, False), False)
        self.valQueuer.start(workers=1, max_q_size=10)
        imageList = glob.glob(self.path + '/testing/image_2/*.png')
        self.testData = pd.DataFrame(imageList, columns = ['input'])
         
        
    def generator(self, batchSize, train, **kwargs):
        start=0
        inputShape = mpimg.imread(self.trainData.iloc[0]['input']).shape
        data = self.trainData if train else self.valData
        while(1):
            imageArr = np.zeros((batchSize, inputShape[0], inputShape[1], inputShape[2]))
            labelArr = np.zeros((batchSize, inputShape[0] * inputShape[1]))
            #weights = np.zeros((batchSize, self.numberOfClasses))
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
            yield({'inputImg': imageArr}, {'outputImg': labelArr}, weights)
            
            
    def convertTarget(self, image):
        #print(image.shape)
        target = np.zeros((int(image.shape[0]),int(image.shape[1])))
        image = np.reshape(image, (int(image.shape[0]), int(image.shape[1]) , 3))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                target[i,j] = 1 if np.any(image[i,j,0]) else 0
        return target
    
    def getTrainDistribution(self):
        arr = np.zeros(self.numberOfClasses)
        for handle in self.trainData['output']:
            target = mpimg.imread(handle)
            target = (target*255).astype('uint8')
            target = self.convertTarget(target)
            for val in target.ravel():
                arr[int(val)] += 1
        return arr
    
    def setWeights(self, weights):
        self.weights = weights
        
    
        
        