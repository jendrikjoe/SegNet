'''
Created on Jun 8, 2017

@author: jendrik
'''
from SegNet.Network.Network import SegNet, SegNetDepth
import numpy as np
import pickle
from matplotlib import pyplot as plt
from SegNet.Data.Datasets import Mapillary, CamVid, ApolloScape
import torch
from torch.autograd import Variable
from SegNet.Network.Trainer import SegNetTrainer, SegNetDepthTrainer
import functools
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import argparse

SMALLNET = False


def convert_target_to_image(target, colour_dict_inv):
    image = np.zeros((target.shape[0], target.shape[1], 3))
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if i == 719 and j == target.shape[1] // 2:
                print(target[i, j], np.argmax(target[i, j]),
                      colour_dict_inv.get(np.argmax(target[i, j])))
            image[i, j, :] = colour_dict_inv.get(target[i, j])[:]
    return image.astype('uint8')


def inverstigate_datasets(datasets, classDict):
    arr = np.zeros(datasets[0].number_of_classes)
    for data in datasets:
        arr += data.get_train_distribution()
    # names = []
    # for i in range(datasets[0].number_of_classes):
    #    names.append(classDict.get(i))     
    plt.plot(range(datasets[0].number_of_classes), arr)
    # plt.xticks(range(datasets[0].number_of_classes), names, size='small', rotation='vertical')
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

    parser = argparse.ArgumentParser(description="Train SegNet")

    parser.add_argument("--mapillary-path", help=f"Path to the mapillary dataset", type=str, default='')
    parser.add_argument("--apollo-path", help=f"Path to the apollo dataset", type=str, default='')

    parser.add_argument("--cam-vid-path", help=f"Path to the CamVid dataset", type=str, default='')

    args = parser.parse_args()

    colorDictInv = {v: k for k, v in colorDict.items()}
    mapillary = Mapillary(args.mapillary_path, number_of_classes=15)
    apollo = ApolloScape(args.apollo_path, number_of_classes=15)
    # camVid = CamVid('/media/jendrik/DataSwap1/Datasets/CamVid/', number_of_classes=15)
    number_of_classes = mapillary.number_of_classes
    datasets = [apollo, mapillary]

    if 0:
        weights = inverstigate_datasets(datasets, classDict)
        with open('normWeights.pickle', 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('normWeights.pickle', 'rb') as handle:
            weights = pickle.load(handle)
        plt.plot(range(mapillary.number_of_classes), weights)
        plt.xticks(range(mapillary.number_of_classes), 
                   [mapillary.id2label[i].name for i in range(mapillary.number_of_classes)],
                   rotation=90)
        plt.subplots_adjust(bottom=0.3)
        
        #plt.show()
    print(weights)
    #weights = np.array([1 if val < np.e else 1/np.log(val) for val in weights])
    weights = np.array([1/np.log(val/1e7) if val > 1e7*np.e else 1 for val in weights])
    plt.plot(range(mapillary.number_of_classes), weights)
    plt.xticks(range(mapillary.number_of_classes),
               [mapillary.id2label[i].name for i in range(mapillary.number_of_classes)],
               rotation=90)    
    plt.subplots_adjust(bottom=0.3)
    #plt.show()
    print(weights)
    #mapillary.setWeights(weights)
    
    patience = 30 
    num_epochs = 220
    batches_per_epoch = 1024
    batches_per_val = 1024
    
    
    """shadowVars = {}
    for var in tf.get_collection('movingVars'):
        shadowVars.update({ema.average_name(var) : var})
    print(shadowVars)"""
    for dataset in datasets:
        dataset.launch_generators(max_q_size=30, max_proc=mp.cpu_count()//2)
    data_probs = []
    saveGraph = True
    print("Start building Segnet!")
    seg_net = SegNetDepth(number_of_classes, dropProb=.3)
    device = 'cuda:1'
    seg_net.to(device)

    for dataset in datasets:
        if len(data_probs) == 0:
            data_probs.append(len(dataset.train_data))
        else:
            data_probs.append(len(dataset.train_data) + sum(data_probs))
    optim_part = functools.partial(optim.SGD, momentum=0.9)
    lambda_f = lambda epoch: 0.99 ** epoch
    trainer = SegNetDepthTrainer(seg_net, optim.Adam, 1e-4, weight_decay=1e-2,
                                 scheduler_class=LambdaLR, log_path='./logs/', log_prefix='',
                                 weights=Variable(torch.from_numpy(weights)).float().to(device),
                                 ckpt_file=None, lr_lambda=[lambda_f])
    print("Everything Setup!")
    trainer.auto_train(num_epochs, datasets, data_probs, batches_per_epoch,
                       batches_per_val, patience, dist=None)
    switch = False

    print("Finished")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
