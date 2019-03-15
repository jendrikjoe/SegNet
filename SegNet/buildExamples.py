'''
Created on Apr 2, 2018

@author: jendrik
'''
import numpy as np
from Network.Network import SegNet
from torch.autograd import Variable
from Data.Datasets import Mapillary
import matplotlib.image as mpimg
import torch
import time
import glob
from scipy import ndimage
import cv2
from torch.nn import AvgPool2d as AvgPool2d
import imageio

from collections import namedtuple  

MEDIAN_THRESHOLD = 50

def runsOfZerosArray(bits, roadBits):
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits > 0, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    runStarts = np.where(difs < 0)[0]
    runEnds = np.where(difs > 0)[0]
    runStarts = np.insert(runStarts, 0, 0)
    runEnds = np.append(runEnds, len(bits)-1)
    if len(runStarts) == 0:
        return [], 0, 0 
    #print(runStarts, runEnds)
    lengths = runEnds - runStarts
    pixelLengths = []
    for start, end in zip(runStarts, runEnds):
        pixelLengths.append(np.sum(roadBits[start:end] > 0))
    pixelLengths = np.array(pixelLengths)
    idx = np.argmax(pixelLengths)
    return runStarts, idx, lengths[idx], pixelLengths[idx]

def filterRoad(annotation):
    upperLim = annotation.shape[0]//2 - 30
    lowerLim = annotation.shape[0] - MEDIAN_THRESHOLD
    avgRegion = 7
    pad = (avgRegion-1) // 2
    
    anns = annotation.astype(np.float32)
    anns = Variable(torch.from_numpy(anns[upperLim:lowerLim,:,1].reshape((1,1, lowerLim-upperLim, anns.shape[1])))).cuda()
    meanMap = AvgPool2d(kernel_size=avgRegion, stride=1, padding=pad)(anns).cpu().data.numpy().reshape((lowerLim-upperLim, annotation.shape[1]))
    #meanMap = view_as_windows(annotation[upperLim:lowerLim,:,1], (avgRegion,avgRegion)).mean(axis=(-2,-1))
    #meanMap = np.pad(meanMap, ((pad,pad), (pad,pad)), 'constant', constant_values=(0,0))
    #print(annotation.shape, meanMap.shape)
    
    #print(np.unique(meanMap))
    leftIndices = None
    rightIndices = None
    lastStart = -1
    lastEnd = -1
    startTol = 20
    endTol = 20
    
    for i in range(meanMap.shape[0])[::-1]:
        starts, idx, length, pixelLengths = runsOfZerosArray(meanMap[i], annotation[upperLim+i,:,0])
        
        if lastStart != -1:
            if len(starts) == 0 or pixelLengths < 50: continue
            if starts[idx] == 0:
                if np.all(leftIndices[:,1] == 0):
                    leftIndices = np.append(leftIndices, [[i + upperLim, starts[idx]]], axis=0)
                    lastStart = starts[idx]
                    startTol = 200
            elif np.abs(starts[idx] - lastStart) < startTol:
                leftIndices = np.append(leftIndices, [[i + upperLim, starts[idx]]], axis=0)
                lastStart = starts[idx]
                startTol = 30
            else: startTol += 10
            if starts[idx] + length == annotation.shape[1] -1:
                if np.all(rightIndices[:,1] == annotation.shape[1] -1) : 
                    rightIndices = np.append(rightIndices, [[i + upperLim, starts[idx]+length]], axis=0)
                    lastEnd = starts[idx] + length
                    endTol = 200
            elif np.abs(starts[idx] + length - lastEnd) < endTol:
                rightIndices = np.append(rightIndices, [[i + upperLim, starts[idx] + length]], axis=0)
                lastEnd = starts[idx] + length
                endTol = 30
            else: endTol += 5
        else:
            leftIndices = np.array([[i + upperLim, starts[idx]]]) 
            rightIndices = np.array([[i + upperLim, starts[idx] + length]])
            lastStart = starts[idx]
            lastEnd = starts[idx] + length
    leftIndices = np.array(leftIndices)
    rightIndices = np.array(rightIndices)
    if(len(leftIndices) == 0 or len(rightIndices) == 0): return annotation
    leftIndicesY = np.arange(np.min(leftIndices[:,0]), np.max(leftIndices[:,0]), 1)
    leftIndicesX = np.interp(leftIndicesY, 
                            leftIndices[:,0][::-1], leftIndices[:,1][::-1])
    
    rightIndicesY = np.arange(np.min(rightIndices[:,0]), np.max(rightIndices[:,0]), 1)
    rightIndicesX = np.interp(rightIndicesY, 
                            rightIndices[:,0][::-1], rightIndices[:,1][::-1])
    for x, y in zip(rightIndicesX, rightIndicesY):
        annotation[int(y), int(x):, 0] = 0
        #annotation[int(y), :int(x), 2] = 255
    for x, y in zip(leftIndicesX, leftIndicesY):
        annotation[int(y), :int(x), 0] = 0
        #annotation[int(y), :int(x), 2] = 0
    return annotation


def filterByCones(annotation):
    upperLim = annotation.shape[0] // 4
    lowerLim = annotation.shape[0] *6 //10
    sideIndent = 150
    leftLim = sideIndent
    rightLim = annotation.shape[1] - sideIndent
    
    avgRegion = (21,5)
    pad = (10,2)
    
    anns = annotation.astype(np.float32)
    anns = anns[upperLim:lowerLim:,leftLim:rightLim,2].reshape((1,1, lowerLim-upperLim, rightLim-leftLim))[:,:,::-1].copy()
    anns = Variable(torch.from_numpy(anns)).cuda()
    meanMap = AvgPool2d(kernel_size=avgRegion, stride=1, padding=pad)(anns)
    minIndex = torch.max(torch.max(meanMap, 3)[0],2)
    if minIndex[0].cpu().data.numpy() < 254:
        return annotation
    minIndex = int(minIndex[1].cpu().data.numpy())
    annotation[:lowerLim-minIndex,:,0] = 0
    return annotation
    
    
def medianTrajectory(image, annotation, polyArr):
    upperLim = annotation.shape[0] // 2- MEDIAN_THRESHOLD
    lowerLim = annotation.shape[0] - MEDIAN_THRESHOLD
    
    image = image.copy()
    roadIndices = []
    for i in np.arange(upperLim, lowerLim, 1):
        indices = np.where(annotation[i, :, 0] > 0)
        indices = indices[0]
        if len(indices) > 20:
            roadIndices.append([i, int(np.median(indices))])

    roadIndices = np.array(roadIndices)
    
    filteredRoadIndices = [[annotation.shape[0]- MEDIAN_THRESHOLD, annotation.shape[1]//2]]
    filterLen = 3
    
    for i in range(len(roadIndices)-filterLen*2):
        val = 0
        for j in np.arange(-1*filterLen,filterLen,1):
            val += roadIndices[i-j,1]
        val //= filterLen * 2 + 1
        filteredRoadIndices.append([roadIndices[i,0], val])
    filteredRoadIndices = np.array(filteredRoadIndices)
    weights = np.ones(len(filteredRoadIndices))
    weights[0] = 1e9
    
    z = np.polyfit(filteredRoadIndices[:,0], filteredRoadIndices[:,1], 2, w=weights)
    if len(polyArr) == 0:
        polyArr = z
    else:
        polyArr = .9 * polyArr + .1 * z
    p = np.poly1d(polyArr)
            
    
    for y in np.arange(upperLim, lowerLim, 1):
        if p(y) > 0 and p(y) < 960:
            image[y, int(p(y)), 0] = 255
            image[y, int(p(y))+1, 0] = 255
            image[y, int(p(y))-1, 0] = 255
            image[y, int(p(y)), 1] = 0
            image[y, int(p(y))+1, 1] = 0
            image[y, int(p(y))-1, 1] = 0
            image[y, int(p(y))-1, 0] = 0
            image[y, int(p(y)), 2] = 0
            image[y, int(p(y))+1, 2] = 0
            image[y, int(p(y))-1, 2] = 0
    
    for arr in filteredRoadIndices:
        image[arr[0], arr[1], 0] = 255
    return image, polyArr

def convertTargetToImage(target, colourDictInv):
    image = np.zeros((target.shape[0], target.shape[1], 3))
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            #if (label == None):
            #    print(image[i,j])
            if(i == 719 and j == target.shape[1]//2): print(target[i,j],
                     target[i,j], colourDictInv.get(target[i,j]))
            image[i,j, :] = colourDictInv.get(target[i,j])[:]
    return image.astype('uint8')

if __name__ == '__main__':
    
    ls = glob.glob('/home/jendrik/tempImgs/tra/*')
    ls.sort()
    with imageio.get_writer('./race.mp4', mode='I') as writer:
        for filename in ls:
            seg = mpimg.imread('/home/jendrik/tempImgs/seg/'+filename.split('/')[-1])*255.
            image = mpimg.imread(filename)*255.
            out = np.zeros((image.shape[0], image.shape[1]*2, image.shape[2]))
            out[:,:image.shape[1]] = image
            out[:,image.shape[1]:] = seg
            writer.append_data(out)
    quit()
    ls = glob.glob('/media/jendrik/DataSwap/Thunderhill Data/1610/jpeg/*')
    ls.sort()
    with imageio.get_writer('./movie.mp4', mode='I') as writer:
        for filename in ls:
            image = mpimg.imread(filename)[::-1]
            writer.append_data(image)
    quit()
    #dataset = Mapillary('/media/jendrik/DataSwap/Datasets/mapillary-vistas/', numberOfClasses=15)
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
    labelsThunderhill = [
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
    
    colorDict = { label.id : label.color for label in labelsThunderhill}
    segNet = SegNet(15, dropProb = .2, ckptFile = '../15ClassesNew.ckpt')
    segNet.half()
    segNet.cuda()
    img = mpimg.imread('./testImg.jpg')
    lab = (mpimg.imread('./testImgLabels.png')*255).astype(np.uint8)
    dat = img.reshape((-1, img.shape[0], img.shape[1], img.shape[2]))
    dat = dat / 255.
    dat = Variable(torch.from_numpy(dat), requires_grad=False).half().permute(0,3,1,2)
    res = cv2.addWeighted(img, .5, lab, .5, 0)
    mpimg.imsave('./testImgOverlay.png', res)
    resS = segNet.eval(dat.cuda()).data.cpu().numpy()[0]
    resS = convertTargetToImage(resS, colorDict)
    res = cv2.addWeighted(img, .5, resS, .5, 0)
    mpimg.imsave('./testImgOut.png', res)
    
    #ls = ['/media/jendrik/DataSwap1/Thunderhill Data/1610/jpeg/15395510-1490933073639359.jpeg']
    polyArr = []
    writerSeg = imageio.get_writer('./segMovie.mp4', mode='I')
    writerFilt = imageio.get_writer('./filtMovie.mp4', mode='I')
    writerTra = imageio.get_writer('./traMovie.mp4', mode='I')
    for path in ls:
        img = mpimg.imread(path)[::-1]
        dat = img.reshape((-1, img.shape[0], img.shape[1], img.shape[2]))
        dat = dat / 255.
        dat = Variable(torch.from_numpy(dat), requires_grad=False).half().permute(0,3,1,2)
        start_time = time.time()
        resS = segNet.evalWithAvg(dat.cuda()).data.cpu().numpy()[0]

        tmp = (resS == 14)
        
        res = np.dstack(((resS==1)*255, tmp*255, (resS==8)*255)).astype(np.uint8)
        
        out = np.zeros((res.shape[0], res.shape[1]*2, res.shape[2]))
        out[:,:res.shape[1]] = img
        out[:,res.shape[1]:] = res
        writerSeg.append_data(out)
        
        res = filterByCones(res)
        filteredAnnotation = filterRoad(res)
        
        out[:,:res.shape[1]] = img
        out[:,res.shape[1]:] = filteredAnnotation
        writerFilt.append_data(out)
        
        res, polyArr = medianTrajectory(img, filteredAnnotation, polyArr)
        
        out[:,:res.shape[1]] = res
        out[:,res.shape[1]:] = filteredAnnotation
        writerTra.append_data(out)
        
    writerSeg.close()
    writerFilt.close()
    writerTra.close()
    
    
    