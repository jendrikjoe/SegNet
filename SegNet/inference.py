'''
Created on Jun 9, 2017

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
from skimage.util.shape import view_as_windows
from torch.nn import AvgPool2d as AvgPool2d


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
    anns = anns[upperLim:lowerLim:, leftLim:rightLim,2].reshape((1,1, lowerLim-upperLim, rightLim-leftLim))[:,:,::-1].copy()
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

def pipeline(image, model, colorDictInv, sess):
    image = cv2.resize(image, (960,720))
    inputShape = image.shape
    image = np.reshape(image, (1,inputShape[0],inputShape[1],inputShape[2])) / 255.
    print(np.max(image))
    return convertTargetToImage(np.reshape(model.evalWithAverage(image, sess), 
                        (inputShape[0], inputShape[1])), colorDictInv)

if __name__ == '__main__':
    #dataset = Mapillary('/media/jendrik/DataSwap1/Datasets/mapillary-vistas/', numberOfClasses=15)
    segNet = SegNet(15, dropProb = .2, ckptFile = '../15ClassesNew.ckpt')
    segNet.half()
    segNet.cuda()
    ls = glob.glob('/media/jendrik/DataSwap1/Thunderhill Data/1610/jpeg/*')
    #ls = ['/media/jendrik/DataSwap1/Thunderhill Data/1610/jpeg/15395510-1490933073639359.jpeg']
    ls.sort()
    polyArr = []
    for path in ls:
        img = mpimg.imread(path)[::-1]
        dat = img.reshape((-1, img.shape[0], img.shape[1], img.shape[2]))
        dat = dat / 255.
        dat = Variable(torch.from_numpy(dat)).half().permute(0,3,1,2)
        start_time = time.time()
        resS = segNet.evalWithAvg(dat.cuda()).data.cpu().numpy()[0]
        print("--- Eval Time: %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        #label, num_label = ndimage.label(resS == 1)
        #size = np.bincount(label.ravel())
        #biggest_label = size[1:].argmax() + 1
        #clump_mask = label == biggest_label
        tmp = (resS == 14)
        #res = res[:, 1:] & res[:, :-1]
        #res = res[:, 1:] & res[:, :-1]
        #res = res[:, 1:] & res[:, :-1]
        res = cv2.addWeighted(np.dstack((np.zeros_like(resS), tmp*255, np.zeros_like(resS))).astype(np.uint8), 
                              1, np.dstack(((resS==1)*255, np.zeros_like(resS), np.zeros_like(resS))).astype(np.uint8), 
                              1, 0)
        filteredAnnotation = filterRoad(res)
        filteredAnnotation = cv2.addWeighted(filteredAnnotation, 
                              1, np.dstack((np.zeros_like(resS), np.zeros_like(resS), (resS==8)*255)).astype(np.uint8), 
                              1, 0)
        res = filterByCones(filteredAnnotation)
        print("--- Filter Annotation Time: %s seconds ---" % (time.time() - start_time))
        mpimg.imsave('/media/jendrik/DataSwap1/Thunderhill Data/1610/annotation/'+path.split('/')[-1].split('.')[0]+'.png', res)
        res = cv2.addWeighted(img, .5, res, 0.5, 0)
        mpimg.imsave('/media/jendrik/DataSwap1/Thunderhill Data/1610/overlay/'+path.split('/')[-1].split('.')[0]+'.png', res)
        
        start_time = time.time()
        res, polyArr = medianTrajectory(img, filteredAnnotation, polyArr)
        print("--- Median Trajectory Time: %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        mpimg.imsave('/media/jendrik/DataSwap1/Thunderhill Data/1610/trajectory/'+path.split('/')[-1].split('.')[0]+'.png', res)
        print("--- Image Write Time: %s seconds ---" % (time.time() - start_time))
        #quit()
    image = mpimg.imread('./lastYearImage.jpeg')[::-1,::1]
    image2 = mpimg.imread('./complexImage.jpeg')[::-1,::1]
    for text, img in zip(['lastYearImage', 'complexImage'], [image, image2]):
        img = img.reshape((-1, img.shape[0], img.shape[1], img.shape[2]))
        img = img/255.
        img = Variable(torch.from_numpy(img)).half().permute(0,3,1,2)
        start_time = time.time()
        resS = segNet.eval(img.cuda()).data.cpu().numpy()[0]
        print("--- %s seconds ---" % (time.time() - start_time))
        #outImg = dataset.convertTargetToImage(res)
        #mpimg.imsave('./' + text + 'Out.png', outImg)
        tmp = (resS == 14)#[1:] & (res == 14)[:-1]
        res = tmp[:, 1:] & tmp[:, :-1]
        res = res[:, 1:] & res[:, :-1]
        res = res[:, 1:] & res[:, :-1]
        res = res[:, 1:] & res[:, :-1]
        res = res[:, 1:] & res[:, :-1]
        res = res[:, 1:] & res[:, :-1]
        #res=tmp
        res = np.pad(res, ((0,0),(3,3)), 'constant', constant_values=0)
        
        res = np.dstack((res*255, np.zeros_like(res), np.zeros_like(res))).astype(np.uint8)
        mpimg.imsave('./' + text + 'Lanes.png', res)
        
        res = np.dstack(((np.bitwise_or(resS==1, resS==6))*255, np.zeros_like(resS), np.zeros_like(resS))).astype(np.uint8)
        mpimg.imsave('./' + text + 'Road.png', res)
    