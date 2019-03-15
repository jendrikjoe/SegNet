'''
Created on Jun 9, 2017

@author: jendrik
'''
import numpy as np
from Network.Network import SegNet
from torch.autograd import Variable
from Data.Datasets import Mapillary
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import torch
import cv2

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
    #dataset = Mapillary('/media/jendrik/DataSwap1/Datasets/mapillary-vistas/', numberOfClasses=15)
    #segNet = SegNet(dataset.numberOfClasses, dropProb = .2, ckptFile = '../15Classes.ckpt')
    #segNet.cuda()
    """image = mpimg.imread('./lastYearImage.jpeg')[::-2,::-2]
    image = image/255.
    print(np.min(image))
    print(np.max(image))
    
    orgPoints = np.array([[400., 360.],
                         [560., 360.],
                         [960., 720.],
                         [0.,720.]])
    
    targPoints = np.array([[0., 0.],
                         [960., 0.],
                         [960., 720.],
                         [0.,720.]])
    M = cv2.getPerspectiveTransform(orgPoints, targPoints)
    warped = cv2.warpPerspective(image, M, (960, 720))
    plt.figure()
    plt.imshow(warped)
    plt.show()
    image = image.reshape((-1, image.shape[0], image.shape[1], image.shape[2]))
    image = image/255.
    image = Variable(torch.from_numpy(image)).float().permute(0,3,1,2)
    res = dataset.convertTargetToImage(segNet.eval(image.cuda()).data.cpu().numpy()[0])
    mpimg.imsave('./testOut.png', res)"""
    """image = mpimg.imread('./complexImageLanes.png')
    print(np.unique(image))
    print(image.shape)
    leftIndices = []
    rightIndices = []
    maxYL = 0
    maxYR = 0
    for i in np.arange(image.shape[0]//2-50, image.shape[0]-30, 1):
        indices = np.where(image[i, :480, 0] >0)
        indices = indices[0]
        if len(indices)>0: 
            leftIndices.append([i, np.max(indices)])
        indices = np.where(image[i, 480:, 0] >0)
        indices = indices[0]
        if len(indices)>0: 
            rightIndices.append([i, np.min(indices)+480])
    leftIndices = np.array(leftIndices)
    rightIndices = np.array(rightIndices)
    
    z = np.polyfit(leftIndices[:,0], leftIndices[:,1], 2)
    z2 = np.polyfit(rightIndices[:,0], rightIndices[:,1], 2)
    
    print(z)
    print(z2)
    image = np.zeros_like(image)
    image = image[:,:,:3]
    image = image.astype(np.uint8)
    print(image.shape)
    for arr in leftIndices:
        image[arr[0], arr[1], 0] = 255
    for arr in rightIndices:
        image[arr[0], arr[1], 0] = 255
    
    mpimg.imsave('./innermostPixels.png', image)
    image = mpimg.imread('./complexImage.jpeg').copy()[::-1]
    p = np.poly1d(z)
    p2 = np.poly1d(z2)
    print(image.shape)
    for y in np.arange(image.shape[0]//2, image.shape[0], 1):
        if p(y) > 0 and p(y) < 480:
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
    for y in np.arange(image.shape[0]//2, image.shape[0], 1):
        if p2(y) < 959 and p2(y) > 480:
            image[y, int(p2(y)), 0] = 255
            image[y, int(p2(y))+1, 0] = 255
            image[y, int(p2(y))-1, 0] = 255
            image[y, int(p2(y))+1, 1] = 0
            image[y, int(p2(y))-1, 1] = 0
            image[y, int(p2(y))-1, 0] = 0
            image[y, int(p2(y)), 2] = 0
            image[y, int(p2(y))+1, 2] = 0
            image[y, int(p2(y))-1, 2] = 0
            
    
    mpimg.imsave('./lines.png', image)"""
    
    
    image = mpimg.imread('./complexImageRoad.png')
    roadIndices = []
    for i in np.arange(image.shape[0]//2-80, image.shape[0]-30, 1):
        indices = np.where(image[i, :, 0] >0)
        indices = indices[0]
        if len(indices)>20: 
            roadIndices.append([i, int(np.median(indices))])
    roadIndices = np.array(roadIndices)
    
    image = np.zeros_like(image)
    image = image[:,:,:3]
    image = image.astype(np.uint8)
    print(image.shape)
    print(roadIndices)
    for arr in roadIndices:
        image[arr[0], arr[1], 0] = 255
    mpimg.imsave('./medianPixels.png', image)
    
    image = mpimg.imread('./complexImage.jpeg').copy()[::-1]
    z = np.polyfit(roadIndices[:,0], roadIndices[:,1], 2)
    p = np.poly1d(z)
            
    for y in np.arange(image.shape[0]//2-50, image.shape[0], 1):
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
    
            
    
    mpimg.imsave('./trajectory.png', image)
    
    
    
    
    
    
    
    
    
    
    
    