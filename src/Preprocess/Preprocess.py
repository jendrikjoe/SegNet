'''
Created on Feb 19, 2017

@author: jjordening
'''

import cv2
import numpy as np
import time
from PIL.ImageFilter import BLUR

def minValImage(arr, channel = 0):
    """
    Determines the minimum value of arr in a channel
    """
    return np.min(np.min(np.min(arr[:,:,:,channel], axis=1),axis=1),axis=0)

def maxValImage(arr, channel = 0):
    """
    Determines the maximium value of arr in a channel
    """
    return np.max(np.max(np.max(arr[:,:,:,channel], axis=0),axis=0),axis=0)

def addGrayLayer(image):
    """
    Adds a gray layer as a fourth channel to an image
    
    Input: 
        image
        
    Output:
        an array of images with the channels RGBGray
    """
    gray = cv2.cvtColor(image, cv2.CV_HLS2GRAY)
    return np.concatenate((image, gray.reshape((image.shape[0], image.shape[1],1))), axis=2)

def addGradientLayer(image, sobel_kernel=3, magThresh=(0, 255), dirThresh=(0, np.pi/2)):
    """
    Adds a gray layer as a fourth channel to an image
    
    Input: 
        image
        
    Output:
        an array of images with the channels RGBGray
    """
    sobelx = cv2.Sobel(image[:,:,1], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image[:,:,1], cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradMag = np.sqrt(sobelx**2 + sobely**2)
    absGradDir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Rescale to 8 bit
    scaleFactor = np.max(gradMag)/255
    gradMag = (gradMag/scaleFactor).astype(np.uint8)
    # Create a binary image of ones where thresholds are met, zeros otherwise
    binaryOutput = np.zeros_like(gradMag)
    binaryOutput[(gradMag >= magThresh[0]) & (gradMag <= magThresh[1]) &
                    (absGradDir >= dirThresh[0]) & (absGradDir <= dirThresh[1])] = 1 
    # Return the binary image
    return np.concatenate((image, binaryOutput.reshape((image.shape[0], image.shape[1],1))) ,axis=2)
    
    

def applyNormalisation(image):
    """
        Applies a normalisation to an image with the channels RGBGray.
        It applies a CLAHE normalisation to the gray layer and then normalises the
        values such, that they have a mean of 0 and a deviation of 1
        
        Input: 
            image
        
        Output:
            an array of images with the channels RGBGray
    """
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #image[:,:,3] = clahe.apply(image[:,:,3])
    return image/ 255.

def convertRGBToHSV(img):
    temp = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = img[i,j,0]
            g = img[i,j,1]
            b = img[i,j,2]
            maximum = np.max([r, g, b])
            minimum = np.min([r, g, b])
            if(maximum == minimum):
                temp[i,j,0] = 0
            elif(maximum == r):
                temp[i,j,0] = 1./6.*(g - b)/(maximum-minimum)
            elif(maximum == g):
                temp[i,j,0] = 1./6.*(2+(b - r)/(maximum-minimum))
            else :
                temp[i,j,0] = 1./6.*(4+(r - g)/(maximum-minimum))
            if(temp[i,j,0] < 0): temp[i,j,0] += 1.
            temp[i,j,1] = 0 if maximum==0 else (maximum-minimum)/maximum
            temp[i,j,2] = maximum
    return temp

def convertHSVToRGB(img):
    temp = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            f = img[i,j,0]*6.
            hi = int(f)
            f -= hi
            p = img[i,j,2]*(1.- img[i,j,1])
            q = img[i,j,2]*(1.- f*img[i,j,1])
            t = img[i,j,2]*(1.- (1.-f)*img[i,j,1])
            if(hi == 0 or hi==6): temp[i,j] = [img[i,j,2], t, p]
            elif(hi == 1): temp[i,j] = [q, img[i,j,2], p]
            elif(hi == 2): temp[i,j] = [p, img[i,j,2], t]
            elif(hi == 3): temp[i,j] = [p, q, img[i,j,2]]
            elif(hi == 4): temp[i,j] = [t, p, img[i,j,2]]
            else : temp[i,j] = [img[i,j,2], p, q]
    return temp

def augmentImageAndLabel(image, label):
    image = lightImage(image)
    shiftHor = np.random.randint(-30,31)
    shiftVer = np.random.randint(-30,31)
    image = shiftImg(image, shiftHor, shiftVer)
    image = blurImage(image)
    label = shiftImg(label, shiftHor, shiftVer)
    
    return image, label

def augmentImage(image, label):
    #rot = int(np.random.rand()**2*90 -45)
    rot = int(np.random.rand()**2*60 - 30)
    image = rotateImage(image, rot)
    # Add a part of the rotated angle, as it is counted counter-clockwise.
    # If you turn counter-clockwise, this looks like the car would be more left
    # and needs to drive to the right -> add some angle 
    # divide it by the maximum of the steering angle in deg ->25
    label[0] -= 3*rot/(30)
    label[1] -= .2*rot/20
    #label[2] += .1*rot/10
    shiftHor = np.random.randint(-40,41)
    shiftVer = np.random.randint(-5,6)
    image = shiftImg(image, shiftHor, shiftVer)
    #label[0] *= (1-shiftVer/100)
    label[0] += 2.5*shiftHor/(40)
    label[1] -= .3*shiftHor/40
    #label[2] += .1*shiftHor/40steering
    label[0] = min(max(label[0], -5),5)
    label[1] = min(max(label[1], -1),1)
    label[2] = min(max(label[2], -1),1)
    #image = lightImage(blurImage(image))
    return image, label

def blurImage(img):
    # Blur image with random kernel
    kernel_size = np.random.randint(1, 5)
    if kernel_size % 2 != 1:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def lightImage(img):
    # HSV brightness transform
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    brightness = np.random.uniform(0.8,1.1)
    img[:,:,2] = img[:,:,2]*brightness
    np.clip(img, 0, 1, img)
    return cv2.cvtColor(img,cv2.COLOR_HSV2RGB)


def preprocessImage(image):
    """
    This function represents the default preprocessing for 
    an image to prepare them for the network
    """
    #image = cv2.resize(image, (320, 160))[::-1]
    #image = image[image.shape[0]*1//5:image.shape[0]*7//8,:,:]
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #image = cv2.convertScaleAbs(image, alpha=(1))
    #image = addGradientLayer(image, 7, (100,255), (0, np.pi/2))
    image = applyNormalisation(image)
    return image

def preprocessImages(arr):
    """
    This function represents the default preprocessing for 
    images to prepare them for the network
    """
    return np.array([preprocessImage(image) for image in arr])


def perspectiveTransform(img, M):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
        

def shiftImg(arr, horizontal, vertical):
    if len(arr.shape) == 2:
        if(horizontal>0):arr = np.concatenate((arr[horizontal:,:],np.zeros((horizontal,arr.shape[1]))), axis=0)
        elif(horizontal<0):arr = np.concatenate((np.zeros((np.abs(horizontal),arr.shape[1])), arr[:horizontal,:]), axis=0)
        if(vertical>0):arr = np.concatenate((arr[:,vertical:],np.zeros((arr.shape[0],vertical))), axis=1)
        elif(vertical<0):arr = np.concatenate((np.zeros((arr.shape[0], np.abs(vertical))), arr[:,:vertical]), axis=1)
   
    else:
        if(horizontal>0):arr = np.concatenate((arr[horizontal:,:],np.zeros((horizontal,arr.shape[1],arr.shape[2]))), axis=0)
        elif(horizontal<0):arr = np.concatenate((np.zeros((np.abs(horizontal),arr.shape[1],arr.shape[2])), arr[:horizontal,:]), axis=0)
        if(vertical>0):arr = np.concatenate((arr[:,vertical:],np.zeros((arr.shape[0],vertical,arr.shape[2]))), axis=1)
        elif(vertical<0):arr = np.concatenate((np.zeros((arr.shape[0], np.abs(vertical),arr.shape[2])), arr[:,:vertical]), axis=1)
    return arr

"""def shiftImg(arr, horizontal, vertical):
    
        This function shifts an image horizontally and vertically
        Input:
            horizontal - amplitude of shift in pixels (positive to the right
            negative to the left)
            vertical - aplitude of the ishift in pixels (positive downwards 
            negative upwards)
    
    
    M = np.float32([[1,0,1*horizontal],[0,1,vertical]])
    return cv2.warpAffine(arr,M,(arr.shape[1], arr.shape[0]))"""

def mirrorImage(img):
    """
        This function mirrors the handed image around the y-axis
    """
    return img[:,::-1]

def rotateImage(img, angle):
    """
        Rotates image around the point in the middle of the bottom of the picture by
        angle degrees.
    """
    rotation = cv2.getRotationMatrix2D((img.shape[0], img.shape[1]//2), angle, 1)
    return cv2.warpAffine(img, rotation, (img.shape[1], img.shape[0]))
    
def rotateImages(arr, angles):
    """
        Rotates multiple images by the given angles.
    """
    arr = [rotateImage(img, angle) for img, angle in zip(arr, angles)]
    return arr

