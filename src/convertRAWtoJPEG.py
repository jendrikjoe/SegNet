'''
Created on Jul 5, 2017

@author: jendrik
'''
import glob
import argparse
import numpy as np
from PIL import Image
from matplotlib import image as mpimg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'path',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()
    for dat in glob.glob(args.path):
        outname = './jpeg/' + dat.split('/')[-1][:-3] + '.jpeg'
        file = open(dat, 'rb')
        img = np.array(Image.frombytes('RGB', [960,480], file.read(), 'raw'))
        file.close()
        mpimg.imsave(outname, img, format='jpeg')